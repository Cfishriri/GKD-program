from dataclasses import dataclass

import torch

from .prompts import extract_framework, format_rollout_framework_prompt, format_student_prompt


@dataclass
class GuidedRollout:
    question: str
    framework: list[str]
    student_prompt: str
    completion: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def _generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=0.9 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True)


def generate_guided_rollout(
    teacher,
    student,
    tokenizer,
    question: str,
    *,
    framework_max_new_tokens: int,
    solution_max_new_tokens: int,
    temperature: float,
) -> GuidedRollout:
    framework_prefix = format_rollout_framework_prompt(question)
    framework_text = "<framework>\n" + _generate_text(
        teacher, tokenizer, framework_prefix, framework_max_new_tokens, temperature
    )
    framework = extract_framework(framework_text)
    student_prompt = format_student_prompt(question, framework)
    completion = _generate_text(student, tokenizer, student_prompt, solution_max_new_tokens, temperature)

    prompt_ids = tokenizer(student_prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (not completion_ids or completion_ids[-1] != tokenizer.eos_token_id):
        completion_ids.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100] * len(prompt_ids) + completion_ids], dtype=torch.long)
    return GuidedRollout(
        question=question,
        framework=framework,
        student_prompt=student_prompt,
        completion=completion,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

