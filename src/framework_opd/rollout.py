from dataclasses import dataclass

import torch

from .prompts import (
    extract_framework,
    format_rollout_framework_prompt,
    format_student_prompt,
    format_vanilla_student_prompt,
)


@dataclass
class GuidedRollout:
    question: str
    framework: list[str]
    student_prompt: str
    completion: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def validate_mode(mode: str) -> str:
    if mode not in {"vanilla", "guided"}:
        raise ValueError("mode must be 'vanilla' or 'guided'")
    return mode


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
    framework = generate_framework(
        teacher, tokenizer, question, max_new_tokens=framework_max_new_tokens, temperature=temperature
    )
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


def generate_framework(teacher, tokenizer, question: str, *, max_new_tokens: int, temperature: float) -> list[str]:
    framework_prefix = format_rollout_framework_prompt(question)
    framework_text = "<framework>\n" + _generate_text(
        teacher, tokenizer, framework_prefix, max_new_tokens, temperature
    )
    return extract_framework(framework_text)


def generate_opd_rollout(
    teacher,
    student,
    tokenizer,
    question: str,
    *,
    mode: str,
    framework_max_new_tokens: int,
    solution_max_new_tokens: int,
    temperature: float,
) -> GuidedRollout:
    mode = validate_mode(mode)
    if mode == "guided":
        return generate_guided_rollout(
            teacher,
            student,
            tokenizer,
            question,
            framework_max_new_tokens=framework_max_new_tokens,
            solution_max_new_tokens=solution_max_new_tokens,
            temperature=temperature,
        )

    student_prompt = format_vanilla_student_prompt(question)
    completion = _generate_text(student, tokenizer, student_prompt, solution_max_new_tokens, temperature)
    prompt_ids = tokenizer(student_prompt, add_special_tokens=True)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None and (not completion_ids or completion_ids[-1] != tokenizer.eos_token_id):
        completion_ids.append(tokenizer.eos_token_id)
    input_ids = torch.tensor([prompt_ids + completion_ids], dtype=torch.long)
    return GuidedRollout(
        question=question,
        framework=[],
        student_prompt=student_prompt,
        completion=completion,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=torch.tensor([[-100] * len(prompt_ids) + completion_ids], dtype=torch.long),
    )
