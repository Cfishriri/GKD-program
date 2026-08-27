from dataclasses import dataclass

import torch
from transformers import StoppingCriteriaList

from .answer_stopping import AnswerLineStoppingCriteria, truncate_after_first_complete_answer
from .framework_validation import require_valid_framework
from .prompts import (
    extract_framework,
    format_framework_prompt,
    format_rollout_framework_prompt,
    format_student_prompt,
    format_vanilla_student_prompt,
    has_complete_framework,
)


@dataclass(frozen=True)
class GenerationResult:
    token_ids: tuple[int, ...]
    text: str
    ended_with_eos: bool
    hit_max_tokens: bool
    prompt_tokens: int = 0
    stopped_on_answer: bool = False


@dataclass(frozen=True)
class FrameworkGenerationResult:
    steps: tuple[str, ...]
    attempts: int
    used_fallback: bool
    validation_errors: tuple[str, ...]
    prompt_tokens: int
    generated_tokens: int
    hit_max_attempts: int
    last_ended_with_eos: bool
    closed_tag: bool

    @property
    def total_generated_tokens(self) -> int:
        """Backward-compatible descriptive alias used by cost reports."""
        return self.generated_tokens


@dataclass
class GuidedRollout:
    question: str
    framework: list[str]
    student_prompt: str
    completion: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    generated_token_ids: tuple[int, ...]
    ended_with_eos: bool
    hit_max_tokens: bool
    stopped_on_answer: bool
    framework_attempts: int
    framework_fallback: bool
    framework_validation_errors: tuple[str, ...]
    framework_prompt_tokens: int
    framework_generated_tokens: int
    framework_hit_max_attempts: int
    framework_last_ended_with_eos: bool
    framework_closed_tag: bool


FALLBACK_FRAMEWORK = (
    "Identify the quantities and relationships described in the problem",
    "Translate the relationships into a mathematical model",
    "Apply the required operations to derive the result",
    "Check the reasoning and present the conclusion",
)


def validate_mode(mode: str) -> str:
    if mode not in {"vanilla", "guided"}:
        raise ValueError("mode must be 'vanilla' or 'guided'")
    return mode


def _generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    *,
    stop_on_answer: bool = False,
) -> GenerationResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_kwargs.update(temperature=temperature, top_p=0.9)
    if stop_on_answer:
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [AnswerLineStoppingCriteria(tokenizer, encoded["input_ids"].shape[1])]
        )
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            **generation_kwargs,
        )
    sequences = getattr(output_ids, "sequences", output_ids)
    generated = sequences[0, encoded["input_ids"].shape[1] :]
    token_ids = tuple(int(token_id) for token_id in generated.detach().cpu().tolist())
    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = ()
    elif isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)
    else:
        eos_token_ids = tuple(eos_token_ids)
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_token_ids)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    text, stopped_on_answer = truncate_after_first_complete_answer(decoded_text)
    return GenerationResult(
        token_ids=token_ids,
        text=text,
        ended_with_eos=ended_with_eos,
        hit_max_tokens=(
            len(token_ids) >= max_new_tokens and not ended_with_eos and not stopped_on_answer
        ),
        prompt_tokens=int(encoded["input_ids"].shape[1]),
        stopped_on_answer=stopped_on_answer,
    )


def _build_rollout(
    *,
    question: str,
    framework: list[str],
    student_prompt: str,
    generation: GenerationResult,
    tokenizer,
    framework_attempts: int = 0,
    framework_fallback: bool = False,
    framework_validation_errors: tuple[str, ...] = (),
    framework_prompt_tokens: int = 0,
    framework_generated_tokens: int = 0,
    framework_hit_max_attempts: int = 0,
    framework_last_ended_with_eos: bool = False,
    framework_closed_tag: bool = False,
) -> GuidedRollout:
    prompt_ids = tokenizer(student_prompt, add_special_tokens=True)["input_ids"]
    generated_ids = list(generation.token_ids)
    input_ids = torch.tensor([prompt_ids + generated_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100] * len(prompt_ids) + generated_ids], dtype=torch.long)
    return GuidedRollout(
        question=question,
        framework=framework,
        student_prompt=student_prompt,
        completion=generation.text,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        generated_token_ids=generation.token_ids,
        ended_with_eos=generation.ended_with_eos,
        hit_max_tokens=generation.hit_max_tokens,
        stopped_on_answer=generation.stopped_on_answer,
        framework_attempts=framework_attempts,
        framework_fallback=framework_fallback,
        framework_validation_errors=framework_validation_errors,
        framework_prompt_tokens=framework_prompt_tokens,
        framework_generated_tokens=framework_generated_tokens,
        framework_hit_max_attempts=framework_hit_max_attempts,
        framework_last_ended_with_eos=framework_last_ended_with_eos,
        framework_closed_tag=framework_closed_tag,
    )


def generate_guided_rollout(
    teacher,
    student,
    tokenizer,
    question: str,
    *,
    framework_max_new_tokens: int,
    solution_max_new_tokens: int,
    framework_temperature: float,
    solution_temperature: float,
    framework_max_attempts: int = 3,
) -> GuidedRollout:
    framework_result = generate_framework_result(
        teacher,
        tokenizer,
        question,
        max_new_tokens=framework_max_new_tokens,
        temperature=framework_temperature,
        max_attempts=framework_max_attempts,
    )
    framework = list(framework_result.steps)
    student_prompt = format_student_prompt(question, framework)
    generation = _generate_text(
        student,
        tokenizer,
        student_prompt,
        solution_max_new_tokens,
        solution_temperature,
        stop_on_answer=True,
    )
    return _build_rollout(
        question=question,
        framework=framework,
        student_prompt=student_prompt,
        generation=generation,
        tokenizer=tokenizer,
        framework_attempts=framework_result.attempts,
        framework_fallback=framework_result.used_fallback,
        framework_validation_errors=framework_result.validation_errors,
        framework_prompt_tokens=framework_result.prompt_tokens,
        framework_generated_tokens=framework_result.generated_tokens,
        framework_hit_max_attempts=framework_result.hit_max_attempts,
        framework_last_ended_with_eos=framework_result.last_ended_with_eos,
        framework_closed_tag=framework_result.closed_tag,
    )


def generate_framework(
    teacher,
    tokenizer,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    max_attempts: int = 3,
) -> list[str]:
    return list(
        generate_framework_result(
            teacher,
            tokenizer,
            question,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_attempts=max_attempts,
        ).steps
    )


def generate_framework_result(
    teacher,
    tokenizer,
    question: str,
    *,
    max_new_tokens: int,
    temperature: float,
    max_attempts: int = 3,
    reference_answer: str | None = None,
) -> FrameworkGenerationResult:
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    retry_reasons: tuple[str, ...] | None = None
    validation_errors: list[str] = []
    prompt_tokens = 0
    generated_tokens = 0
    hit_max_attempts = 0
    last_ended_with_eos = False
    closed_tag = False
    for attempt in range(1, max_attempts + 1):
        prompt = (
            format_framework_prompt(question, reference_answer, retry_reasons=retry_reasons)
            if reference_answer is not None
            else format_rollout_framework_prompt(question, retry_reasons=retry_reasons)
        )
        generation = _generate_text(teacher, tokenizer, prompt, max_new_tokens, temperature)
        prompt_tokens += generation.prompt_tokens
        generated_tokens += len(generation.token_ids)
        hit_max_attempts += int(generation.hit_max_tokens)
        last_ended_with_eos = generation.ended_with_eos
        framework_text = "<framework>\n" + generation.text
        closed_tag = has_complete_framework(framework_text)
        attempt_reasons: list[str] = []
        if not closed_tag:
            if generation.hit_max_tokens:
                attempt_reasons.append("hit_max_tokens")
            attempt_reasons.append("framework_not_closed")
        if attempt_reasons:
            validation_errors.extend(attempt_reasons)
            retry_reasons = tuple(attempt_reasons)
            continue
        try:
            steps = require_valid_framework(
                extract_framework(framework_text, require_closed=True), reference_answer
            )
            return FrameworkGenerationResult(
                tuple(steps),
                attempt,
                False,
                tuple(validation_errors),
                prompt_tokens,
                generated_tokens,
                hit_max_attempts,
                last_ended_with_eos,
                closed_tag,
            )
        except ValueError as error:
            retry_reasons = tuple(getattr(error, "reasons", ("parse_error",)))
            validation_errors.extend(retry_reasons)

    return FrameworkGenerationResult(
        FALLBACK_FRAMEWORK,
        max_attempts,
        True,
        tuple(validation_errors),
        prompt_tokens,
        generated_tokens,
        hit_max_attempts,
        last_ended_with_eos,
        closed_tag,
    )


def generate_opd_rollout(
    teacher,
    student,
    tokenizer,
    question: str,
    *,
    mode: str,
    framework_max_new_tokens: int,
    solution_max_new_tokens: int,
    framework_temperature: float,
    solution_temperature: float,
    framework_max_attempts: int = 3,
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
            framework_temperature=framework_temperature,
            solution_temperature=solution_temperature,
            framework_max_attempts=framework_max_attempts,
        )

    student_prompt = format_vanilla_student_prompt(question)
    generation = _generate_text(
        student,
        tokenizer,
        student_prompt,
        solution_max_new_tokens,
        solution_temperature,
        stop_on_answer=True,
    )
    return _build_rollout(
        question=question,
        framework=[],
        student_prompt=student_prompt,
        generation=generation,
        tokenizer=tokenizer,
    )
