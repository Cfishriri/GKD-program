from dataclasses import dataclass

from .evaluation import extract_final_answer
from .prompts import format_student_prompt
from .rollout import _generate_text


@dataclass(frozen=True)
class SemanticVerificationResult:
    correct: bool
    predicted_answer: str | None
    reference_answer: str | None
    prediction: str
    prompt_tokens: int
    generated_tokens: int
    stopped_on_answer: bool
    hit_max_tokens: bool
    ended_with_eos: bool


def verify_framework_semantics(
    model,
    tokenizer,
    question: str,
    framework: list[str],
    reference: str,
    max_new_tokens: int,
) -> SemanticVerificationResult:
    """Execute a privileged label framework and require a strict reference match."""

    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= 2048:
        raise ValueError("semantic max_new_tokens must be between 1 and 2048")
    prompt = format_student_prompt(question, framework)
    generation = _generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        0.0,
        stop_on_answer=True,
    )
    predicted_answer = extract_final_answer(generation.text)
    reference_answer = extract_final_answer(reference)
    return SemanticVerificationResult(
        correct=(
            predicted_answer is not None
            and reference_answer is not None
            and predicted_answer == reference_answer
        ),
        predicted_answer=predicted_answer,
        reference_answer=reference_answer,
        prediction=generation.text,
        prompt_tokens=generation.prompt_tokens,
        generated_tokens=len(generation.token_ids),
        stopped_on_answer=generation.stopped_on_answer,
        hit_max_tokens=generation.hit_max_tokens,
        ended_with_eos=generation.ended_with_eos,
    )
