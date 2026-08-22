import re
from collections.abc import Sequence


FRAMEWORK_SYSTEM_PROMPT = """You are a planning teacher for mathematical reasoning.
Given a problem and its reference solution, output only an abstract, concise numbered framework.
State which quantities or relationships to identify and which operations to perform in words.
Never solve an operation, copy a calculation, write an equation, repeat any numeric value from the
problem or solution, reveal the final answer, or use the marker ####. Do not spell out concrete
numbers in words. Relational operations such as halve, double, triple, or compare are allowed when
they are not evaluated. Apart from each step-number prefix, the step text must contain no digits. Use placeholders such as
"the given amount" and "the resulting total" instead of digits. Write 2 to 6 non-empty steps,
enclosed in <framework> and </framework>, with no text outside those tags."""

ROLLOUT_FRAMEWORK_SYSTEM_PROMPT = """You are a planning teacher for mathematical reasoning.
Given only a problem, output an abstract, concise numbered solution framework.
State operations in words without carrying them out. Never write an equation, any numeric value,
a solved intermediate value, the final answer, or the marker ####. Do not spell out concrete numbers
in words. Relational operations such as halve, double, triple, or compare are allowed when they are
not evaluated. Apart from each step-number prefix, the step text must contain no digits. Refer to problem quantities with
words such as "the given amount". Write 2 to 6 non-empty steps enclosed in <framework> and
</framework>, with no text outside those tags."""

STUDENT_SYSTEM_PROMPT = """You are a mathematical reasoning student.
Complete the solution by following every supplied framework step.
Give concise calculations and reasoning without restating or repeating the problem.
End with the final answer in the form: #### number"""

VANILLA_STUDENT_SYSTEM_PROMPT = """You are a mathematical reasoning student.
Solve the problem step by step.
Give concise calculations and reasoning without restating or repeating the problem.
End with the final answer in the form: #### number"""


def format_framework_prompt(
    question: str,
    answer: str,
    retry_reasons: Sequence[str] | None = None,
) -> str:
    prompt = (
        f"{FRAMEWORK_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n"
        f"Reference solution:\n{answer}\n"
    )
    if retry_reasons:
        prompt += (
            "\nThe previous attempt was rejected for: "
            + ", ".join(retry_reasons)
            + ". Produce a corrected framework.\n"
        )
    return prompt + "\n<framework>\n"


def format_rollout_framework_prompt(
    question: str,
    retry_reasons: Sequence[str] | None = None,
) -> str:
    prompt = f"{ROLLOUT_FRAMEWORK_SYSTEM_PROMPT}\n\nProblem:\n{question}\n"
    if retry_reasons:
        prompt += (
            "\nThe previous attempt was rejected for: "
            + ", ".join(retry_reasons)
            + ". Produce a corrected framework.\n"
        )
    return prompt + "\n<framework>\n"


def format_student_prompt(question: str, framework: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(framework, 1))
    return (
        f"{STUDENT_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n"
        f"<framework>\n{numbered}\n</framework>\n\n<solution>\n"
    )


def format_vanilla_student_prompt(question: str) -> str:
    return f"{VANILLA_STUDENT_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n<solution>\n"


def has_complete_framework(text: str) -> bool:
    return re.search(
        r"<framework>.*?</framework>", text, flags=re.DOTALL | re.IGNORECASE
    ) is not None


def extract_framework(text: str, *, require_closed: bool = False) -> list[str]:
    match = re.search(
        r"<framework>(.*?)</framework>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if require_closed and match is None:
        raise ValueError("framework_not_closed")
    if match is not None:
        body = match.group(1)
    else:
        opening = re.search(r"<framework>", text, flags=re.IGNORECASE)
        body = text[opening.end() :] if opening else text
    steps = []
    for line in body.splitlines():
        step = re.sub(r"^\s*\d+\s*[.、)]\s*", "", line).strip()
        if step != line.strip() and step:
            steps.append(step)
    if not steps:
        raise ValueError("teacher output contains no numbered steps")
    return steps
