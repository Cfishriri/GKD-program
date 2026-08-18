import re
from collections.abc import Sequence


FRAMEWORK_SYSTEM_PROMPT = """You are a planning teacher for mathematical reasoning.
Given a problem and its reference solution, output only a concise numbered framework.
Describe the operations needed, but do not copy calculations or reveal the final numeric answer.
Use 2 to 6 steps enclosed in <framework> and </framework>."""

ROLLOUT_FRAMEWORK_SYSTEM_PROMPT = """You are a planning teacher for mathematical reasoning.
Given only a problem, output a concise numbered solution framework.
Do not solve the problem and do not state the final answer.
Use 2 to 6 steps enclosed in <framework> and </framework>."""

STUDENT_SYSTEM_PROMPT = """You are a mathematical reasoning student.
Complete the solution by following every supplied framework step.
End with the final answer in the form: #### number"""


def format_framework_prompt(question: str, answer: str) -> str:
    return (
        f"{FRAMEWORK_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n"
        f"Reference solution:\n{answer}\n\n<framework>\n"
    )


def format_rollout_framework_prompt(question: str) -> str:
    return f"{ROLLOUT_FRAMEWORK_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n<framework>\n"


def format_student_prompt(question: str, framework: Sequence[str]) -> str:
    numbered = "\n".join(f"{index}. {step}" for index, step in enumerate(framework, 1))
    return (
        f"{STUDENT_SYSTEM_PROMPT}\n\nProblem:\n{question}\n\n"
        f"<framework>\n{numbered}\n</framework>\n\n<solution>\n"
    )


def extract_framework(text: str) -> list[str]:
    match = re.search(r"<framework>(.*?)(?:</framework>|$)", text, flags=re.DOTALL | re.IGNORECASE)
    body = match.group(1) if match else text
    steps = []
    for line in body.splitlines():
        step = re.sub(r"^\s*\d+\s*[.、)]\s*", "", line).strip()
        if step != line.strip() and step:
            steps.append(step)
    if not steps:
        raise ValueError("teacher output contains no numbered steps")
    return steps[:6]

