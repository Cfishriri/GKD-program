import re

import torch
from transformers import StoppingCriteria


# Commas are accepted only as real thousands separators. Horizontal whitespace
# is explicit because ``\s`` would weaken the physical-line answer contract.
NUMBER_PATTERN = r"[-+]?\$?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?"
COMPLETE_MARKED_ANSWER_LINE_PATTERN = re.compile(
    rf"(?m)^[ \t]*####[ \t]+({NUMBER_PATTERN})[ \t]*(?:\r?\n)"
)


def truncate_after_first_complete_answer(text: str) -> tuple[str, bool]:
    """Trim text after its first complete strict marked-answer line."""

    match = COMPLETE_MARKED_ANSWER_LINE_PATTERN.search(text)
    if match is None:
        return text, False
    return text[: match.end()].rstrip("\r\n"), True


class AnswerLineStoppingCriteria(StoppingCriteria):
    """Stop once newly generated text contains a complete strict answer line."""

    def __init__(self, tokenizer, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)

    def __call__(self, input_ids, scores, **kwargs):
        decisions = []
        for sequence in input_ids:
            generated_ids = sequence[self.prompt_length :].detach().cpu().tolist()
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            _, found = truncate_after_first_complete_answer(generated_text)
            decisions.append(found)
        return torch.tensor(decisions, dtype=torch.bool, device=input_ids.device)
