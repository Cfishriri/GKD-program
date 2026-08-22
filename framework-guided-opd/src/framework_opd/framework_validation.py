import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any


MIN_FRAMEWORK_STEPS = 2
MAX_FRAMEWORK_STEPS = 6

ANSWER_MARKER_REASON = "answer_marker"
EVALUATED_EQUATION_REASON = "evaluated_equation"
FINAL_ANSWER_LEAK_REASON = "final_answer_leak"
NUMERIC_LITERAL_REASON = "numeric_literal"
NUMBER_WORD_REASON = "number_word"
LEAKAGE_REASONS = frozenset(
    {
        ANSWER_MARKER_REASON,
        EVALUATED_EQUATION_REASON,
        FINAL_ANSWER_LEAK_REASON,
        NUMERIC_LITERAL_REASON,
        NUMBER_WORD_REASON,
    }
)

_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?(?![\w.])"
)
_CARDINAL_WORDS = (
    r"zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|"
    r"billion|trillion"
)
_ORDINAL_WORDS = (
    r"first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|"
    r"eighteenth|nineteenth|twentieth|thirtieth|fortieth|fiftieth|sixtieth|"
    r"seventieth|eightieth|ninetieth|hundredth|thousandth|millionth|billionth|"
    r"trillionth"
)
_FRACTION_WORDS = r"half|halves|quarters?"
_QUANTIFIER_WORDS = (
    r"singles?|both|pairs?|couples?|dozens?|scores?|gross(?:es)?|nought|nil|"
    r"duos?|trios?|quartets?|quintets?|sextets?|septets?|octets?|nonets?"
)
_NUMBER_WORD_RE = re.compile(
    rf"\b(?:{_CARDINAL_WORDS}|{_ORDINAL_WORDS}|{_FRACTION_WORDS}|{_QUANTIFIER_WORDS})\b|"
    r"[零〇一二三四五六七八九十百千万萬亿億两兩壹贰貳叁參肆伍陆陸柒捌玖拾佰仟]",
    flags=re.IGNORECASE,
)
_DERIVED_NUMBER_WORD_RE = re.compile(
    rf"\b(?:{_CARDINAL_WORDS}|{_QUANTIFIER_WORDS}|double|triple|quadruple)[-\s]?fold\b",
    flags=re.IGNORECASE,
)
_ASCII_ROMAN_TOKEN_RE = re.compile(r"\b[MDCLXVI]+\b")
_EVALUATED_EQUATION_RE = re.compile(
    r"\d[^\n=＝]{0,80}(?:=|＝|equals?|gives?|yields?|produces?|becomes?|等于|得到|得出|结果为|可得|->|→)"
    r"\s*[$¥€£]?\s*[-+]?\d",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FrameworkValidationResult:
    valid: bool
    steps: tuple[str, ...]
    reasons: tuple[str, ...]


class FrameworkValidationError(ValueError):
    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("invalid framework: " + ", ".join(self.reasons))


def _normalise_number(value: str) -> str | None:
    candidate = value.strip().replace(",", "")
    candidate = candidate.lstrip("$¥€£")
    is_percent = candidate.endswith("%")
    if is_percent:
        candidate = candidate[:-1]
    try:
        if "/" in candidate:
            normalised = str(Fraction(candidate))
        else:
            number = Decimal(candidate)
            if number == 0:
                number = abs(number)
            normalised = format(number.normalize(), "f")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return normalised + ("%" if is_percent else "")


def normalise_final_answer(reference_answer: str | None) -> str | None:
    """Return a canonical numeric final answer, preferring GSM8K's #### marker."""
    if not reference_answer:
        return None
    answer_region = reference_answer.rsplit("####", 1)[-1]
    matches = _NUMBER_RE.findall(answer_region)
    if not matches and "####" not in reference_answer:
        matches = _NUMBER_RE.findall(reference_answer)
    return _normalise_number(matches[-1]) if matches else None


# US-spelling alias for callers that use "normalize" throughout their codebase.
normalize_final_answer = normalise_final_answer


def _contains_unicode_numeric(text: str) -> bool:
    """Catch every Unicode numeric class, including full-width and superscript digits."""
    return any(unicodedata.category(character).startswith("N") for character in text)


def _contains_ascii_roman_numeral(text: str) -> bool:
    """Catch standalone uppercase ASCII Roman numeral tokens, including ``I``."""
    return _ASCII_ROMAN_TOKEN_RE.search(text) is not None


def validate_framework(
    steps: Sequence[str] | object,
    reference_answer: str | None = None,
    *,
    forbid_numbers: bool = True,
) -> FrameworkValidationResult:
    """Validate that a framework is an abstract plan rather than a worked solution."""
    reasons: list[str] = []
    if isinstance(steps, (str, bytes)) or not isinstance(steps, Sequence):
        return FrameworkValidationResult(False, (), ("framework_not_list",))

    if any(not isinstance(step, str) for step in steps):
        reasons.append("non_string_step")
    clean_steps = tuple(step.strip() if isinstance(step, str) else "" for step in steps)
    if not MIN_FRAMEWORK_STEPS <= len(clean_steps) <= MAX_FRAMEWORK_STEPS:
        reasons.append("step_count")
    if any(not step for step in clean_steps):
        reasons.append("empty_step")

    body = "\n".join(clean_steps)
    if "####" in body:
        reasons.append(ANSWER_MARKER_REASON)
    if _EVALUATED_EQUATION_RE.search(body):
        reasons.append(EVALUATED_EQUATION_REASON)

    reference_value = normalise_final_answer(reference_answer)
    framework_values = {
        normalised
        for match in _NUMBER_RE.findall(body)
        if (normalised := _normalise_number(match)) is not None
    }
    if reference_value is not None and reference_value in framework_values:
        reasons.append(FINAL_ANSWER_LEAK_REASON)
    if forbid_numbers and (_contains_unicode_numeric(body) or _contains_ascii_roman_numeral(body)):
        reasons.append(NUMERIC_LITERAL_REASON)
    if forbid_numbers and (_NUMBER_WORD_RE.search(body) or _DERIVED_NUMBER_WORD_RE.search(body)):
        reasons.append(NUMBER_WORD_REASON)

    unique_reasons = tuple(dict.fromkeys(reasons))
    return FrameworkValidationResult(not unique_reasons, clean_steps, unique_reasons)


def require_valid_framework(
    steps: Sequence[str] | object,
    reference_answer: str | None = None,
    *,
    forbid_numbers: bool = True,
) -> list[str]:
    result = validate_framework(steps, reference_answer, forbid_numbers=forbid_numbers)
    if not result.valid:
        raise FrameworkValidationError(result.reasons)
    return list(result.steps)


def audit_framework_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarise purity and structure failures in framework JSONL records."""
    reason_counts: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    valid = 0
    leakage_records = 0

    for index, record in enumerate(records):
        record_reasons: list[str] = []
        if not isinstance(record, dict):
            reason_counts.update(["record_not_object"])
            if len(invalid_examples) < 20:
                invalid_examples.append({"index": index, "reasons": ["record_not_object"]})
            continue
        question_value = record.get("question")
        answer_value = record.get("answer")
        question = question_value.strip() if isinstance(question_value, str) else ""
        answer = answer_value.strip() if isinstance(answer_value, str) else ""
        if not question:
            record_reasons.append("missing_question")
        if not answer:
            record_reasons.append("missing_answer")

        result = validate_framework(record.get("framework"), answer or None)
        record_reasons.extend(result.reasons)
        record_reasons = list(dict.fromkeys(record_reasons))
        if record_reasons:
            reason_counts.update(record_reasons)
            if LEAKAGE_REASONS.intersection(record_reasons):
                leakage_records += 1
            if len(invalid_examples) < 20:
                invalid_examples.append({"index": index, "reasons": record_reasons})
        else:
            valid += 1

    total = len(records)
    invalid = total - valid
    return {
        "schema_version": 2,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "reasons": dict(sorted(reason_counts.items())),
        "leakage_records": leakage_records,
        "leakage_rate": leakage_records / total if total else 0.0,
        "invalid_examples": invalid_examples,
    }
