import math
import re
from decimal import Decimal, InvalidOperation


NUMBER_PATTERN = r"[-+]?\$?\d[\d,]*(?:\.\d+)?"


def normalize_answer(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.replace("$", "").replace(",", "").strip()
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None
    normalized = format(number.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized


def extract_final_answer(text: str) -> str | None:
    marked = re.findall(rf"####\s*({NUMBER_PATTERN})", text)
    if marked:
        return normalize_answer(marked[-1])
    numbers = re.findall(NUMBER_PATTERN, text)
    return normalize_answer(numbers[-1]) if numbers else None


def score_prediction(prediction: str, reference: str) -> dict:
    predicted_answer = extract_final_answer(prediction)
    reference_answer = extract_final_answer(reference)
    return {
        "predicted_answer": predicted_answer,
        "reference_answer": reference_answer,
        "correct": predicted_answer is not None and predicted_answer == reference_answer,
        "has_answer_marker": "####" in prediction,
    }


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    marker_count = sum(bool(row["has_answer_marker"]) for row in rows)
    accuracy = correct / total if total else 0.0
    if total:
        z = 1.96
        denominator = 1 + z * z / total
        center = (accuracy + z * z / (2 * total)) / denominator
        margin = z * math.sqrt(accuracy * (1 - accuracy) / total + z * z / (4 * total * total)) / denominator
        ci_low, ci_high = center - margin, center + margin
    else:
        ci_low = ci_high = 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "answer_format_rate": marker_count / total if total else 0.0,
    }
