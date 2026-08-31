import hashlib
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .answer_stopping import NUMBER_PATTERN, truncate_after_first_complete_answer


# Commas are accepted only as real thousands separators.  Horizontal whitespace is
# spelled out deliberately: ``\s`` would also match a newline and weaken the
# physical-line contract used by the strict metric.
MARKED_ANSWER_PATTERN = re.compile(
    rf"[ \t]*####[ \t]+({NUMBER_PATTERN})[ \t]*"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> dict:
    artifact = Path(path)
    if not artifact.exists():
        raise FileNotFoundError(f"artifact does not exist: {artifact}")
    if artifact.is_file():
        files = [artifact]
        root = artifact.parent
    else:
        root = artifact
        files = sorted(
            candidate
            for candidate in artifact.rglob("*")
            if candidate.is_file()
            and (candidate.name == "adapter_config.json" or candidate.name.startswith("adapter_model."))
        )
        if not files:
            raise ValueError(f"adapter directory contains no adapter weights/config: {artifact}")
    entries = [
        {
            "relative_path": file.relative_to(root).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in files
    ]
    combined = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"path": str(artifact), "type": "file" if artifact.is_file() else "directory", "files": entries, "sha256": combined}


def experiment_signature(config: dict) -> str:
    """Hash result-affecting configuration while ignoring the resume switch."""

    stable_config = {key: value for key, value in config.items() if key != "resume"}
    payload = json.dumps(stable_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_resume_identity(manifest: dict, signature: str, fingerprints: dict) -> None:
    if manifest.get("experiment_signature") != signature:
        raise ValueError("resume config does not match the original experiment")
    if manifest.get("artifact_fingerprints") != fingerprints:
        raise ValueError("dataset or adapter fingerprints changed; refusing to resume")


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
    """Extract a strict answer from the last non-empty physical line only."""

    last_nonempty = next((line for line in reversed(text.splitlines()) if line.strip()), None)
    if last_nonempty is None:
        return None
    marked = MARKED_ANSWER_PATTERN.fullmatch(last_nonempty)
    return normalize_answer(marked.group(1)) if marked else None


def extract_relaxed_answer(text: str) -> str | None:
    """Extract a marked answer, or the last number as a diagnostic fallback."""

    marked = extract_final_answer(text)
    if marked is not None:
        return marked
    numbers = re.findall(NUMBER_PATTERN, text)
    return normalize_answer(numbers[-1]) if numbers else None


def score_prediction(prediction: str, reference: str) -> dict:
    predicted_answer = extract_final_answer(prediction)
    relaxed_predicted_answer = extract_relaxed_answer(prediction)
    reference_answer = extract_final_answer(reference)
    if reference_answer is None:
        reference_answer = extract_relaxed_answer(reference)
    has_answer_marker = predicted_answer is not None
    return {
        "predicted_answer": predicted_answer,
        "relaxed_predicted_answer": relaxed_predicted_answer,
        "reference_answer": reference_answer,
        "correct": predicted_answer is not None and predicted_answer == reference_answer,
        "relaxed_correct": relaxed_predicted_answer is not None and relaxed_predicted_answer == reference_answer,
        "has_answer_marker": has_answer_marker,
        "contains_marker_token": "####" in prediction,
    }


def _wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    rate = correct / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(bool(row["correct"]) for row in rows)
    relaxed_correct = sum(bool(row.get("relaxed_correct", row["correct"])) for row in rows)
    marker_count = sum(bool(row["has_answer_marker"]) for row in rows)
    answer_stop_count = sum(bool(row.get("stopped_on_answer", False)) for row in rows)
    eos_count = sum(bool(row.get("ended_with_eos", False)) for row in rows)
    truncated_count = sum(bool(row.get("hit_max_tokens", False)) for row in rows)
    framework_failures = sum(bool(row.get("framework_failure", False)) for row in rows)
    shared_framework_failures = sum(
        bool(row.get("shared_framework_failure", row.get("framework_failure", False)))
        for row in rows
    )
    generated_tokens = [int(row.get("generated_tokens", 0)) for row in rows]
    student_prompt_tokens = [int(row.get("student_prompt_tokens", 0)) for row in rows]
    student_output_tokens = [
        int(row.get("student_output_tokens", row.get("generated_tokens", 0))) for row in rows
    ]
    framework_prompt_tokens = [int(row.get("framework_prompt_tokens", 0)) for row in rows]
    framework_output_tokens = [int(row.get("framework_output_tokens", 0)) for row in rows]
    token_cost_proxy = [
        int(
            row.get(
                "token_cost_proxy",
                student_prompt_tokens[index]
                + student_output_tokens[index]
                + framework_prompt_tokens[index]
                + framework_output_tokens[index],
            )
        )
        for index, row in enumerate(rows)
    ]
    student_latencies = [float(row.get("student_latency_seconds", 0.0)) for row in rows]
    framework_latencies = [float(row.get("framework_latency_seconds", 0.0)) for row in rows]
    framework_calls = [int(row.get("framework_4b_calls", 0)) for row in rows]
    ci_low, ci_high = _wilson_interval(correct, total)
    relaxed_ci_low, relaxed_ci_high = _wilson_interval(relaxed_correct, total)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "accuracy_ci95_low": ci_low,
        "accuracy_ci95_high": ci_high,
        "relaxed_correct": relaxed_correct,
        "relaxed_accuracy": relaxed_correct / total if total else 0.0,
        "relaxed_accuracy_ci95_low": relaxed_ci_low,
        "relaxed_accuracy_ci95_high": relaxed_ci_high,
        "answer_format_rate": marker_count / total if total else 0.0,
        "answer_stop_rate": answer_stop_count / total if total else 0.0,
        "eos_rate": eos_count / total if total else 0.0,
        "truncation_rate": truncated_count / total if total else 0.0,
        "framework_failure_rate": framework_failures / total if total else 0.0,
        "shared_framework_failure_rate": shared_framework_failures / total if total else 0.0,
        "average_generated_tokens": sum(generated_tokens) / total if total else 0.0,
        "average_student_prompt_tokens": sum(student_prompt_tokens) / total if total else 0.0,
        "average_student_output_tokens": sum(student_output_tokens) / total if total else 0.0,
        "average_framework_prompt_tokens": sum(framework_prompt_tokens) / total if total else 0.0,
        "average_framework_output_tokens": sum(framework_output_tokens) / total if total else 0.0,
        "average_token_cost_proxy": sum(token_cost_proxy) / total if total else 0.0,
        "total_framework_4b_calls": sum(framework_calls),
        "average_framework_4b_calls": sum(framework_calls) / total if total else 0.0,
        "total_framework_latency_seconds": sum(framework_latencies),
        "average_framework_latency_seconds": sum(framework_latencies) / total if total else 0.0,
        "average_student_latency_seconds": sum(student_latencies) / total if total else 0.0,
    }


def exact_mcnemar_pvalue(baseline_only: int, comparison_only: int) -> float:
    """Two-sided exact McNemar p-value under a Binomial(n, 0.5) null."""

    if baseline_only < 0 or comparison_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = baseline_only + comparison_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(baseline_only, comparison_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _paired_bootstrap_ci(values: list[int], seed: int, bootstrap_samples: int) -> tuple[float, float]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    total = len(values)
    estimates = [
        sum(values[rng.randrange(total)] for _ in range(total)) / total
        for _ in range(bootstrap_samples)
    ]
    estimates.sort()
    return _percentile(estimates, 0.025), _percentile(estimates, 0.975)


def _index_paired_rows(rows: list[dict], label: str) -> dict[object, dict]:
    indexed: dict[object, dict] = {}
    for row in rows:
        example_id = row["example_id"]
        if example_id in indexed:
            raise ValueError(f"duplicate example_id in {label}: {example_id}")
        indexed[example_id] = row
    return indexed


def paired_comparison(
    baseline_rows: list[dict],
    comparison_rows: list[dict],
    *,
    baseline_name: str,
    comparison_name: str,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict:
    """Compare strict exact-match outcomes on the same examples."""

    baseline = _index_paired_rows(baseline_rows, baseline_name)
    comparison = _index_paired_rows(comparison_rows, comparison_name)
    if set(baseline) != set(comparison):
        raise ValueError("paired comparisons require identical example_id sets")

    example_ids = sorted(baseline, key=str)
    differences: list[int] = []
    both_correct = baseline_only = comparison_only = both_wrong = 0
    for example_id in example_ids:
        baseline_correct = bool(baseline[example_id]["correct"])
        comparison_correct = bool(comparison[example_id]["correct"])
        differences.append(int(comparison_correct) - int(baseline_correct))
        if baseline_correct and comparison_correct:
            both_correct += 1
        elif baseline_correct:
            baseline_only += 1
        elif comparison_correct:
            comparison_only += 1
        else:
            both_wrong += 1

    total = len(differences)
    delta = sum(differences) / total if total else 0.0
    ci_low, ci_high = _paired_bootstrap_ci(differences, seed, bootstrap_samples)
    return {
        "type": "paired_difference",
        "baseline": baseline_name,
        "comparison": comparison_name,
        "total": total,
        "baseline_accuracy": sum(bool(row["correct"]) for row in baseline.values()) / total if total else 0.0,
        "comparison_accuracy": sum(bool(row["correct"]) for row in comparison.values()) / total if total else 0.0,
        "accuracy_delta": delta,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "outcomes": {
            "both_correct": both_correct,
            "baseline_only_correct": baseline_only,
            "comparison_only_correct": comparison_only,
            "both_wrong": both_wrong,
            "discordant": baseline_only + comparison_only,
        },
        "mcnemar_exact_pvalue": exact_mcnemar_pvalue(baseline_only, comparison_only),
    }


def paired_interaction(
    vanilla_no_rows: list[dict],
    guided_no_rows: list[dict],
    vanilla_with_rows: list[dict],
    guided_with_rows: list[dict],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict:
    """Estimate the paired 2×2 difference-in-differences interaction."""

    indexed = {
        "vanilla_no_framework": _index_paired_rows(vanilla_no_rows, "vanilla_no_framework"),
        "guided_no_framework": _index_paired_rows(guided_no_rows, "guided_no_framework"),
        "vanilla_with_framework": _index_paired_rows(vanilla_with_rows, "vanilla_with_framework"),
        "guided_with_framework": _index_paired_rows(guided_with_rows, "guided_with_framework"),
    }
    id_sets = [set(rows) for rows in indexed.values()]
    if any(example_ids != id_sets[0] for example_ids in id_sets[1:]):
        raise ValueError("paired interaction requires identical example_id sets")
    differences = []
    for example_id in sorted(id_sets[0], key=str):
        values = {
            cell: int(bool(rows[example_id]["correct"]))
            for cell, rows in indexed.items()
        }
        differences.append(
            (values["guided_with_framework"] - values["guided_no_framework"])
            - (values["vanilla_with_framework"] - values["vanilla_no_framework"])
        )
    total = len(differences)
    interaction = sum(differences) / total if total else 0.0
    ci_low, ci_high = _paired_bootstrap_ci(differences, seed, bootstrap_samples)
    return {
        "type": "interaction",
        "formula": "(guided_with-guided_no) - (vanilla_with-vanilla_no)",
        "total": total,
        "accuracy_delta": interaction,
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "outcomes": None,
        "mcnemar_exact_pvalue": None,
    }
