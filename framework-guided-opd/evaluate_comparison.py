import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import peft
import torch
import transformers
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from framework_opd.data import load_records
from framework_opd.evaluation import (
    artifact_fingerprint,
    experiment_signature,
    file_sha256,
    paired_comparison,
    paired_interaction,
    score_prediction,
    summarize,
    truncate_after_first_complete_answer,
)
from framework_opd.framework_validation import validate_framework
from framework_opd.prompts import format_student_prompt, format_vanilla_student_prompt
from framework_opd.rollout import FALLBACK_FRAMEWORK, GenerationResult, generate_framework_result


SCHEMA_VERSION = 4
CORE_CELL_ORDER = [
    "vanilla_no_framework",
    "guided_no_framework",
    "vanilla_with_framework",
    "guided_with_framework",
]
SOURCE_FILES = [
    "evaluate_comparison.py",
    "src/framework_opd/data.py",
    "src/framework_opd/evaluation.py",
    "src/framework_opd/framework_validation.py",
    "src/framework_opd/prompts.py",
    "src/framework_opd/rollout.py",
]
MODEL_IDENTITY_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
MODEL_WEIGHT_SUFFIXES = {".bin", ".safetensors"}
TRAINING_ALLOWED_DIFFERENCES = {
    "artifact_type",
    "mode",
    "output_dir",
    "framework_teacher_adapter",
    "resume_from_checkpoint",
    "role",
    "run_id",
}


class FrameworkGenerationFailure(RuntimeError):
    """A domain failure after framework generation exhausted its retry policy."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256_text(payload)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path, *, allow_truncated_final_line: bool = False) -> list[dict]:
    rows = []
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            truncated = line_number == len(lines) and not line.endswith(("\n", "\r"))
            if allow_truncated_final_line and truncated:
                break
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected a JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_causal_model(path: str, device: str):
    return AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(device)


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


def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int) -> GenerationResult:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    encoded = {name: tensor.to(model.device) for name, tensor in encoded.items()}
    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [AnswerLineStoppingCriteria(tokenizer, encoded["input_ids"].shape[1])]
            ),
        )
    generated = output_ids[0, encoded["input_ids"].shape[1] :].detach().cpu().tolist()
    token_ids = tuple(int(token_id) for token_id in generated)
    eos_token_ids = tokenizer.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = ()
    elif isinstance(eos_token_ids, int):
        eos_token_ids = (eos_token_ids,)
    else:
        eos_token_ids = tuple(eos_token_ids)
    ended_with_eos = bool(token_ids and token_ids[-1] in eos_token_ids)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    truncated_text, stopped_on_answer = truncate_after_first_complete_answer(decoded_text)
    return GenerationResult(
        token_ids=token_ids,
        text=truncated_text,
        ended_with_eos=ended_with_eos,
        hit_max_tokens=(
            len(token_ids) >= max_new_tokens and not ended_with_eos and not stopped_on_answer
        ),
        prompt_tokens=int(encoded["input_ids"].shape[1]),
    )


def validate_config(config: dict) -> None:
    required = {
        "student_model",
        "teacher_model",
        "framework_teacher_adapter",
        "dataset",
        "output_dir",
        "student_device",
        "teacher_device",
        "seed",
        "limit",
        "max_new_tokens",
        "framework_max_new_tokens",
        "framework_max_attempts",
        "framework_teacher_expected_records",
        "bootstrap_samples",
        "adapters",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"evaluation config is missing keys: {sorted(missing)}")
    adapters = config["adapters"]
    if not isinstance(adapters, dict) or set(adapters) != {"vanilla", "guided"}:
        raise ValueError("adapters must contain exactly 'vanilla' and 'guided'")
    for key in (
        "limit",
        "max_new_tokens",
        "framework_max_new_tokens",
        "framework_max_attempts",
        "framework_teacher_expected_records",
        "bootstrap_samples",
        "progress_every",
    ):
        value = config.get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_new_tokens"] > 2048:
        raise ValueError("max_new_tokens must not exceed 2048")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    if config.get("framework_failure_policy", "fallback") not in {"fallback", "error"}:
        raise ValueError("framework_failure_policy must be 'fallback' or 'error'")
    if "resume" in config and not isinstance(config["resume"], bool):
        raise ValueError("resume must be a boolean")
    if "include_base" in config and not isinstance(config["include_base"], bool):
        raise ValueError("include_base must be a boolean")


def _identity_entries(root: Path, files: list[Path]) -> list[dict]:
    return [
        {
            "relative_path": file.relative_to(root).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in sorted(files)
    ]


def model_identity(path: str | Path) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {root}")
    files = [file for file in root.iterdir() if file.is_file() and file.name in MODEL_IDENTITY_FILES]
    if not any(file.name == "config.json" for file in files):
        raise ValueError(f"model identity is missing config.json: {root}")
    entries = _identity_entries(root, files)
    weight_files = [
        file
        for file in root.iterdir()
        if file.is_file() and file.suffix in MODEL_WEIGHT_SUFFIXES
    ]
    if not weight_files:
        raise ValueError(f"model identity found no weight files: {root}")
    weight_metadata = [
        {
            "relative_path": file.relative_to(root).as_posix(),
            "size_bytes": file.stat().st_size,
            "mtime_ns": file.stat().st_mtime_ns,
        }
        for file in sorted(weight_files)
    ]
    return {
        "path": str(root),
        "resolved_path": str(root.resolve()),
        "identity_scope": "config, tokenizer, and weight-index metadata",
        "files": entries,
        "weight_files": weight_metadata,
        "sha256": sha256_json({"identity_files": entries, "weight_files": weight_metadata}),
    }


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_identity(repo_root: Path) -> dict:
    entries = _identity_entries(
        repo_root,
        [repo_root / relative for relative in SOURCE_FILES if (repo_root / relative).is_file()],
    )
    status = _git_value(repo_root, "status", "--porcelain")
    return {
        "files": entries,
        "source_sha256": sha256_json(entries),
        "git": {
            "commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "branch": _git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None,
            "status_sha256": sha256_text(status) if status is not None else None,
        },
    }


def runtime_identity() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "platform": platform.platform(),
    }


def _metadata_root(adapter_path: str | Path) -> Path:
    adapter = Path(adapter_path)
    for candidate in (adapter, adapter.parent):
        if (candidate / "run_config.json").is_file():
            return candidate
    raise FileNotFoundError(f"no run_config.json found for adapter: {adapter}")


def load_adapter_metadata(adapter_path: str | Path) -> dict:
    adapter = Path(adapter_path)
    root = _metadata_root(adapter)
    run_config_path = root / "run_config.json"
    complete_path = root / "RUN_COMPLETE"
    if not complete_path.is_file():
        raise ValueError(f"adapter is not marked complete: {adapter}")
    adapter_fingerprint = artifact_fingerprint(adapter)
    run_config = read_json(run_config_path)
    completion = read_json(complete_path)
    run_config_hash = file_sha256(run_config_path)
    if completion.get("status") != "complete":
        raise ValueError(f"adapter completion status is not complete: {adapter}")
    if completion.get("run_config_sha256") != run_config_hash:
        raise ValueError(f"adapter completion marker does not match run_config: {adapter}")
    if completion.get("adapter_artifact_sha256") != adapter_fingerprint["sha256"]:
        raise ValueError(f"adapter completion marker does not match adapter weights: {adapter}")
    metadata = {
        "adapter": adapter_fingerprint,
        "run_root": str(root),
        "run_config": run_config,
        "run_config_sha256": run_config_hash,
        "completion": completion,
        "completion_sha256": file_sha256(complete_path),
    }
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        metadata["run_manifest"] = read_json(manifest_path)
        metadata["run_manifest_sha256"] = file_sha256(manifest_path)
    return metadata


def _training_value(run_config: dict, key: str):
    if key in run_config:
        return run_config[key]
    nested = run_config.get("config")
    return nested.get(key) if isinstance(nested, dict) else None


def _comparable_training_config(run_config: dict) -> dict:
    nested = run_config.get("config")
    values = dict(nested) if isinstance(nested, dict) else dict(run_config)
    for key in TRAINING_ALLOWED_DIFFERENCES:
        values.pop(key, None)
    for key in ("schema_version", "created_at", "completed_at", "resume_history"):
        values.pop(key, None)
    return values


def _recompute_student_training_signature(run_config: dict, mode: str) -> dict:
    # Delayed import keeps config/provenance helpers importable without pulling the
    # training entrypoint into normal evaluation startup before it is needed.
    from train_opd import _resume_signature

    return _resume_signature(
        run_config,
        mode,
        project_root=Path(__file__).resolve().parent,
    )


def _validate_framework_teacher_metadata(config: dict, framework: dict) -> None:
    from train_teacher import lightweight_model_identity

    framework_config = framework["run_config"]
    if framework_config.get("role") != "framework_teacher":
        raise ValueError("framework adapter run_config role must be 'framework_teacher'")
    if framework_config.get("artifact_type") != "framework_teacher_adapter":
        raise ValueError("framework adapter artifact_type is invalid")
    if framework["completion"].get("role") != "framework_teacher":
        raise ValueError("framework adapter RUN_COMPLETE role is invalid")
    if framework["completion"].get("artifact_type") != "framework_teacher_adapter":
        raise ValueError("framework adapter RUN_COMPLETE artifact_type is invalid")
    framework_run_id = framework_config.get("run_id")
    if not framework_run_id or framework["completion"].get("run_id") != framework_run_id:
        raise ValueError("framework adapter run_id is missing or inconsistent")
    if framework_config.get("base_model") != config["teacher_model"]:
        raise ValueError("framework adapter base model does not match evaluation teacher_model")
    expected_model_identity = lightweight_model_identity(config["teacher_model"])
    if framework_config.get("base_model_identity") != expected_model_identity:
        raise ValueError("framework adapter base_model_identity does not match current model")

    expected_records = int(config["framework_teacher_expected_records"])
    if (
        framework_config.get("expected_records") != expected_records
        or framework_config.get("num_records") != expected_records
    ):
        raise ValueError("framework adapter record count does not match evaluation policy")
    purity_audit = framework_config.get("purity_audit")
    if not isinstance(purity_audit, dict):
        raise ValueError("framework adapter provenance is missing purity_audit")
    if (
        purity_audit.get("total") != expected_records
        or purity_audit.get("valid") != expected_records
        or purity_audit.get("invalid") != 0
        or purity_audit.get("leakage_records", purity_audit.get("leakage")) != 0
    ):
        raise ValueError("framework adapter purity audit is not clean and count-complete")

    data_path = Path(framework_config.get("data", ""))
    if not data_path.is_file():
        raise ValueError("framework adapter training data is unavailable")
    data_sha256 = file_sha256(data_path)
    if framework_config.get("data_sha256") != data_sha256:
        raise ValueError("framework adapter data_sha256 does not match training data")
    generation_audit = framework_config.get("generation_audit")
    if not isinstance(generation_audit, dict):
        raise ValueError("framework adapter provenance is missing generation_audit")
    if (
        generation_audit.get("status") != "complete"
        or generation_audit.get("requested_valid") != expected_records
        or generation_audit.get("valid") != expected_records
        or generation_audit.get("output_sha256") != data_sha256
    ):
        raise ValueError("framework generation audit is incomplete or does not bind the data")
    audit_path = Path(generation_audit.get("path", ""))
    if not audit_path.is_file() or generation_audit.get("sha256") != file_sha256(audit_path):
        raise ValueError("framework generation audit file fingerprint changed")
    on_disk_audit = read_json(audit_path)
    for key in ("status", "requested_valid", "valid", "output_sha256"):
        if on_disk_audit.get(key) != generation_audit.get(key):
            raise ValueError(f"framework generation audit snapshot mismatch: {key}")


def validate_training_metadata(config: dict, metadata: dict[str, dict]) -> None:
    vanilla = metadata["vanilla"]
    guided = metadata["guided"]
    if vanilla["adapter"]["sha256"] == guided["adapter"]["sha256"]:
        raise ValueError("vanilla and guided adapters have the same fingerprint")
    for mode, item in (("vanilla", vanilla), ("guided", guided)):
        run_config = item["run_config"]
        if run_config.get("role") != "opd_student":
            raise ValueError(f"{mode} run_config role must be 'opd_student'")
        if _training_value(run_config, "mode") != mode:
            raise ValueError(f"{mode} adapter was not trained in {mode} mode")
        if float(_training_value(run_config, "beta")) != 1.0:
            raise ValueError(f"{mode} adapter beta must be 1.0 for reverse-KL OPD")
        if _training_value(run_config, "student_model") != config["student_model"]:
            raise ValueError(f"{mode} adapter student_model does not match evaluation")
        if _training_value(run_config, "teacher_model") != config["teacher_model"]:
            raise ValueError(f"{mode} adapter teacher_model does not match evaluation")
        if item["completion"].get("role") != "opd_student":
            raise ValueError(f"{mode} RUN_COMPLETE role must be 'opd_student'")
        run_id = run_config.get("run_id")
        if not run_id or item["completion"].get("run_id") != run_id:
            raise ValueError(f"{mode} run_id is missing or inconsistent")
        run_manifest = item.get("run_manifest")
        if not isinstance(run_manifest, dict):
            raise ValueError(f"{mode} adapter is missing its training run_manifest")
        if (
            run_manifest.get("schema_version") != 2
            or run_manifest.get("run_id") != run_id
            or run_manifest.get("role") != "opd_student"
            or run_manifest.get("status") != "complete"
            or run_manifest.get("run_config_sha256") != item["run_config_sha256"]
            or run_manifest.get("adapter_artifact_sha256") != item["adapter"]["sha256"]
        ):
            raise ValueError(f"{mode} training run_manifest is inconsistent")
        expected_signature = _recompute_student_training_signature(run_config, mode)
        if run_manifest.get("run_signature") != expected_signature:
            raise ValueError(f"{mode} training run_signature does not match current inputs/code")
    if _comparable_training_config(vanilla["run_config"]) != _comparable_training_config(
        guided["run_config"]
    ):
        raise ValueError("vanilla and guided training configs differ outside the allowed fields")

    trained_framework = _training_value(guided["run_config"], "framework_teacher_adapter")
    if not trained_framework:
        raise ValueError("guided training provenance is missing framework_teacher_adapter")
    if artifact_fingerprint(trained_framework)["sha256"] != metadata["framework_teacher"]["adapter"]["sha256"]:
        raise ValueError("guided training used a different framework teacher adapter")

    _validate_framework_teacher_metadata(config, metadata["framework_teacher"])


def collect_provenance(config: dict, repo_root: Path) -> dict:
    metadata = {
        "vanilla": load_adapter_metadata(config["adapters"]["vanilla"]),
        "guided": load_adapter_metadata(config["adapters"]["guided"]),
        "framework_teacher": load_adapter_metadata(config["framework_teacher_adapter"]),
    }
    validate_training_metadata(config, metadata)
    return {
        "source": source_identity(repo_root),
        "runtime": runtime_identity(),
        "dataset": artifact_fingerprint(config["dataset"]),
        "models": {
            "student": model_identity(config["student_model"]),
            "teacher": model_identity(config["teacher_model"]),
        },
        "training_artifacts": metadata,
    }


def validate_resume_provenance(manifest: dict, signature: str, provenance: dict) -> None:
    if manifest.get("experiment_signature") != signature:
        raise ValueError("resume config does not match the original experiment")
    if manifest.get("provenance") != provenance:
        raise ValueError("source, runtime, dataset, model, or adapter provenance changed")


def select_records(dataset: str, limit: int, seed: int) -> list[dict]:
    indexed = [dict(record, example_id=index) for index, record in enumerate(load_records(dataset))]
    random.Random(seed).shuffle(indexed)
    return indexed[:limit]


def stable_framework_id(
    example_id: int, question: str, framework: list[str], framework_adapter_sha256: str
) -> str:
    identity = {
        "example_id": example_id,
        "question": question,
        "framework": framework,
        "framework_adapter_sha256": framework_adapter_sha256,
    }
    return "fw_" + sha256_json(identity)[:24]


def load_framework_cache(
    path: Path,
    records: list[dict],
    framework_adapter_sha256: str,
) -> dict[int, dict]:
    if not path.exists():
        return {}
    expected = {record["example_id"]: record for record in records}
    cache: dict[int, dict] = {}
    required = {
        "framework_id",
        "example_id",
        "question",
        "framework",
        "framework_valid",
        "framework_failure",
        "framework_fallback",
        "framework_attempts",
        "framework_validation_errors",
        "framework_prompt_tokens",
        "framework_output_tokens",
        "framework_latency_seconds",
        "framework_hit_max_attempts",
        "framework_last_ended_with_eos",
        "framework_closed_tag",
    }
    for entry in read_jsonl(path, allow_truncated_final_line=True):
        missing = required - set(entry)
        if missing:
            raise ValueError(f"framework cache entry is missing fields: {sorted(missing)}")
        example_id = entry["example_id"]
        if example_id not in expected:
            raise ValueError(f"framework cache contains unexpected example_id: {example_id}")
        if example_id in cache:
            raise ValueError(f"framework cache contains duplicate example_id: {example_id}")
        if entry["question"] != expected[example_id]["question"]:
            raise ValueError(f"framework cache question mismatch for example_id {example_id}")
        validation = validate_framework(entry["framework"], None)
        if not validation.valid:
            raise ValueError(f"cached framework is structurally invalid for example_id {example_id}")
        if bool(entry["framework_valid"]) == bool(entry["framework_failure"]):
            raise ValueError(f"cached framework validity flags disagree for example_id {example_id}")
        if bool(entry["framework_failure"]) != bool(entry["framework_fallback"]):
            raise ValueError(f"cached framework fallback flags disagree for example_id {example_id}")
        expected_id = stable_framework_id(
            example_id,
            entry["question"],
            entry["framework"],
            framework_adapter_sha256,
        )
        if entry["framework_id"] != expected_id:
            raise ValueError(f"cached framework_id mismatch for example_id {example_id}")
        for key in (
            "framework_attempts",
            "framework_prompt_tokens",
            "framework_output_tokens",
            "framework_hit_max_attempts",
        ):
            if int(entry[key]) < 0:
                raise ValueError(f"negative {key} for example_id {example_id}")
        if float(entry["framework_latency_seconds"]) < 0:
            raise ValueError(f"negative framework latency for example_id {example_id}")
        cache[example_id] = entry
    return cache


def generate_framework_cache(
    config: dict,
    tokenizer,
    records: list[dict],
    output_dir: Path,
    framework_adapter_sha256: str,
    *,
    resume: bool,
) -> tuple[dict[int, dict], dict]:
    policy = config.get("framework_failure_policy", "fallback")
    max_attempts = int(config["framework_max_attempts"])
    progress_every = int(config.get("progress_every", 10))
    cache_path = output_dir / "framework_cache.jsonl"
    cache = (
        load_framework_cache(cache_path, records, framework_adapter_sha256) if resume else {}
    )
    ordered_cached = [cache[record["example_id"]] for record in records if record["example_id"] in cache]
    write_jsonl(cache_path, ordered_cached)
    missing_records = [record for record in records if record["example_id"] not in cache]

    framework_base = framework_teacher = None
    if missing_records:
        framework_base = load_causal_model(config["teacher_model"], config["teacher_device"])
        framework_teacher = PeftModel.from_pretrained(
            framework_base,
            config["framework_teacher_adapter"],
            is_trainable=False,
        )
        framework_teacher.eval()

    with cache_path.open("a", encoding="utf-8") as stream:
        for position, record in enumerate(missing_records, 1):
            started = time.perf_counter()
            result = generate_framework_result(
                framework_teacher,
                tokenizer,
                record["question"],
                max_new_tokens=config["framework_max_new_tokens"],
                temperature=0.0,
                max_attempts=max_attempts,
            )
            latency = time.perf_counter() - started
            framework = list(result.steps)
            used_fallback = bool(result.used_fallback)
            if used_fallback and policy == "error":
                raise FrameworkGenerationFailure(
                    "; ".join(result.validation_errors) or "framework generation used fallback"
                )
            if used_fallback:
                framework = list(FALLBACK_FRAMEWORK)
            entry = {
                "framework_id": stable_framework_id(
                    record["example_id"], record["question"], framework, framework_adapter_sha256
                ),
                "example_id": record["example_id"],
                "question": record["question"],
                "framework": framework,
                "framework_valid": not used_fallback,
                "framework_failure": used_fallback,
                "framework_fallback": used_fallback,
                "framework_attempts": int(result.attempts),
                "framework_validation_errors": list(result.validation_errors),
                "framework_prompt_tokens": int(result.prompt_tokens),
                "framework_output_tokens": int(result.generated_tokens),
                "framework_latency_seconds": latency,
                "framework_hit_max_attempts": int(result.hit_max_attempts),
                "framework_last_ended_with_eos": bool(result.last_ended_with_eos),
                "framework_closed_tag": bool(result.closed_tag),
            }
            cache[record["example_id"]] = entry
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            completed = len(cache)
            if position == 1 or completed == len(records) or position % progress_every == 0:
                failures = sum(bool(item["framework_failure"]) for item in cache.values())
                print(f"[frameworks] {completed}/{len(records)} (fallbacks={failures})", flush=True)

    if framework_teacher is not None:
        del framework_teacher
        del framework_base
        clear_cuda_cache()
    failures = sum(bool(item["framework_failure"]) for item in cache.values())
    attempts = sum(int(item["framework_attempts"]) for item in cache.values())
    prompt_tokens = sum(int(item["framework_prompt_tokens"]) for item in cache.values())
    output_tokens = sum(int(item["framework_output_tokens"]) for item in cache.values())
    latency = sum(float(item["framework_latency_seconds"]) for item in cache.values())
    hit_max_attempts = sum(int(item["framework_hit_max_attempts"]) for item in cache.values())
    closed = sum(bool(item["framework_closed_tag"]) for item in cache.values())
    return cache, {
        "total": len(records),
        "failures": failures,
        "failure_rate": failures / len(records) if records else 0.0,
        "total_4b_calls": attempts,
        "average_4b_calls": attempts / len(records) if records else 0.0,
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "total_latency_seconds": latency,
        "average_latency_seconds": latency / len(records) if records else 0.0,
        "hit_max_attempts": hit_max_attempts,
        "closed_tag_rate": closed / len(records) if records else 0.0,
        "failure_policy": policy,
        "max_attempts": max_attempts,
    }


def model_specs(config: dict) -> list[tuple[str, str | None]]:
    specs: list[tuple[str, str | None]] = []
    if config.get("include_base", False):
        specs.append(("base", None))
    specs.extend(
        [
            ("vanilla", config["adapters"]["vanilla"]),
            ("guided", config["adapters"]["guided"]),
        ]
    )
    return specs


def ordered_cells(include_base: bool) -> list[str]:
    cells = list(CORE_CELL_ORDER)
    if include_base:
        cells = ["base_no_framework", "base_with_framework", *cells]
    return cells


def prompt_for_cell(record: dict, cache_entry: dict, use_framework: bool) -> str:
    return (
        format_student_prompt(record["question"], cache_entry["framework"])
        if use_framework
        else format_vanilla_student_prompt(record["question"])
    )


def pending_records(records: list[dict], existing_rows: list[dict]) -> list[dict]:
    existing_ids = [row["example_id"] for row in existing_rows]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("existing prediction rows contain duplicate example_id")
    expected_ids = {record["example_id"] for record in records}
    unexpected = set(existing_ids) - expected_ids
    if unexpected:
        raise ValueError(f"existing prediction rows contain unexpected ids: {sorted(unexpected)}")
    completed = set(existing_ids)
    return [record for record in records if record["example_id"] not in completed]


def load_prediction_rows(
    path: Path,
    records: list[dict],
    framework_cache: dict[int, dict],
    expected_cells: list[str],
    tokenizer=None,
) -> dict[str, list[dict]]:
    grouped: dict[str, dict[int, dict]] = {cell: {} for cell in expected_cells}
    if not path.exists():
        return {cell: [] for cell in expected_cells}
    expected_records = {record["example_id"]: record for record in records}
    required = {
        "cell",
        "adapter",
        "framework_condition",
        "example_id",
        "question",
        "reference",
        "framework",
        "framework_id",
        "shared_framework_id",
        "framework_used",
        "framework_failure",
        "shared_framework_failure",
        "shared_framework_attempts",
        "shared_framework_hit_max_attempts",
        "shared_framework_closed_tag",
        "prediction",
        "student_prompt_sha256",
        "student_prompt_tokens",
        "student_output_tokens",
        "student_latency_seconds",
        "framework_prompt_tokens",
        "framework_output_tokens",
        "framework_latency_seconds",
        "framework_4b_calls",
        "token_cost_proxy",
        "generated_tokens",
        "ended_with_eos",
        "hit_max_tokens",
    }
    for row in read_jsonl(path, allow_truncated_final_line=True):
        missing = required - set(row)
        if missing:
            raise ValueError(f"prediction row is missing fields: {sorted(missing)}")
        cell = row["cell"]
        if cell not in grouped:
            raise ValueError(f"predictions contain unexpected cell: {cell}")
        example_id = row["example_id"]
        if example_id not in expected_records:
            raise ValueError(f"predictions contain unexpected example_id: {example_id}")
        if example_id in grouped[cell]:
            raise ValueError(f"predictions contain duplicate cell/example_id: {cell}/{example_id}")
        record = expected_records[example_id]
        cached = framework_cache[example_id]
        if row["question"] != record["question"] or row["reference"] != record["answer"]:
            raise ValueError(f"prediction source mismatch for {cell}/{example_id}")
        use_framework = cell.endswith("_with_framework")
        adapter = cell.removesuffix("_with_framework").removesuffix("_no_framework")
        condition = "with_framework" if use_framework else "no_framework"
        if row["adapter"] != adapter or row["framework_condition"] != condition:
            raise ValueError(f"prediction cell mapping mismatch for {cell}/{example_id}")
        if bool(row["framework_used"]) != use_framework:
            raise ValueError(f"prediction framework usage mismatch for {cell}/{example_id}")
        expected_framework = cached["framework"] if use_framework else []
        expected_framework_id = cached["framework_id"] if use_framework else None
        if row["framework"] != expected_framework or row["framework_id"] != expected_framework_id:
            raise ValueError(f"prediction framework mismatch for {cell}/{example_id}")
        if row["shared_framework_id"] != cached["framework_id"]:
            raise ValueError(f"prediction shared framework mismatch for {cell}/{example_id}")
        actual_failure = bool(cached["framework_failure"]) if use_framework else False
        if bool(row["framework_failure"]) != actual_failure:
            raise ValueError(f"prediction actual framework failure mismatch for {cell}/{example_id}")
        if bool(row["shared_framework_failure"]) != bool(cached["framework_failure"]):
            raise ValueError(f"prediction shared framework failure mismatch for {cell}/{example_id}")
        if int(row["shared_framework_attempts"]) != int(cached["framework_attempts"]):
            raise ValueError(f"prediction shared framework attempts mismatch for {cell}/{example_id}")
        if int(row["shared_framework_hit_max_attempts"]) != int(cached["framework_hit_max_attempts"]):
            raise ValueError(f"prediction shared framework truncation mismatch for {cell}/{example_id}")
        if bool(row["shared_framework_closed_tag"]) != bool(cached["framework_closed_tag"]):
            raise ValueError(f"prediction shared framework closure mismatch for {cell}/{example_id}")
        prompt = prompt_for_cell(record, cached, use_framework)
        if row["student_prompt_sha256"] != sha256_text(prompt):
            raise ValueError(f"prediction prompt hash mismatch for {cell}/{example_id}")
        if tokenizer is not None:
            prompt_tokens = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
            if int(row["student_prompt_tokens"]) != prompt_tokens:
                raise ValueError(f"prediction prompt token count mismatch for {cell}/{example_id}")
        framework_prompt_tokens = int(cached["framework_prompt_tokens"]) if use_framework else 0
        framework_output_tokens = int(cached["framework_output_tokens"]) if use_framework else 0
        framework_latency = float(cached["framework_latency_seconds"]) if use_framework else 0.0
        framework_calls = int(cached["framework_attempts"]) if use_framework else 0
        if int(row["framework_prompt_tokens"]) != framework_prompt_tokens:
            raise ValueError(f"prediction framework prompt cost mismatch for {cell}/{example_id}")
        if int(row["framework_output_tokens"]) != framework_output_tokens:
            raise ValueError(f"prediction framework output cost mismatch for {cell}/{example_id}")
        if float(row["framework_latency_seconds"]) != framework_latency:
            raise ValueError(f"prediction framework latency mismatch for {cell}/{example_id}")
        if int(row["framework_4b_calls"]) != framework_calls:
            raise ValueError(f"prediction framework call count mismatch for {cell}/{example_id}")
        if int(row["student_output_tokens"]) != int(row["generated_tokens"]):
            raise ValueError(f"prediction output token aliases disagree for {cell}/{example_id}")
        expected_cost = (
            int(row["student_prompt_tokens"])
            + int(row["student_output_tokens"])
            + framework_prompt_tokens
            + framework_output_tokens
        )
        if int(row["token_cost_proxy"]) != expected_cost:
            raise ValueError(f"prediction token cost proxy mismatch for {cell}/{example_id}")
        rescored = score_prediction(row["prediction"], row["reference"])
        if any(row.get(key) != value for key, value in rescored.items()):
            raise ValueError(f"stored score mismatch for {cell}/{example_id}")
        grouped[cell][example_id] = row
    return {
        cell: [grouped[cell][record["example_id"]] for record in records if record["example_id"] in grouped[cell]]
        for cell in expected_cells
    }


def evaluate_cells(
    config: dict,
    tokenizer,
    records: list[dict],
    framework_cache: dict[int, dict],
    output_dir: Path,
    manifest: dict,
    *,
    resume: bool,
) -> dict[str, list[dict]]:
    progress_every = int(config.get("progress_every", 10))
    predictions_path = output_dir / "predictions.jsonl"
    cell_order = ordered_cells(bool(config.get("include_base", False)))
    rows_by_cell = (
        load_prediction_rows(predictions_path, records, framework_cache, cell_order, tokenizer)
        if resume
        else {cell: [] for cell in cell_order}
    )
    preserved_rows = [row for cell in cell_order for row in rows_by_cell[cell]]
    write_jsonl(predictions_path, preserved_rows)
    manifest["completed_cells"] = [
        cell for cell in cell_order if len(rows_by_cell[cell]) == len(records)
    ]
    manifest["partial_cells"] = {
        cell: len(rows_by_cell[cell])
        for cell in cell_order
        if 0 < len(rows_by_cell[cell]) < len(records)
    }
    write_json(output_dir / "run_manifest.json", manifest)

    with predictions_path.open("a", encoding="utf-8") as stream:
        for adapter_name, adapter_path in model_specs(config):
            cells = [f"{adapter_name}_no_framework", f"{adapter_name}_with_framework"]
            pending_by_cell = {
                cell: pending_records(records, rows_by_cell[cell]) for cell in cells
            }
            if not any(pending_by_cell.values()):
                print(f"[{adapter_name}] reusing two completed cells", flush=True)
                continue
            model = load_causal_model(config["student_model"], config["student_device"])
            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
            model.eval()
            for cell in cells:
                use_framework = cell.endswith("_with_framework")
                condition = "with_framework" if use_framework else "no_framework"
                missing = pending_by_cell[cell]
                existing_count = len(rows_by_cell[cell])
                for position, record in enumerate(missing, 1):
                    cached = framework_cache[record["example_id"]]
                    framework = cached["framework"] if use_framework else []
                    prompt = prompt_for_cell(record, cached, use_framework)
                    started = time.perf_counter()
                    generated = generate_completion(model, tokenizer, prompt, config["max_new_tokens"])
                    student_latency = time.perf_counter() - started
                    prompt_tokens = int(generated.prompt_tokens)
                    output_tokens = len(generated.token_ids)
                    framework_prompt_tokens = int(cached["framework_prompt_tokens"]) if use_framework else 0
                    framework_output_tokens = int(cached["framework_output_tokens"]) if use_framework else 0
                    framework_latency = float(cached["framework_latency_seconds"]) if use_framework else 0.0
                    framework_calls = int(cached["framework_attempts"]) if use_framework else 0
                    row = {
                        "cell": cell,
                        "adapter": adapter_name,
                        "framework_condition": condition,
                        "example_id": record["example_id"],
                        "question": record["question"],
                        "reference": record["answer"],
                        "framework": framework,
                        "framework_id": cached["framework_id"] if use_framework else None,
                        "shared_framework_id": cached["framework_id"],
                        "framework_used": use_framework,
                        "framework_valid": cached["framework_valid"] if use_framework else None,
                        "framework_failure": bool(cached["framework_failure"]) if use_framework else False,
                        "framework_fallback": bool(cached["framework_fallback"]) if use_framework else False,
                        "framework_attempts": int(cached["framework_attempts"]) if use_framework else 0,
                        "framework_validation_errors": cached["framework_validation_errors"] if use_framework else [],
                        "shared_framework_failure": bool(cached["framework_failure"]),
                        "shared_framework_attempts": int(cached["framework_attempts"]),
                        "shared_framework_hit_max_attempts": int(cached["framework_hit_max_attempts"]),
                        "shared_framework_closed_tag": bool(cached["framework_closed_tag"]),
                        "prediction": generated.text,
                        "student_prompt_sha256": sha256_text(prompt),
                        "student_prompt_tokens": prompt_tokens,
                        "student_output_tokens": output_tokens,
                        "student_latency_seconds": student_latency,
                        "framework_prompt_tokens": framework_prompt_tokens,
                        "framework_output_tokens": framework_output_tokens,
                        "framework_latency_seconds": framework_latency,
                        "framework_4b_calls": framework_calls,
                        "token_cost_proxy": prompt_tokens
                        + output_tokens
                        + framework_prompt_tokens
                        + framework_output_tokens,
                        "generated_tokens": output_tokens,
                        "ended_with_eos": generated.ended_with_eos,
                        "hit_max_tokens": generated.hit_max_tokens,
                        **score_prediction(generated.text, record["answer"]),
                    }
                    rows_by_cell[cell].append(row)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    completed = existing_count + position
                    if position == 1 or completed == len(records) or position % progress_every == 0:
                        print(f"[{cell}] {completed}/{len(records)}", flush=True)
                if len(rows_by_cell[cell]) == len(records) and cell not in manifest["completed_cells"]:
                    manifest["completed_cells"].append(cell)
                manifest["partial_cells"].pop(cell, None)
                write_json(output_dir / "run_manifest.json", manifest)
            del model
            clear_cuda_cache()
    return rows_by_cell


def write_accuracy(
    output_dir: Path, rows_by_cell: dict[str, list[dict]], cell_order: list[str]
) -> list[dict]:
    summaries = [{"cell": cell, **summarize(rows_by_cell[cell])} for cell in cell_order]
    with (output_dir / "accuracy.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(
        output_dir / "summary.json",
        {
            "metric": "strict exact match from the last non-empty physical `#### number` line",
            "token_cost_proxy": "student prompt + student output; with-framework also adds all 4B framework-attempt prompt and output tokens",
            "cells": summaries,
        },
    )
    return summaries


def build_comparisons(config: dict, rows_by_cell: dict[str, list[dict]]) -> list[dict]:
    definitions = [
        (
            "guided_minus_vanilla_no_framework",
            "vanilla_no_framework",
            "guided_no_framework",
            "primary",
            "guided-vs-vanilla adapter effect under the same no-framework inference prompt",
        ),
        (
            "guided_minus_vanilla_with_framework",
            "vanilla_with_framework",
            "guided_with_framework",
            "primary",
            "guided-vs-vanilla adapter effect under the same shared framework-conditioned prompt",
        ),
        (
            "framework_effect_on_vanilla",
            "vanilla_no_framework",
            "vanilla_with_framework",
            "exploratory",
            "framework-conditioned prompt bundle effect (framework text plus different system instruction) on vanilla",
        ),
        (
            "framework_effect_on_guided",
            "guided_no_framework",
            "guided_with_framework",
            "exploratory",
            "framework-conditioned prompt bundle effect (framework text plus different system instruction) on guided",
        ),
        (
            "guided_system_minus_vanilla_system",
            "vanilla_no_framework",
            "guided_with_framework",
            "system",
            "full guided system bundle versus full vanilla system bundle",
        ),
    ]
    if config.get("include_base", False):
        definitions.extend(
            [
                ("vanilla_minus_base_no_framework", "base_no_framework", "vanilla_no_framework", "exploratory", "vanilla adapter effect versus base under no-framework inference"),
                ("guided_minus_base_no_framework", "base_no_framework", "guided_no_framework", "exploratory", "guided adapter effect versus base under no-framework inference"),
                ("vanilla_minus_base_with_framework", "base_with_framework", "vanilla_with_framework", "exploratory", "vanilla adapter effect versus base under shared framework-conditioned inference"),
                ("guided_minus_base_with_framework", "base_with_framework", "guided_with_framework", "exploratory", "guided adapter effect versus base under shared framework-conditioned inference"),
            ]
        )
    comparisons = []
    for offset, (name, baseline, comparison, tier, estimand) in enumerate(definitions):
        result = paired_comparison(
            rows_by_cell[baseline],
            rows_by_cell[comparison],
            baseline_name=baseline,
            comparison_name=comparison,
            seed=int(config["seed"]) + offset,
            bootstrap_samples=int(config["bootstrap_samples"]),
        )
        comparisons.append(
            {"name": name, "analysis_tier": tier, "estimand": estimand, **result}
        )
    interaction = paired_interaction(
        rows_by_cell["vanilla_no_framework"],
        rows_by_cell["guided_no_framework"],
        rows_by_cell["vanilla_with_framework"],
        rows_by_cell["guided_with_framework"],
        seed=int(config["seed"]) + len(definitions),
        bootstrap_samples=int(config["bootstrap_samples"]),
    )
    comparisons.append(
        {
            "name": "adapter_framework_interaction",
            "analysis_tier": "exploratory",
            "estimand": "difference-in-differences for adapter by framework-conditioned prompt bundle",
            **interaction,
        }
    )
    return comparisons


def write_paired_outputs(output_dir: Path, comparisons: list[dict], config: dict) -> None:
    write_json(
        output_dir / "paired_comparisons.json",
        {
            "metric": "strict_exact_match",
            "primary_comparisons": [row["name"] for row in comparisons if row["analysis_tier"] == "primary"],
            "framework_effect_caveat": "framework_effect_on_* estimates the framework-conditioned prompt bundle, including its system instruction; it is not a framework-text-only intervention",
            "delta_definition": "comparison_accuracy - baseline_accuracy",
            "bootstrap": "paired percentile bootstrap",
            "bootstrap_samples": int(config["bootstrap_samples"]),
            "comparisons": comparisons,
        },
    )
    outcome_rows = []
    for comparison in comparisons:
        outcomes = comparison["outcomes"] or {
            "both_correct": None,
            "baseline_only_correct": None,
            "comparison_only_correct": None,
            "both_wrong": None,
            "discordant": None,
        }
        outcome_rows.append(
            {
                "name": comparison["name"],
                "analysis_tier": comparison["analysis_tier"],
                "type": comparison["type"],
                "baseline": comparison.get("baseline"),
                "comparison": comparison.get("comparison"),
                **outcomes,
                "accuracy_delta": comparison["accuracy_delta"],
                "bootstrap_ci95_low": comparison["bootstrap_ci95_low"],
                "bootstrap_ci95_high": comparison["bootstrap_ci95_high"],
                "mcnemar_exact_pvalue": comparison["mcnemar_exact_pvalue"],
            }
        )
    with (output_dir / "paired_outcomes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(outcome_rows[0]))
        writer.writeheader()
        writer.writerows(outcome_rows)


def plot_grouped_accuracy(output_dir: Path, summaries: list[dict], include_base: bool) -> None:
    by_cell = {row["cell"]: row for row in summaries}
    adapters = ["base", "vanilla", "guided"] if include_base else ["vanilla", "guided"]
    labels = {"base": "Base", "vanilla": "Vanilla OPD", "guided": "Guided OPD"}
    positions = list(range(len(adapters)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for offset, (condition, label, color) in enumerate(
        (("no_framework", "No framework", "#64748b"), ("with_framework", "Shared framework", "#10b981"))
    ):
        values = [by_cell[f"{adapter}_{condition}"]["accuracy"] for adapter in adapters]
        lower = [value - by_cell[f"{adapter}_{condition}"]["accuracy_ci95_low"] for adapter, value in zip(adapters, values)]
        upper = [by_cell[f"{adapter}_{condition}"]["accuracy_ci95_high"] - value for adapter, value in zip(adapters, values)]
        x_values = [position + (offset - 0.5) * width for position in positions]
        bars = axis.bar(x_values, values, width, label=label, color=color, yerr=[lower, upper], capsize=5)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.1%}", ha="center", fontsize=9)
    axis.set_xticks(positions, [labels[adapter] for adapter in adapters])
    axis.set_ylabel("Strict GSM8K exact-match accuracy")
    axis.set_ylim(0, 1.12)
    axis.set_title(f"2×2 adapter × inference prompt evaluation (n={summaries[0]['total']})")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "grouped_accuracy.png", dpi=180)
    plt.close(fig)


def plot_paired_deltas(output_dir: Path, comparisons: list[dict]) -> None:
    core = [row for row in comparisons if row["analysis_tier"] in {"primary", "system", "exploratory"}]
    labels = [row["name"].replace("_", " ") for row in core]
    values = [row["accuracy_delta"] for row in core]
    colors = []
    for row in core:
        if row["bootstrap_ci95_low"] <= 0 <= row["bootstrap_ci95_high"]:
            colors.append("#94a3b8")
        elif row["accuracy_delta"] > 0:
            colors.append("#10b981")
        else:
            colors.append("#ef4444")
    fig, axis = plt.subplots(figsize=(11, max(5.5, len(core) * 0.58)))
    y_values = list(range(len(core)))
    for y_value, value, row, color in zip(y_values, values, core, colors):
        axis.hlines(y_value, row["bootstrap_ci95_low"], row["bootstrap_ci95_high"], color=color, linewidth=2)
        axis.plot(value, y_value, "o", markersize=7, color=color)
    axis.axvline(0, color="#0f172a", linewidth=1)
    axis.set_yticks(y_values, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Paired strict-accuracy delta")
    axis.set_title("Paired deltas (neutral when the 95% bootstrap CI crosses zero)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_deltas.png", dpi=180)
    plt.close(fig)


def plot_paired_outcomes(output_dir: Path, comparisons: list[dict]) -> None:
    paired = [row for row in comparisons if row.get("outcomes")]
    labels = [row["name"].replace("_", "\n") for row in paired]
    categories = [
        ("both_correct", "Both correct", "#10b981"),
        ("baseline_only_correct", "Baseline only", "#f59e0b"),
        ("comparison_only_correct", "Comparison only", "#2563eb"),
        ("both_wrong", "Both wrong", "#94a3b8"),
    ]
    fig, axis = plt.subplots(figsize=(max(10, len(paired) * 1.5), 6))
    bottoms = [0] * len(paired)
    for key, label, color in categories:
        values = [row["outcomes"][key] for row in paired]
        axis.bar(range(len(paired)), values, bottom=bottoms, label=label, color=color)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]
    axis.set_xticks(range(len(paired)), labels, fontsize=8)
    axis.set_ylabel("Paired examples")
    axis.set_title("Paired correctness outcomes")
    axis.legend(ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "paired_outcomes.png", dpi=180)
    plt.close(fig)


def plot_accuracy_vs_cost(output_dir: Path, summaries: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for row in summaries:
        with_framework = row["cell"].endswith("_with_framework")
        color = "#10b981" if with_framework else "#64748b"
        marker = "s" if with_framework else "o"
        axis.errorbar(
            row["average_token_cost_proxy"],
            row["accuracy"],
            yerr=[[row["accuracy"] - row["accuracy_ci95_low"]], [row["accuracy_ci95_high"] - row["accuracy"]]],
            fmt=marker,
            color=color,
            capsize=4,
            markersize=8,
        )
        annotation = (
            f"{row['cell']}\n4B calls={row['total_framework_4b_calls']}, "
            f"4B latency={row['total_framework_latency_seconds']:.1f}s"
        )
        axis.annotate(annotation, (row["average_token_cost_proxy"], row["accuracy"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Average token-cost proxy per example")
    axis.set_ylabel("Strict exact-match accuracy")
    axis.set_ylim(0, 1.05)
    axis.set_title("Accuracy vs cost proxy (student prompt+output; + all 4B attempt tokens when used)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_cost.png", dpi=180)
    plt.close(fig)


def plot_diagnostics(output_dir: Path, summaries: list[dict], framework_stats: dict) -> None:
    cells = [row["cell"] for row in summaries]
    labels = [cell.replace("_", "\n") for cell in cells]
    colors = ["#10b981" if "with_framework" in cell else "#64748b" for cell in cells]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        ("answer_format_rate", "Valid final-line #### answer rate", (0, 1.05)),
        ("eos_rate", "EOS termination rate", (0, 1.05)),
        ("truncation_rate", "Student max-token truncation rate", (0, 1.05)),
        ("average_generated_tokens", "Average student output tokens", None),
    ]
    for axis, (key, title, limits) in zip(axes.flat, panels):
        values = [row[key] for row in summaries]
        axis.bar(range(len(cells)), values, color=colors)
        axis.set_xticks(range(len(cells)), labels, fontsize=8)
        axis.set_title(title)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Generation diagnostics; shared-cache fallback rate={framework_stats['failure_rate']:.1%}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostics.png", dpi=180)
    plt.close(fig)


def output_inventory(output_dir: Path) -> dict[str, dict]:
    inventory = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.endswith(".tmp"):
            inventory[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
    return inventory


def verify_completed_output_integrity(output_dir: Path, manifest: dict) -> None:
    if manifest.get("status") != "completed":
        raise ValueError("completed output integrity can only be checked for completed runs")
    integrity = manifest.get("manifest_integrity")
    if not isinstance(integrity, dict):
        raise ValueError("completed run is missing manifest_integrity")
    payload = dict(manifest)
    payload.pop("manifest_integrity", None)
    if integrity.get("sha256") != sha256_json(payload):
        raise ValueError("run_manifest canonical payload hash mismatch")
    manifest_path = output_dir / "run_manifest.json"
    if integrity.get("size_bytes") != manifest_path.stat().st_size:
        raise ValueError("run_manifest size mismatch")

    inventory = manifest.get("outputs")
    if not isinstance(inventory, dict) or not inventory:
        raise ValueError("completed run is missing its output inventory")
    actual_names = {
        path.name for path in output_dir.iterdir() if path.is_file() and path.name != "run_manifest.json"
    }
    if set(inventory) != actual_names:
        raise ValueError("completed output inventory does not match files on disk")
    for name, expected in inventory.items():
        if Path(name).name != name or not isinstance(expected, dict):
            raise ValueError(f"invalid output inventory entry: {name}")
        path = output_dir / name
        if expected.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"completed output size mismatch: {name}")
        if expected.get("sha256") != file_sha256(path):
            raise ValueError(f"completed output hash mismatch: {name}")


def canonicalize_predictions(
    path: Path,
    rows_by_cell: dict[str, list[dict]],
    cell_order: list[str],
    records: list[dict],
) -> None:
    ordered_rows = []
    expected_ids = [record["example_id"] for record in records]
    for cell in cell_order:
        indexed = {row["example_id"]: row for row in rows_by_cell[cell]}
        if len(indexed) != len(rows_by_cell[cell]) or set(indexed) != set(expected_ids):
            raise ValueError(f"cannot canonicalize incomplete or duplicate prediction cell: {cell}")
        ordered_rows.extend(indexed[example_id] for example_id in expected_ids)
    write_jsonl(path, ordered_rows)


def finalize_manifest(output_dir: Path, manifest: dict) -> None:
    manifest["status"] = "completed"
    manifest["completed_at"] = utc_now()
    manifest["outputs"] = output_inventory(output_dir)
    manifest["output_inventory_scope"] = "all result files except the self-referential run_manifest.json"
    payload = dict(manifest)
    payload.pop("manifest_integrity", None)
    manifest["manifest_integrity"] = {
        "hash_scope": "canonical JSON payload excluding manifest_integrity",
        "sha256": sha256_json(payload),
        "size_bytes": 0,
    }
    write_json(output_dir / "run_manifest.json", manifest)
    for _ in range(3):
        size = (output_dir / "run_manifest.json").stat().st_size
        if manifest["manifest_integrity"]["size_bytes"] == size:
            break
        manifest["manifest_integrity"]["size_bytes"] = size
        write_json(output_dir / "run_manifest.json", manifest)


def run_evaluation(config: dict, output_dir: Path, manifest: dict, *, resume: bool) -> None:
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"], local_files_only=True, trust_remote_code=True
    )
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
    framework_tokenizer = AutoTokenizer.from_pretrained(
        config["teacher_model"], local_files_only=True, trust_remote_code=True
    )
    if framework_tokenizer.pad_token_id is None:
        framework_tokenizer.pad_token = framework_tokenizer.eos_token

    records = select_records(config["dataset"], int(config["limit"]), int(config["seed"]))
    if not records:
        raise ValueError("evaluation dataset selection is empty")
    selection = [(row["example_id"], row["question"], row["answer"]) for row in records]
    selection_sha256 = sha256_json(selection)
    if resume and manifest.get("selection_sha256") not in {None, selection_sha256}:
        raise ValueError("selected evaluation examples do not match the resumable run")
    manifest["selected_examples"] = len(records)
    manifest["selection_sha256"] = selection_sha256
    write_json(output_dir / "run_manifest.json", manifest)

    framework_adapter_sha = manifest["provenance"]["training_artifacts"]["framework_teacher"]["adapter"]["sha256"]
    previous_framework_hash = manifest.get("framework_cache_sha256")
    previous_complete = bool(manifest.get("framework_cache_complete", False))
    framework_cache, framework_stats = generate_framework_cache(
        config,
        framework_tokenizer,
        records,
        output_dir,
        framework_adapter_sha,
        resume=resume,
    )
    framework_hash = file_sha256(output_dir / "framework_cache.jsonl")
    if resume and previous_complete and previous_framework_hash != framework_hash:
        raise ValueError("completed framework cache fingerprint changed")
    manifest["framework_generation"] = framework_stats
    manifest["framework_cache_sha256"] = framework_hash
    manifest["framework_cache_complete"] = len(framework_cache) == len(records)
    write_json(output_dir / "run_manifest.json", manifest)

    rows_by_cell = evaluate_cells(
        config,
        student_tokenizer,
        records,
        framework_cache,
        output_dir,
        manifest,
        resume=resume,
    )
    cell_order = ordered_cells(bool(config.get("include_base", False)))
    if any(len(rows_by_cell[cell]) != len(records) for cell in cell_order):
        raise RuntimeError("evaluation ended with incomplete cells")
    canonicalize_predictions(
        output_dir / "predictions.jsonl", rows_by_cell, cell_order, records
    )
    summaries = write_accuracy(output_dir, rows_by_cell, cell_order)
    comparisons = build_comparisons(config, rows_by_cell)
    write_paired_outputs(output_dir, comparisons, config)
    plot_grouped_accuracy(output_dir, summaries, bool(config.get("include_base", False)))
    plot_paired_deltas(output_dir, comparisons)
    plot_paired_outcomes(output_dir, comparisons)
    plot_accuracy_vs_cost(output_dir, summaries)
    plot_diagnostics(output_dir, summaries, framework_stats)
    finalize_manifest(output_dir, manifest)
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


def resolve_project_paths(config: dict, project_root: Path | None = None) -> dict:
    """Resolve project-owned config paths without depending on the caller's cwd."""
    root = (project_root or Path(__file__).resolve().parent).resolve()
    resolved = dict(config)
    for key in (
        "student_model",
        "teacher_model",
        "framework_teacher_adapter",
        "dataset",
        "output_dir",
    ):
        value = resolved.get(key)
        if value:
            path = Path(value)
            resolved[key] = str(path if path.is_absolute() else (root / path).resolve())
    adapters = resolved.get("adapters")
    if isinstance(adapters, dict):
        resolved["adapters"] = {
            name: str(path if (path := Path(value)).is_absolute() else (root / path).resolve())
            for name, value in adapters.items()
        }
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a provenance-locked paired 2×2 OPD evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="resume only after immutable provenance checks")
    args = parser.parse_args()
    config_path = Path(args.config)
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("evaluation config must be a JSON object")
    repo_root = Path(__file__).resolve().parent
    config = resolve_project_paths(config, repo_root)
    if args.resume:
        config["resume"] = True
    validate_config(config)

    provenance = collect_provenance(config, repo_root)
    signature = experiment_signature(config)
    output_dir = Path(config["output_dir"])
    resume = bool(config.get("resume", False))
    has_existing_output = output_dir.exists() and any(output_dir.iterdir())
    manifest_path = output_dir / "run_manifest.json"
    if has_existing_output and not resume:
        raise FileExistsError(
            f"output_dir is not empty: {output_dir}; use --resume only for this exact run"
        )
    if resume and not has_existing_output:
        raise FileNotFoundError("cannot resume because output_dir is empty")

    if has_existing_output:
        if not manifest_path.is_file():
            raise ValueError("cannot safely resume without run_manifest.json")
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("run manifest protocol is incompatible with safe resume")
        if manifest.get("status") == "completed":
            # This must precede provenance validation, resume-history updates, and
            # every cache/prediction rewrite so a damaged completed run is immutable.
            verify_completed_output_integrity(output_dir, manifest)
        validate_resume_provenance(manifest, signature, provenance)
        manifest.setdefault("resume_history", []).append(
            {
                "resumed_at": utc_now(),
                "runtime": provenance["runtime"],
                "source_sha256": provenance["source"]["source_sha256"],
            }
        )
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.pop("failed_at", None)
        manifest.pop("completed_at", None)
        manifest.pop("outputs", None)
        manifest.pop("output_inventory_scope", None)
        manifest.pop("manifest_integrity", None)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": utc_now(),
            "experiment_signature": signature,
            "config_path": str(config_path.resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config": config,
            "provenance": provenance,
            "metric": "strict answer on the last non-empty physical line only",
            "relaxed_metric": "last-number fallback (diagnostic only)",
            "token_cost_proxy": "student prompt+output; with-framework adds every 4B framework attempt prompt+output token",
            "framework_reuse": "one question-only framework cache entry shared by every with-framework cell",
            "framework_effect_estimand": "framework-conditioned prompt bundle (framework plus its system instruction), not framework text alone",
            "primary_comparisons": [
                "guided_minus_vanilla_no_framework",
                "guided_minus_vanilla_with_framework",
            ],
            "cells": ordered_cells(bool(config.get("include_base", False))),
            "completed_cells": [],
            "partial_cells": {},
            "resume_history": [],
        }
    write_json(manifest_path, manifest)
    try:
        run_evaluation(config, output_dir, manifest, resume=resume)
    except Exception as exception:
        manifest["status"] = "failed"
        manifest["failed_at"] = utc_now()
        manifest["error"] = f"{type(exception).__name__}: {exception}"
        write_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
