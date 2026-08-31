import argparse
import hashlib
import json
import os
import random
import tempfile
import uuid
from pathlib import Path

from framework_opd.framework_validation import audit_framework_records, require_valid_framework
from framework_opd.prompts import format_rollout_framework_prompt


ADAPTER_ARTIFACT_TYPE = "framework_teacher_adapter"
FRAMEWORK_TEACHER_ROLE = "framework_teacher"
BASE_MODEL_METADATA_NAMES = {
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
}
BASE_MODEL_WEIGHT_SUFFIXES = {".bin", ".pt", ".safetensors"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lightweight_model_identity(path: str | Path) -> dict:
    """Identify a local base model without hashing its multi-gigabyte weight shards."""
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"base model directory does not exist: {resolved}")
    metadata: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        relative = item.relative_to(resolved).as_posix()
        if item.name in BASE_MODEL_METADATA_NAMES:
            metadata[relative] = {"size": item.stat().st_size, "sha256": _sha256_file(item)}
        elif item.suffix.lower() in BASE_MODEL_WEIGHT_SUFFIXES:
            stat = item.stat()
            weights[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not metadata or not weights:
        raise ValueError(f"base model identity is incomplete: {resolved}")
    payload = {"path": str(resolved), "metadata": metadata, "weights": weights}
    return {**payload, "sha256": _canonical_sha256(payload)}


def adapter_artifact_identity(output: str | Path) -> dict:
    """Hash the complete PEFT adapter payload used by downstream consumers."""
    root = Path(output).resolve()
    config_path = root / "adapter_config.json"
    weight_paths = sorted(item for item in root.glob("adapter_model.*") if item.is_file())
    if not config_path.is_file() or not weight_paths:
        raise ValueError(f"adapter artifact is incomplete: {root}")
    files: dict[str, dict] = {}
    fingerprint_entries: list[dict] = []
    for item in [config_path, *weight_paths]:
        size = item.stat().st_size
        digest = _sha256_file(item)
        files[item.name] = {"size": size, "sha256": digest}
        fingerprint_entries.append(
            {"relative_path": item.name, "size_bytes": size, "sha256": digest}
        )
    # Keep this digest identical to framework_opd.evaluation.artifact_fingerprint
    # so every producer and consumer verifies the same adapter bytes.
    fingerprint_sha256 = hashlib.sha256(
        json.dumps(
            fingerprint_entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {"files": files, "sha256": fingerprint_sha256}


def validate_generation_audit(
    data_path: str | Path,
    generation_audit_path: str | Path,
    actual_records: int,
) -> dict:
    """Bind Teacher training data to a successfully published generation audit."""
    data = Path(data_path)
    if data.name.endswith(".partial"):
        raise ValueError("refusing to train on a partial framework dataset")
    audit_path = Path(generation_audit_path)
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid generation audit JSON: {error.msg}") from error
    if not isinstance(audit, dict):
        raise ValueError("generation audit must contain a JSON object")
    if audit.get("status") != "complete":
        raise ValueError("generation audit status must be complete")
    requested = audit.get("requested_valid")
    valid = audit.get("valid")
    if requested != valid or valid != actual_records:
        raise ValueError(
            "generation audit counts must satisfy requested_valid == valid == actual records"
        )
    data_sha256 = _sha256_file(data)
    if audit.get("output_sha256") != data_sha256:
        raise ValueError("generation audit output_sha256 does not match framework data")
    if audit.get("data_sha256") not in (None, data_sha256):
        raise ValueError("generation audit data_sha256 does not match framework data")
    declared_output = audit.get("output")
    if not isinstance(declared_output, str) or not declared_output.strip():
        raise ValueError("generation audit is missing its formal output path")
    if Path(declared_output).resolve() != data.resolve():
        raise ValueError("generation audit output path does not identify the training data")
    if audit.get("partial_output"):
        raise ValueError("complete generation audit must not reference a partial output")
    semantic_checks = audit.get("semantic_checks")
    semantic_passes = audit.get("semantic_passes")
    semantic_failures = audit.get("semantic_failures")
    if (
        audit.get("schema_version") != 3
        or not isinstance(semantic_checks, int)
        or semantic_checks < actual_records
        or semantic_passes != actual_records
        or semantic_failures != semantic_checks - semantic_passes
    ):
        raise ValueError(
            "generation audit semantic counts must prove every published label passed"
        )
    return audit


def prepare_empty_output_directory(output: str | Path) -> Path:
    """Create a new output directory, refusing to reuse any non-empty artifact."""
    path = Path(output)
    if path.exists():
        if not path.is_dir() or next(path.iterdir(), None) is not None:
            raise FileExistsError(f"teacher output must be absent or empty: {path}")
    else:
        path.mkdir(parents=True)
    return path


def _atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _write_run_metadata(output: Path, run_config: dict) -> None:
    if run_config.get("artifact_type") != ADAPTER_ARTIFACT_TYPE:
        raise ValueError("run_config artifact_type mismatch")
    if run_config.get("role") != FRAMEWORK_TEACHER_ROLE:
        raise ValueError("run_config role mismatch")
    artifact = adapter_artifact_identity(output)
    if run_config.get("adapter_artifact") != artifact:
        raise ValueError("run_config adapter artifact does not match saved adapter files")
    if run_config.get("adapter_artifact_sha256") != artifact["sha256"]:
        raise ValueError("run_config adapter artifact SHA does not match saved adapter files")
    run_id = run_config.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_config requires a non-empty run_id")
    run_config_path = output / "run_config.json"
    _atomic_write_json(run_config_path, run_config)
    _atomic_write_json(
        output / "RUN_COMPLETE",
        {
            "status": "complete",
            "artifact_type": ADAPTER_ARTIFACT_TYPE,
            "role": FRAMEWORK_TEACHER_ROLE,
            "run_id": run_id,
            "run_config_sha256": _sha256_file(run_config_path),
            "adapter_artifact_sha256": artifact["sha256"],
        },
    )


def verify_teacher_artifact(output: str | Path) -> dict:
    """Verify completion metadata and detect adapter or run-config replacement."""
    root = Path(output)
    run_config_path = root / "run_config.json"
    marker_path = root / "RUN_COMPLETE"
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid teacher artifact metadata JSON: {error.msg}") from error
    if not isinstance(run_config, dict) or not isinstance(marker, dict):
        raise ValueError("teacher artifact metadata must contain JSON objects")
    if marker.get("status") != "complete":
        raise ValueError("teacher artifact completion marker is not complete")
    for metadata in (run_config, marker):
        if metadata.get("artifact_type") != ADAPTER_ARTIFACT_TYPE:
            raise ValueError("teacher artifact type mismatch")
        if metadata.get("role") != FRAMEWORK_TEACHER_ROLE:
            raise ValueError("teacher artifact role mismatch")
    run_id = run_config.get("run_id")
    if not isinstance(run_id, str) or not run_id or marker.get("run_id") != run_id:
        raise ValueError("teacher artifact run_id mismatch")
    if marker.get("run_config_sha256") != _sha256_file(run_config_path):
        raise ValueError("teacher run_config hash mismatch")
    artifact = adapter_artifact_identity(root)
    if run_config.get("adapter_artifact") != artifact:
        raise ValueError("saved adapter files do not match run_config manifest")
    if run_config.get("adapter_artifact_sha256") != artifact["sha256"]:
        raise ValueError("saved adapter files do not match run_config artifact SHA")
    if marker.get("adapter_artifact_sha256") != artifact["sha256"]:
        raise ValueError("saved adapter files do not match completion marker")
    return {"run_config": run_config, "completion": marker, "adapter_artifact": artifact}


def load_framework_training_records(
    path: str | Path, expected_records: int | None = None
) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on framework data line {line_number}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"framework data line {line_number} must contain a JSON object")
            records.append(record)

    if not records:
        raise ValueError("framework training data is empty")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(
            f"framework training data has {len(records)} records; expected exactly {expected_records}"
        )
    audit = audit_framework_records(records)
    if audit["invalid"]:
        raise ValueError(
            "framework training data failed purity validation before model loading: "
            + json.dumps(audit, ensure_ascii=False)
        )

    clean_records: list[dict] = []
    for record in records:
        semantic = record.get("semantic_verification")
        if not isinstance(semantic, dict) or semantic.get("correct") is not True:
            raise ValueError("framework training record is missing successful semantic verification")
        clean_records.append(
            {
                **record,
                "question": str(record["question"]).strip(),
                "answer": str(record["answer"]).strip(),
                "framework": require_valid_framework(record["framework"], str(record["answer"])),
            }
        )
    return clean_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen3 teacher to emit solution frameworks")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--generation-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-records", type=int)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.expected_records is not None and args.expected_records <= 0:
        parser.error("--expected-records must be positive")

    # Validate the complete dataset before allocating either model weights or GPU memory.
    data_path = Path(args.data)
    if data_path.name.endswith(".partial"):
        raise ValueError("refusing to train on a partial framework dataset")
    records = load_framework_training_records(data_path, expected_records=args.expected_records)
    purity_audit = audit_framework_records(records)
    data_sha256 = _sha256_file(data_path)
    generation_audit_path = Path(args.generation_audit)
    generation_audit = validate_generation_audit(
        data_path, generation_audit_path, len(records)
    )
    output = prepare_empty_output_directory(args.output)
    base_model_identity = lightweight_model_identity(args.model)
    run_id = uuid.uuid4().hex
    random.Random(args.seed).shuffle(records)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16
    ).to(args.device)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    for step in range(args.steps):
        record = records[step % len(records)]
        prompt = format_rollout_framework_prompt(record["question"])
        target = "\n".join(f"{index}. {item}" for index, item in enumerate(record["framework"], 1))
        target = target + "\n</framework>"
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        input_ids = torch.tensor([prompt_ids + target_ids], device=args.device)
        labels = torch.tensor([[-100] * len(prompt_ids) + target_ids], device=args.device)
        loss = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), labels=labels).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": step + 1, "loss": loss.item()}))

    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    adapter_artifact = adapter_artifact_identity(output)
    run_config = {
        "schema_version": 3,
        "artifact_type": ADAPTER_ARTIFACT_TYPE,
        "role": FRAMEWORK_TEACHER_ROLE,
        "run_id": run_id,
        "base_model": args.model,
        "base_model_identity": base_model_identity,
        "data": str(data_path.resolve()),
        "data_sha256": data_sha256,
        "generation_audit": {
            "path": str(generation_audit_path.resolve()),
            "sha256": _sha256_file(generation_audit_path),
            "status": generation_audit["status"],
            "requested_valid": generation_audit["requested_valid"],
            "valid": generation_audit["valid"],
            "output_sha256": generation_audit["output_sha256"],
            "semantic_checks": generation_audit["semantic_checks"],
            "semantic_passes": generation_audit["semantic_passes"],
            "semantic_failures": generation_audit["semantic_failures"],
        },
        "purity_audit": purity_audit,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "expected_records": args.expected_records,
        "num_records": len(records),
        "adapter_artifact": adapter_artifact,
        "adapter_artifact_sha256": adapter_artifact["sha256"],
    }
    _write_run_metadata(output, run_config)
    verify_teacher_artifact(output)


if __name__ == "__main__":
    main()
