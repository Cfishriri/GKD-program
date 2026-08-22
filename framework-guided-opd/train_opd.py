import argparse
import hashlib
import json
import os
import platform
import random
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESUME_SIGNATURE_KEYS = (
    "student_model",
    "teacher_model",
    "framework_teacher_adapter",
    "dataset",
    "seed",
    "max_steps",
    "beta",
    "temperature",
    "gradient_accumulation_steps",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "framework_max_new_tokens",
    "framework_max_attempts",
    "solution_max_new_tokens",
    "generation_temperature",
)

SIGNATURE_SOURCE_FILES = (
    "train_opd.py",
    "src/framework_opd/data.py",
    "src/framework_opd/framework_validation.py",
    "src/framework_opd/loss.py",
    "src/framework_opd/masking.py",
    "src/framework_opd/prompts.py",
    "src/framework_opd/rollout.py",
)

MODEL_METADATA_NAMES = {
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}

MODEL_WEIGHT_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _dataset_identity(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"dataset does not exist: {path}")
    resolved = path.resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "size": resolved.stat().st_size,
        }

    files = [item for item in resolved.rglob("*") if item.is_file()]
    if not files:
        raise ValueError(f"dataset directory is empty: {resolved}")
    members = {
        item.relative_to(resolved).as_posix(): _sha256_file(item)
        for item in sorted(files)
    }
    return {
        "path": str(resolved),
        "sha256": _canonical_sha256(members),
        "files": members,
    }


def _is_model_metadata(path: Path) -> bool:
    name = path.name
    return (
        name in MODEL_METADATA_NAMES
        or name.endswith(".index.json")
        or name.startswith("tokenizer")
    )


def _model_identity(path_value: str) -> dict:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"local model does not exist: {path}")
    resolved = path.resolve()
    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "files": {
                resolved.name: {
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "sha256": _sha256_file(resolved),
                }
            },
        }

    metadata: dict[str, dict] = {}
    weights: dict[str, dict] = {}
    other_files: dict[str, dict] = {}
    for item in sorted(candidate for candidate in resolved.rglob("*") if candidate.is_file()):
        relative = item.relative_to(resolved).as_posix()
        stat = item.stat()
        if _is_model_metadata(item):
            metadata[relative] = {
                "size": stat.st_size,
                "sha256": _sha256_file(item),
            }
        elif item.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
            # Multi-GB base weights are identified without rereading every byte.
            weights[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        else:
            other_files[relative] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not metadata and not weights and not other_files:
        raise ValueError(f"local model directory is empty: {resolved}")
    return {
        "path": str(resolved),
        "metadata": metadata,
        "weights": weights,
        "other_files": other_files,
    }


def _adapter_identity(path_value: str) -> dict:
    path = Path(path_value)
    if not path.is_dir():
        raise FileNotFoundError(f"framework teacher adapter does not exist: {path}")
    resolved = path.resolve()
    selected = [
        item
        for item in resolved.rglob("*")
        if item.is_file()
        and (item.name == "adapter_config.json" or item.name.startswith("adapter_model"))
    ]
    if not selected:
        raise ValueError(f"framework teacher adapter has no config or weights: {resolved}")
    files = {}
    for item in sorted(selected):
        relative = item.relative_to(resolved).as_posix()
        files[relative] = {"size": item.stat().st_size, "sha256": _sha256_file(item)}
    return {"path": str(resolved), "files": files}


def _adapter_artifact_sha256(path: Path) -> str:
    """Match framework_opd.evaluation.artifact_fingerprint for adapter directories."""
    if not path.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {path}")
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file()
        and (
            candidate.name == "adapter_config.json"
            or candidate.name.startswith("adapter_model.")
        )
    )
    if not files:
        raise ValueError(f"adapter directory contains no adapter weights/config: {path}")
    entries = [
        {
            "relative_path": file.relative_to(path).as_posix(),
            "size_bytes": file.stat().st_size,
            "sha256": _sha256_file(file),
        }
        for file in files
    ]
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _device_topology(devices: dict[str, str]) -> dict:
    normalized = {role: str(value).strip().lower() for role, value in devices.items()}
    types = {role: value.split(":", 1)[0] for role, value in normalized.items()}
    supported = {"cpu", "cuda"}
    if set(types) != {"student", "teacher"} or any(
        device_type not in supported for device_type in types.values()
    ):
        raise ValueError(f"unsupported RNG device topology: {devices}")
    return {
        "types": types,
        "student_teacher_shared": normalized["student"] == normalized["teacher"],
    }


def _validate_device_topology(saved: dict | None, current_devices: dict[str, str]) -> None:
    current = _device_topology(current_devices)
    if saved != current:
        raise ValueError(
            "resume device topology mismatch: only CUDA device-number migration is allowed; "
            f"checkpoint={saved}, current={current}"
        )


def _implementation_identity(project_root: Path | None = None) -> dict[str, str]:
    root = (project_root or Path(__file__).resolve().parent).resolve()
    identity: dict[str, str] = {}
    for relative in SIGNATURE_SOURCE_FILES:
        source_path = root / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"signature source file does not exist: {source_path}")
        identity[relative] = _sha256_file(source_path)
    return identity


def _resume_signature(config: dict, mode: str, project_root: Path | None = None) -> dict:
    framework_adapter = None
    if mode == "guided" and config.get("framework_teacher_adapter"):
        framework_adapter = _adapter_identity(config["framework_teacher_adapter"])
    payload = {
        "schema_version": 2,
        "mode": mode,
        "hyperparameters": {key: config.get(key) for key in RESUME_SIGNATURE_KEYS},
        "dataset": _dataset_identity(config["dataset"]),
        "models": {
            "student": _model_identity(config["student_model"]),
            "teacher": _model_identity(config["teacher_model"]),
        },
        "framework_teacher_adapter": framework_adapter,
        "implementation": _implementation_identity(project_root),
    }
    return {"sha256": _canonical_sha256(payload), "payload": payload}


def _runtime_snapshot(config: dict) -> dict:
    """Record runtime context without making device topology part of the signature."""
    snapshot = {
        "timestamp": _utc_now(),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "platform": platform.platform(),
        "configured_devices": {
            "student": config.get("student_device"),
            "teacher": config.get("teacher_device"),
        },
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_device_count"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        snapshot["torch"] = None
        snapshot["cuda_available"] = False
        snapshot["cuda_device_count"] = 0
    return snapshot


def _validate_positive_max_steps(value: Any) -> int:
    max_steps = int(value)
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    return max_steps


def _validate_nonempty_records(records: list[dict], dataset_path: str) -> None:
    if not records:
        raise ValueError(f"dataset contains no records: {dataset_path}")


def _validate_new_output_lifecycle(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to reuse non-empty output directory: {output_dir}")


def _load_run_manifest(output_dir: Path) -> dict:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"run manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"run manifest is invalid JSON: {manifest_path}") from error
    if not isinstance(manifest, dict) or not manifest.get("run_id"):
        raise ValueError(f"run manifest has no run_id: {manifest_path}")
    if manifest.get("schema_version") != 2 or manifest.get("role") != "opd_student":
        raise ValueError(f"run manifest has incompatible provenance schema: {manifest_path}")
    return manifest


def _initialize_new_run(output_dir: Path, config: dict, mode: str, run_signature: dict) -> dict:
    _validate_new_output_lifecycle(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    run_config = {
        **config,
        "role": "opd_student",
        "run_id": run_id,
        "mode": mode,
    }
    run_config_path = output_dir / "run_config.json"
    _atomic_write_json(run_config_path, run_config)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "role": "opd_student",
        "status": "running",
        "created_at": _utc_now(),
        "run_signature": run_signature,
        "run_config_sha256": _sha256_file(run_config_path),
        "initial_runtime": _runtime_snapshot(config),
        "resume_history": [],
    }
    _atomic_write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def _validate_resume_checkpoint_location(checkpoint_dir: Path, output_dir: Path) -> Path:
    if not output_dir.is_dir():
        raise FileNotFoundError(f"resume output directory does not exist: {output_dir}")
    if (output_dir / "RUN_COMPLETE").exists():
        raise ValueError(f"refusing to resume a completed run: {output_dir}")
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {checkpoint_dir}")
    checkpoints_root = (output_dir / "checkpoints").resolve()
    resolved_checkpoint = checkpoint_dir.resolve()
    if resolved_checkpoint.parent != checkpoints_root:
        raise ValueError(
            f"resume checkpoint must be directly inside {output_dir / 'checkpoints'}: "
            f"{checkpoint_dir}"
        )
    return resolved_checkpoint


def _validate_manifest_signature(manifest: dict, expected_signature: dict) -> None:
    actual_signature = manifest.get("run_signature")
    if actual_signature != expected_signature:
        actual_hash = (actual_signature or {}).get("sha256", "missing")
        raise ValueError(
            "resume run signature mismatch: "
            f"manifest={actual_hash}, current={expected_signature.get('sha256', 'missing')}"
        )


def _validate_run_config_provenance(output_dir: Path, manifest: dict) -> str:
    run_config_path = output_dir / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(f"run_config does not exist: {run_config_path}")
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"run_config is invalid JSON: {run_config_path}") from error
    if (
        run_config.get("run_id") != manifest.get("run_id")
        or run_config.get("role") != "opd_student"
    ):
        raise ValueError("run_config run_id/role provenance mismatch")
    actual_sha256 = _sha256_file(run_config_path)
    if manifest.get("run_config_sha256") != actual_sha256:
        raise ValueError("run_config fingerprint does not match immutable manifest provenance")
    return actual_sha256


def _append_resume_history(
    output_dir: Path,
    manifest: dict,
    config: dict,
    checkpoint_dir: Path,
    rollout_step: int,
    *,
    preserve_status: bool = False,
) -> dict:
    updated = json.loads(json.dumps(manifest))
    updated.setdefault("resume_history", []).append(
        {
            "checkpoint": str(checkpoint_dir),
            "rollout_step": rollout_step,
            "runtime": _runtime_snapshot(config),
        }
    )
    if not preserve_status:
        updated["status"] = "running"
    _atomic_write_json(output_dir / "run_manifest.json", updated)
    return updated


def _validate_completion_recovery(
    manifest: dict,
    state: dict,
    final_rollout_step: int,
) -> None:
    if manifest.get("status") != "complete":
        return
    if int(manifest.get("final_rollout_step", -1)) != final_rollout_step:
        raise ValueError("completed manifest final_rollout_step is inconsistent")
    if int(state.get("rollout_step", -1)) != final_rollout_step:
        raise ValueError(
            "a completed manifest without RUN_COMPLETE may only resume from its final checkpoint"
        )
    if int(state.get("optimizer_step", -1)) != int(
        manifest.get("final_optimizer_step", -2)
    ):
        raise ValueError("completed manifest/checkpoint optimizer_step mismatch")
    expected_adapter_sha = manifest.get("adapter_artifact_sha256")
    if expected_adapter_sha and state.get("adapter_artifact_sha256") != expected_adapter_sha:
        raise ValueError("completed manifest/checkpoint adapter fingerprint mismatch")


def _completed_adapter_matches(manifest: dict, output_dir: Path) -> bool:
    expected = manifest.get("adapter_artifact_sha256")
    if not expected:
        return False
    try:
        actual = _adapter_artifact_sha256(output_dir / "student_adapter")
    except (FileNotFoundError, ValueError):
        return False
    return actual == expected


def _mark_run_complete(
    output_dir: Path,
    run_id: str,
    *,
    rollout_step: int,
    optimizer_step: int,
) -> dict:
    manifest = _load_run_manifest(output_dir)
    if manifest.get("run_id") != run_id:
        raise ValueError("cannot complete run: manifest run_id mismatch")
    if manifest.get("status") not in {"running", "complete"}:
        raise ValueError(f"cannot complete run from status={manifest.get('status')!r}")
    run_config_sha256 = _validate_run_config_provenance(output_dir, manifest)
    adapter_artifact_sha256 = _adapter_artifact_sha256(output_dir / "student_adapter")
    if manifest.get("status") == "complete":
        if int(manifest.get("final_rollout_step", -1)) != rollout_step:
            raise ValueError("cannot repair completion: final_rollout_step mismatch")
        if int(manifest.get("final_optimizer_step", -1)) != optimizer_step:
            raise ValueError("cannot repair completion: final_optimizer_step mismatch")
        expected_adapter_sha = manifest.get("adapter_artifact_sha256")
        if expected_adapter_sha and expected_adapter_sha != adapter_artifact_sha256:
            raise ValueError("cannot repair completion: student adapter fingerprint mismatch")
    completed_at = manifest.get("completed_at") or _utc_now()
    updated = json.loads(json.dumps(manifest))
    updated.update(
        {
            "status": "complete",
            "completed_at": completed_at,
            "final_rollout_step": rollout_step,
            "final_optimizer_step": optimizer_step,
            "run_config_sha256": run_config_sha256,
            "adapter_artifact_sha256": adapter_artifact_sha256,
        }
    )
    completion = {
        "run_id": run_id,
        "role": "opd_student",
        "status": "complete",
        "run_config_sha256": run_config_sha256,
        "adapter_artifact_sha256": adapter_artifact_sha256,
        "completed_at": completed_at,
        "rollout_step": rollout_step,
        "optimizer_step": optimizer_step,
    }
    completion_path = output_dir / "RUN_COMPLETE"
    if completion_path.exists():
        try:
            existing_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"completion marker is invalid JSON: {completion_path}") from error
        if existing_completion != completion:
            raise ValueError("existing completion marker conflicts with final run provenance")
        if manifest != updated:
            _atomic_write_json(output_dir / "run_manifest.json", updated)
        return updated
    _atomic_write_json(output_dir / "run_manifest.json", updated)
    _atomic_write_json(completion_path, completion)
    return updated


def _unique_sibling_path(target: Path, label: str) -> Path:
    while True:
        candidate = target.with_name(f".{target.name}.{label}.{uuid.uuid4().hex}")
        if not candidate.exists():
            return candidate


def _publish_student_adapter(
    student: Any,
    tokenizer: Any,
    output_dir: Path,
    *,
    allow_existing: bool,
) -> dict:
    target = output_dir / "student_adapter"
    if target.exists() and not allow_existing:
        raise FileExistsError(
            f"refusing to replace existing final adapter outside resume-finalization: {target}"
        )
    temporary = _unique_sibling_path(target, "tmp")
    student.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    adapter_sha256 = _adapter_artifact_sha256(temporary)

    evidence_path = None
    if target.exists():
        evidence_path = _unique_sibling_path(target, "before-finalization")
        target.replace(evidence_path)
    try:
        temporary.replace(target)
    except BaseException:
        if evidence_path is not None and evidence_path.exists() and not target.exists():
            evidence_path.replace(target)
        raise
    return {
        "path": str(target),
        "sha256": adapter_sha256,
        "preserved_previous_path": str(evidence_path) if evidence_path is not None else None,
    }


def _move_optimizer_state(optimizer: Any) -> None:
    """Move moment tensors to their parameters without breaking AdamW step state."""
    import torch

    parameter_groups = {
        parameter: group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter, state in optimizer.state.items():
        group = parameter_groups[parameter]
        step_on_parameter = bool(group.get("capturable", False) or group.get("fused", False))
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            target_device = parameter.device
            if name == "step" and not step_on_parameter:
                target_device = torch.device("cpu")
            state[name] = value.to(target_device)


def _available_checkpoint_dir(output_dir: Path, rollout_step: int) -> Path:
    base = output_dir / "checkpoints" / f"checkpoint-{rollout_step:06d}"
    candidate = base
    suffix = 1
    while candidate.exists() or candidate.with_name(candidate.name + ".tmp").exists():
        candidate = base.with_name(f"{base.name}-resume-{suffix}")
        suffix += 1
    return candidate


def _save_checkpoint(
    student: Any,
    optimizer: Any,
    checkpoint_dir: Path,
    *,
    rollout_step: int,
    optimizer_step: int,
    run_id: str,
    run_signature: dict,
    rng_devices: dict[str, str],
) -> Path:
    import torch

    temporary_dir = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    temporary_dir.mkdir(parents=True, exist_ok=False)
    try:
        student.save_pretrained(temporary_dir)
        checkpoint_adapter_sha256 = _adapter_artifact_sha256(temporary_dir)
        state_path = temporary_dir / "training_state.pt"
        temporary_state_path = temporary_dir / "training_state.pt.tmp"
        cuda_random_states_by_role = {}
        if torch.cuda.is_available():
            for role, device_value in rng_devices.items():
                device = torch.device(device_value)
                if device.type == "cuda":
                    cuda_random_states_by_role[role] = torch.cuda.get_rng_state(device)
        torch.save(
            {
                "run_id": run_id,
                "rollout_step": rollout_step,
                "optimizer_step": optimizer_step,
                "run_signature": run_signature,
                "adapter_artifact_sha256": checkpoint_adapter_sha256,
                "optimizer": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                # Logical roles let a resume move from cuda:N to cuda:M safely.
                "cuda_random_states_by_role": cuda_random_states_by_role,
                "device_topology": _device_topology(rng_devices),
            },
            temporary_state_path,
        )
        temporary_state_path.replace(state_path)
        _atomic_write_json(
            temporary_dir / "CHECKPOINT_COMPLETE",
            {
                "run_id": run_id,
                "rollout_step": rollout_step,
                "adapter_artifact_sha256": checkpoint_adapter_sha256,
            },
        )
        temporary_dir.replace(checkpoint_dir)
    except BaseException:
        # Leave an incomplete .tmp directory as evidence; it is never accepted for resume.
        raise
    return checkpoint_dir


def _load_training_state(
    checkpoint_dir: Path,
    expected_signature: dict,
    expected_run_id: str,
) -> dict:
    marker_path = checkpoint_dir / "CHECKPOINT_COMPLETE"
    if not marker_path.is_file():
        raise ValueError(f"resume checkpoint is incomplete: {checkpoint_dir}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"resume checkpoint has invalid completion marker: {checkpoint_dir}") from error
    if marker.get("run_id") != expected_run_id:
        raise ValueError("resume checkpoint completion marker run_id mismatch")
    state_path = checkpoint_dir / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"resume checkpoint is missing {state_path.name}: {checkpoint_dir}")

    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("run_id") != expected_run_id:
        raise ValueError("resume checkpoint training state run_id mismatch")
    if int(state.get("rollout_step", -1)) != int(marker.get("rollout_step", -2)):
        raise ValueError("resume checkpoint marker/state rollout_step mismatch")
    checkpoint_adapter_sha256 = _adapter_artifact_sha256(checkpoint_dir)
    if (
        state.get("adapter_artifact_sha256") != checkpoint_adapter_sha256
        or marker.get("adapter_artifact_sha256") != checkpoint_adapter_sha256
    ):
        raise ValueError("resume checkpoint adapter fingerprint mismatch")
    actual_signature = state.get("run_signature")
    if actual_signature != expected_signature:
        actual_hash = (actual_signature or {}).get("sha256", "missing")
        raise ValueError(
            "resume checkpoint run signature mismatch: "
            f"checkpoint={actual_hash}, current={expected_signature.get('sha256', 'missing')}"
        )
    return state


def _restore_random_state(state: dict, config: dict) -> None:
    import torch

    current_devices = {
        "student": config["student_device"],
        "teacher": config["teacher_device"],
    }
    _validate_device_topology(state.get("device_topology"), current_devices)
    random.setstate(state["python_random_state"])
    torch.set_rng_state(state["torch_random_state"])
    cuda_states_by_role = state.get("cuda_random_states_by_role", {})
    if torch.cuda.is_available():
        for role, cuda_state in cuda_states_by_role.items():
            device = torch.device(current_devices[role])
            if device.type == "cuda":
                torch.cuda.set_rng_state(cuda_state, device)


def _rewind_metrics_for_resume(
    metrics_path: Path,
    rollout_step: int,
    expected_run_id: str,
) -> None:
    """Validate provenance, then preserve and remove post-checkpoint metrics."""
    if not metrics_path.is_file():
        if rollout_step > 0:
            raise FileNotFoundError(f"resume metrics do not exist: {metrics_path}")
        return

    lines = metrics_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept: list[str] = []
    saw_checkpoint_step = rollout_step == 0
    removed = False
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSON at line {line_number}: {metrics_path}") from error
        if record.get("run_id") != expected_run_id:
            raise ValueError(f"metrics run_id mismatch at line {line_number}: {metrics_path}")
        record_step_value = record.get("rollout_step", record.get("step"))
        if record_step_value is None:
            raise ValueError(f"metrics line {line_number} has no rollout_step: {metrics_path}")
        record_step = int(record_step_value)
        if record_step == rollout_step:
            saw_checkpoint_step = True
        if record_step <= rollout_step:
            kept.append(line if line.endswith("\n") else line + "\n")
        else:
            removed = True
    if not saw_checkpoint_step:
        raise ValueError(
            f"metrics do not contain restored checkpoint step {rollout_step}: {metrics_path}"
        )
    if not removed:
        return

    backup_path = metrics_path.with_name(f"metrics.before-resume-{rollout_step:06d}.jsonl")
    suffix = 1
    while backup_path.exists():
        backup_path = metrics_path.with_name(
            f"metrics.before-resume-{rollout_step:06d}-{suffix}.jsonl"
        )
        suffix += 1
    original_text = "".join(lines)
    _atomic_write_text(backup_path, original_text)
    _atomic_write_text(metrics_path, "".join(kept))


def _normalize_training_config(config: dict) -> tuple[str, int]:
    mode = str(config.get("mode", "guided"))
    if mode not in {"guided", "vanilla"}:
        raise ValueError(f"unsupported OPD mode: {mode}")
    max_steps = _validate_positive_max_steps(config["max_steps"])
    accumulation_steps = int(config["gradient_accumulation_steps"])
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    framework_max_attempts = int(config.get("framework_max_attempts", 3))
    if framework_max_attempts <= 0:
        raise ValueError("framework_max_attempts must be positive")
    checkpoint_steps = int(config.get("checkpoint_steps", 100))
    if checkpoint_steps <= 0:
        raise ValueError("checkpoint_steps must be positive")
    config["max_steps"] = max_steps
    config["gradient_accumulation_steps"] = accumulation_steps
    config["framework_max_attempts"] = framework_max_attempts
    config["checkpoint_steps"] = checkpoint_steps
    config.setdefault("resume_from_checkpoint", None)
    return mode, max_steps


def _framework_rollout_metrics(rollout: Any) -> dict:
    return {
        "framework_attempts": rollout.framework_attempts,
        "framework_fallback": rollout.framework_fallback,
        "framework_validation_errors": rollout.framework_validation_errors,
        "framework_prompt_tokens": rollout.framework_prompt_tokens,
        "framework_generated_tokens": rollout.framework_generated_tokens,
        "framework_hit_max_attempts": rollout.framework_hit_max_attempts,
        "framework_last_ended_with_eos": rollout.framework_last_ended_with_eos,
        "framework_closed_tag": rollout.framework_closed_tag,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Framework-guided on-policy distillation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    mode, configured_max_steps = _normalize_training_config(config)

    output_dir = Path(config["output_dir"])
    resume_checkpoint = (
        Path(config["resume_from_checkpoint"])
        if config.get("resume_from_checkpoint")
        else None
    )
    if resume_checkpoint is None:
        _validate_new_output_lifecycle(output_dir)
    else:
        resume_checkpoint = _validate_resume_checkpoint_location(resume_checkpoint, output_dir)

    # Dataset and provenance checks deliberately happen before importing/loading models.
    from framework_opd.data import load_records

    records = load_records(config["dataset"])
    _validate_nonempty_records(records, config["dataset"])
    max_rollout_steps = min(configured_max_steps, len(records))
    run_signature = _resume_signature(config, mode)

    metrics_path = output_dir / "metrics.jsonl"
    resume_finalization = False
    if resume_checkpoint is None:
        manifest = _initialize_new_run(output_dir, config, mode, run_signature)
        run_id = manifest["run_id"]
        resume_state = None
    else:
        manifest = _load_run_manifest(output_dir)
        if manifest.get("status") not in {"running", "complete"}:
            raise ValueError(f"cannot resume manifest status={manifest.get('status')!r}")
        _validate_run_config_provenance(output_dir, manifest)
        _validate_manifest_signature(manifest, run_signature)
        run_id = manifest["run_id"]
        resume_state = _load_training_state(resume_checkpoint, run_signature, run_id)
        start_rollout_step = int(resume_state["rollout_step"])
        if start_rollout_step > max_rollout_steps:
            raise ValueError(
                f"checkpoint is at rollout {start_rollout_step}, "
                f"beyond requested max_steps {max_rollout_steps}"
            )
        current_rng_devices = {
            "student": config["student_device"],
            "teacher": config["teacher_device"],
        }
        _validate_device_topology(resume_state.get("device_topology"), current_rng_devices)
        _validate_completion_recovery(manifest, resume_state, max_rollout_steps)
        _rewind_metrics_for_resume(metrics_path, int(resume_state["rollout_step"]), run_id)
        was_complete = manifest.get("status") == "complete"
        manifest = _append_resume_history(
            output_dir,
            manifest,
            config,
            resume_checkpoint,
            int(resume_state["rollout_step"]),
            preserve_status=was_complete,
        )
        resume_finalization = start_rollout_step == max_rollout_steps
        if was_complete and _completed_adapter_matches(manifest, output_dir):
            _mark_run_complete(
                output_dir,
                run_id,
                rollout_step=max_rollout_steps,
                optimizer_step=int(resume_state["optimizer_step"]),
            )
            print(
                json.dumps(
                    {
                        "run_id": run_id,
                        "status": "completion_marker_recovered",
                        "rollout_step": max_rollout_steps,
                    }
                )
            )
            return

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from framework_opd.loss import generalized_jsd_loss
    from framework_opd.masking import causal_completion_mask
    from framework_opd.rollout import generate_opd_rollout

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["student_model"],
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        config["teacher_model"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(config["teacher_device"])
    teacher.eval()
    teacher.requires_grad_(False)

    framework_teacher = teacher
    if mode == "guided" and config.get("framework_teacher_adapter"):
        framework_base = AutoModelForCausalLM.from_pretrained(
            config["teacher_model"],
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).to(config["teacher_device"])
        framework_teacher = PeftModel.from_pretrained(
            framework_base,
            config["framework_teacher_adapter"],
            is_trainable=False,
        )
        framework_teacher.eval()
        framework_teacher.requires_grad_(False)

    # Keep Student/LoRA initialization identical across vanilla and guided runs.
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    student_base = AutoModelForCausalLM.from_pretrained(
        config["student_model"],
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    ).to(config["student_device"])
    student_base.config.use_cache = False
    if resume_checkpoint is None:
        student = get_peft_model(
            student_base,
            LoraConfig(
                r=config["lora_r"],
                lora_alpha=config["lora_alpha"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
        )
    else:
        student = PeftModel.from_pretrained(student_base, resume_checkpoint, is_trainable=True)
    student.config.use_cache = False
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"])
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        _move_optimizer_state(optimizer)

    random.shuffle(records)
    start_rollout_step = int(resume_state["rollout_step"]) if resume_state is not None else 0
    optimizer_step = int(resume_state["optimizer_step"]) if resume_state is not None else 0
    optimizer.zero_grad(set_to_none=True)
    if resume_state is not None:
        _restore_random_state(resume_state, config)

    if start_rollout_step > max_rollout_steps:
        raise ValueError(
            f"checkpoint is at rollout {start_rollout_step}, "
            f"beyond requested max_steps {max_rollout_steps}"
        )

    window_token_count = 0
    window_divergence_sum = 0.0
    window_entropy_sum = 0.0
    window_rollouts = 0
    last_checkpoint_step = start_rollout_step
    selected_records = records[start_rollout_step:max_rollout_steps]
    for step, record in enumerate(selected_records, start_rollout_step + 1):
        student.eval()
        rollout = generate_opd_rollout(
            framework_teacher,
            student,
            tokenizer,
            record["question"],
            mode=mode,
            framework_max_new_tokens=config["framework_max_new_tokens"],
            solution_max_new_tokens=config["solution_max_new_tokens"],
            temperature=config["generation_temperature"],
            framework_max_attempts=config["framework_max_attempts"],
        )
        student.train()
        student_ids = rollout.input_ids.to(config["student_device"])
        student_mask = rollout.attention_mask.to(config["student_device"])
        student_logits = student(input_ids=student_ids, attention_mask=student_mask).logits[:, :-1]

        with torch.no_grad():
            teacher_ids = rollout.input_ids.to(config["teacher_device"])
            teacher_mask = rollout.attention_mask.to(config["teacher_device"])
            teacher_logits = teacher(input_ids=teacher_ids, attention_mask=teacher_mask).logits[:, :-1]
            teacher_logits = teacher_logits.to(config["student_device"])

        completion_mask = causal_completion_mask(rollout.labels).to(config["student_device"])
        divergence_sum, loss_metrics = generalized_jsd_loss(
            student_logits,
            teacher_logits,
            completion_mask,
            beta=config["beta"],
            temperature=config["temperature"],
            reduction="sum",
        )
        divergence_sum.backward()
        sample_token_count = int(loss_metrics["num_tokens"].item())
        window_token_count += sample_token_count
        window_divergence_sum += float(loss_metrics["divergence_sum"].item())
        window_entropy_sum += float(loss_metrics["student_entropy_sum"].item())
        window_rollouts += 1

        optimizer_step_applied = (
            window_rollouts >= config["gradient_accumulation_steps"]
            or step == max_rollout_steps
        )
        optimizer_window_loss = None
        optimizer_window_entropy = None
        optimizer_window_tokens = None
        optimizer_window_rollouts = None
        if optimizer_step_applied:
            if window_token_count <= 0:
                raise RuntimeError("optimizer window contains no completion tokens")
            for parameter in student.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(window_token_count)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            optimizer_step += 1
            optimizer.zero_grad(set_to_none=True)
            optimizer_window_loss = window_divergence_sum / window_token_count
            optimizer_window_entropy = window_entropy_sum / window_token_count
            optimizer_window_tokens = window_token_count
            optimizer_window_rollouts = window_rollouts

        checkpoint_due = optimizer_step_applied and (
            step - last_checkpoint_step >= config["checkpoint_steps"]
            or step == max_rollout_steps
        )
        checkpoint_path = _available_checkpoint_dir(output_dir, step) if checkpoint_due else None

        metrics = {
            "run_id": run_id,
            "step": step,
            "rollout_step": step,
            "optimizer_step": optimizer_step,
            "optimizer_step_applied": optimizer_step_applied,
            "mode": mode,
            "loss": loss_metrics["loss"].item(),
            "divergence_sum": loss_metrics["divergence_sum"].item(),
            "num_tokens": sample_token_count,
            "optimizer_window_loss": optimizer_window_loss,
            "optimizer_window_mean_student_entropy": optimizer_window_entropy,
            "optimizer_window_num_tokens": optimizer_window_tokens,
            "optimizer_window_rollouts": optimizer_window_rollouts,
            "generated_token_count": len(rollout.generated_token_ids),
            "ended_with_eos": rollout.ended_with_eos,
            "hit_max_tokens": rollout.hit_max_tokens,
            **_framework_rollout_metrics(rollout),
            "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
            "framework": rollout.framework,
            "completion": rollout.completion,
        }
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            json.dumps(
                {key: value for key, value in metrics.items() if key != "completion"},
                ensure_ascii=False,
            )
        )
        if checkpoint_due:
            _save_checkpoint(
                student,
                optimizer,
                checkpoint_path,
                rollout_step=step,
                optimizer_step=optimizer_step,
                run_id=run_id,
                run_signature=run_signature,
                rng_devices={
                    "student": config["student_device"],
                    "teacher": config["teacher_device"],
                },
            )
            last_checkpoint_step = step
        if optimizer_step_applied:
            window_token_count = 0
            window_divergence_sum = 0.0
            window_entropy_sum = 0.0
            window_rollouts = 0
        del student_logits, teacher_logits, divergence_sum

    _publish_student_adapter(
        student,
        tokenizer,
        output_dir,
        allow_existing=resume_finalization,
    )
    _mark_run_complete(
        output_dir,
        run_id,
        rollout_step=max_rollout_steps,
        optimizer_step=optimizer_step,
    )


if __name__ == "__main__":
    main()
