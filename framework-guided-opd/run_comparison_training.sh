#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

python_bin=/root/blockdata/kv_cache_env/bin/python
teacher_output=outputs/teacher-framework-adapter-v3
if [[ ! -f "$teacher_output/adapter_model.safetensors" || ! -f "$teacher_output/RUN_COMPLETE" ]]; then
  echo "Missing completed v3 framework-teacher adapter. Run ./run_teacher_training.sh first." >&2
  exit 1
fi
"$python_bin" -c '
import pathlib, sys
from train_teacher import verify_teacher_artifact
root = pathlib.Path(sys.argv[1])
config = verify_teacher_artifact(root)["run_config"]
if pathlib.Path(config.get("data", "")).name != "gsm8k_frameworks_v3.jsonl":
    raise SystemExit("framework Teacher was not trained from v3 labels")
if config.get("num_records") != 1000:
    raise SystemExit("framework Teacher provenance does not contain exactly 1000 labels")
if config.get("purity_audit", {}).get("invalid") != 0:
    raise SystemExit("framework Teacher provenance failed its purity gate")
if config.get("generation_audit", {}).get("semantic_passes") != 1000:
    raise SystemExit("framework Teacher provenance failed its semantic gate")
' "$teacher_output"

verify_completed_arm() {
  local config_path=$1
  local output_path=$2
  "$python_bin" -c '
import hashlib, json, pathlib, sys
from framework_opd.evaluation import artifact_fingerprint
expected = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(sys.argv[2])
actual_path = root / "run_config.json"
actual = json.load(open(actual_path, encoding="utf-8"))
marker = json.load(open(root / "RUN_COMPLETE", encoding="utf-8"))
manifest = json.load(open(root / "run_manifest.json", encoding="utf-8"))
ignored = {"resume_from_checkpoint"}
mismatches = [
    key for key, value in expected.items()
    if key not in ignored and actual.get(key) != value
]
if mismatches:
    raise SystemExit("completed run config mismatch: " + ", ".join(sorted(mismatches)))
run_id = actual.get("run_id")
artifact_hash = artifact_fingerprint(root / "student_adapter")["sha256"]
config_hash = hashlib.sha256(actual_path.read_bytes()).hexdigest()
if not run_id or marker.get("run_id") != run_id or manifest.get("run_id") != run_id:
    raise SystemExit("completed run_id provenance is inconsistent")
if marker.get("status") != "complete" or manifest.get("status") != "complete" or manifest.get("role") != "opd_student":
    raise SystemExit("completed run status is inconsistent")
if marker.get("run_config_sha256") != config_hash or manifest.get("run_config_sha256") != config_hash:
    raise SystemExit("completed run_config hash is inconsistent")
if marker.get("adapter_artifact_sha256") != artifact_hash:
    raise SystemExit("completed Student adapter hash is inconsistent")
if manifest.get("adapter_artifact_sha256") != artifact_hash:
    raise SystemExit("completed manifest adapter hash is inconsistent")
' "$config_path" "$output_path"
}

run_arm() {
  local label=$1
  local config_path=$2
  local output_path=$3
  local adapter_path=$output_path/student_adapter/adapter_model.safetensors

  local resume_checkpoint
  resume_checkpoint=$("$python_bin" -c '
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("resume_from_checkpoint") or "")
' "$config_path")

  if [[ -f "$output_path/RUN_COMPLETE" && -f "$adapter_path" && -f "$output_path/run_config.json" && -f "$output_path/run_manifest.json" ]]; then
    verify_completed_arm "$config_path" "$output_path"
    echo "$label OPD v3 arm is already complete and provenance-matched; skipping."
    return
  fi

  if [[ -e "$output_path/RUN_COMPLETE" ]]; then
    echo "$label output has a completion marker but an incomplete final artifact set: $output_path" >&2
    exit 1
  fi

  if [[ -f "$output_path/metrics.jsonl" ]]; then
    if [[ -z "$resume_checkpoint" ]]; then
      echo "$label OPD has an interrupted run. Set resume_from_checkpoint in $config_path." >&2
      exit 1
    fi
    echo "$label OPD will resume from $resume_checkpoint."
  elif [[ -e "$output_path" ]]; then
    echo "$label output directory exists without resumable metrics: $output_path" >&2
    exit 1
  fi

  "$python_bin" train_opd.py --config "$config_path"
}

run_arm "Vanilla" configs/vanilla_opd_v3.json outputs/vanilla-opd-v3
run_arm "Guided" configs/guided_opd_v3.json outputs/guided-opd-v3
