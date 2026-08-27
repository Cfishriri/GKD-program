#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

framework_data=data/gsm8k_frameworks_v3.jsonl
generation_audit=data/gsm8k_frameworks_v3.generation-audit.json
teacher_output=outputs/teacher-framework-adapter-v3

if [[ ! -f "$framework_data" || ! -f "$generation_audit" ]]; then
  echo "Missing $framework_data. Run ./run_framework_data.sh first." >&2
  exit 1
fi
if [[ -e "$teacher_output" ]]; then
  echo "Refusing to write into the existing teacher output at $teacher_output." >&2
  echo "Archive it before starting a new Teacher run." >&2
  exit 1
fi

/root/blockdata/kv_cache_env/bin/python -c '
import hashlib, json, sys
audit = json.load(open(sys.argv[1], encoding="utf-8"))
if audit.get("status") != "complete":
    raise SystemExit("framework generation audit is not complete")
if audit.get("requested_valid") != 1000 or audit.get("valid") != 1000:
    raise SystemExit("framework generation did not publish exactly 1000 valid labels")
if audit.get("schema_version") != 3 or audit.get("semantic_passes") != 1000:
    raise SystemExit("framework generation did not semantically verify every label")
digest = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
if audit.get("output_sha256") != digest:
    raise SystemExit("framework data hash does not match its generation audit")
' "$generation_audit" "$framework_data"

/root/blockdata/kv_cache_env/bin/python audit_framework_data.py \
  --data "$framework_data" \
  --output data/gsm8k_frameworks_v3.audit.json

/root/blockdata/kv_cache_env/bin/python train_teacher.py \
  --model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --data "$framework_data" \
  --generation-audit "$generation_audit" \
  --output "$teacher_output" \
  --device cuda:1 \
  --steps 1000 \
  --learning-rate 0.0001 \
  --seed 42 \
  --expected-records 1000
