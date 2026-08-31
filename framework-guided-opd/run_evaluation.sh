#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for required_file in \
  outputs/teacher-framework-adapter-v3/adapter_model.safetensors \
  outputs/teacher-framework-adapter-v3/RUN_COMPLETE \
  outputs/vanilla-opd-v3/student_adapter/adapter_model.safetensors \
  outputs/vanilla-opd-v3/RUN_COMPLETE \
  outputs/guided-opd-v3/student_adapter/adapter_model.safetensors \
  outputs/guided-opd-v3/RUN_COMPLETE
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required adapter: $required_file" >&2
    exit 1
  fi
done

/root/blockdata/kv_cache_env/bin/python evaluate_comparison.py --config configs/evaluation.json
