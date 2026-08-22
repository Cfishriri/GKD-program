#!/usr/bin/env bash
set -euo pipefail

cd /root/blockdata/framework-guided-opd
export PYTHONPATH=$PWD/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

for required_file in \
  outputs/teacher-framework-adapter-v2/adapter_model.safetensors \
  outputs/teacher-framework-adapter-v2/RUN_COMPLETE \
  outputs/vanilla-opd-v2/student_adapter/adapter_model.safetensors \
  outputs/vanilla-opd-v2/RUN_COMPLETE \
  outputs/guided-opd-v2/student_adapter/adapter_model.safetensors \
  outputs/guided-opd-v2/RUN_COMPLETE
do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required adapter: $required_file" >&2
    exit 1
  fi
done

/root/blockdata/kv_cache_env/bin/python evaluate_comparison.py --config configs/evaluation.json
