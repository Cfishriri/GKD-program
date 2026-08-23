#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR/src"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

data_path=data/gsm8k_frameworks_v2.jsonl
generation_audit_path=data/gsm8k_frameworks_v2.generation-audit.json
purity_audit_path=data/gsm8k_frameworks_v2.audit.json
partial_path=${data_path}.partial

if [[ -e "$data_path" || -e "$partial_path" || -e "$generation_audit_path" || -e "$purity_audit_path" ]]; then
  echo "Refusing to overwrite existing v2 framework data or audit files." >&2
  echo "Archive the existing v2 files before starting a new generation run." >&2
  exit 1
fi

/root/blockdata/kv_cache_env/bin/python prepare_framework_data.py \
  --model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --dataset /root/eb-public/huggingface-datasets/openai/gsm8k/main/train-00000-of-00001.parquet \
  --output "$data_path" \
  --audit-output "$generation_audit_path" \
  --device cuda:1 \
  --limit 1000 \
  --max-attempts 5 \
  --seed 42 \
  --temperature 0.4 \
  --progress-every 5

/root/blockdata/kv_cache_env/bin/python audit_framework_data.py \
  --data "$data_path" \
  --output "$purity_audit_path"

echo "Framework labels are ready: $data_path"
echo "Purity gate passed: $purity_audit_path"
