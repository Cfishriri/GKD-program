#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=/root/blockdata/framework-guided-opd/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/blockdata/kv_cache_env/bin/python /root/blockdata/framework-guided-opd/train_opd.py \
  --config /root/blockdata/framework-guided-opd/configs/smoke.json
