#!/usr/bin/env bash
set -euo pipefail

cd /root/blockdata/framework-guided-opd
export PYTHONPATH=$PWD/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/blockdata/kv_cache_env/bin/python evaluate_comparison.py --config configs/evaluation.json
