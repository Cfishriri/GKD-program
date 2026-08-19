#!/usr/bin/env bash
set -euo pipefail

cd /root/blockdata/framework-guided-opd
export PYTHONPATH=$PWD/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/blockdata/kv_cache_env/bin/python train_opd.py --config configs/vanilla_opd.json
/root/blockdata/kv_cache_env/bin/python train_opd.py --config configs/guided_opd.json
