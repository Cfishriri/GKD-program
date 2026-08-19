#!/usr/bin/env bash
set -euo pipefail

cd /root/blockdata/framework-guided-opd
export PYTHONPATH=$PWD/src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

/root/blockdata/kv_cache_env/bin/python train_teacher.py \
  --model /root/eb-public/huggingface-models/Qwen/Qwen3-4B \
  --data data/gsm8k_frameworks.jsonl \
  --output outputs/teacher-framework-adapter \
  --device cuda:1 \
  --steps 1000 \
  --learning-rate 0.0001
