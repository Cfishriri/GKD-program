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
  echo "Missing framework-teacher adapter. Run ./run_teacher_training.sh first." >&2
  exit 1
fi
"$python_bin" -c 'from train_teacher import verify_teacher_artifact; import sys; verify_teacher_artifact(sys.argv[1])' "$teacher_output"

for smoke_spec in \
  "configs/vanilla_smoke_v3.json:outputs/vanilla-smoke-v3" \
  "configs/guided_smoke_v3.json:outputs/guided-smoke-v3"
do
  smoke_config=${smoke_spec%%:*}
  smoke_output=${smoke_spec#*:}
  if [[ -f "$smoke_output/RUN_COMPLETE" ]]; then
    echo "Smoke run already completed at $smoke_output; skipping."
  elif [[ -e "$smoke_output" ]]; then
    echo "Incomplete smoke output exists at $smoke_output; archive it before retrying." >&2
    exit 1
  else
    "$python_bin" train_opd.py --config "$smoke_config"
  fi
done
