#!/usr/bin/env bash
set -uo pipefail
PROJECT=/root/blockdata/framework-guided-opd/framework-guided-opd
PYTHON=/root/blockdata/kv_cache_env/bin/python
BUNDLE="$PROJECT/outputs/closeout-human36-test50-20260925-v1"
KIND="${1:-full}"
if [[ "$KIND" != full && "$KIND" != smoke && "$KIND" != capacity ]]; then
  echo 'Usage: bash run_closeout.sh [full|smoke|capacity]' >&2
  exit 2
fi
cd "$PROJECT" || exit 2
test -f "$BUNDLE/manifest.json" || { echo "Missing frozen data: $BUNDLE" >&2; exit 2; }
# Do not compete with an already occupied device. Recheck immediately before start.
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) || exit 2
while read -r memory; do
  if (( memory > 1000 )); then
    echo 'GPU is occupied; no workers started. Check nvidia-smi.' >&2
    exit 2
  fi
done <<< "$GPU_MEMORY"
RUN_DIR=$(mktemp -d "$PROJECT/outputs/closeout-${KIND}-XXXXXXXX") || exit 2
echo "RUN_DIR=$RUN_DIR"
echo 'GPU0=vanilla; GPU1=guided; each worker loads local 4B teacher and 1.7B student.'
EXTRA=()
LIMIT=10800
if [[ "$KIND" == smoke ]]; then
  EXTRA=(--smoke --train-tokens 128 --eval-tokens 128 --batch-size 2)
  LIMIT=1200
fi
if [[ "$KIND" == capacity ]]; then
  EXTRA=(--smoke)
  LIMIT=1200
fi
timeout --signal=TERM --kill-after=30s "$LIMIT" env PYTHONPATH="$PROJECT/src" PYTHONUNBUFFERED=1 \
  "$PYTHON" -B closeout_experiment.py worker --bundle "$BUNDLE" --mode vanilla --device cuda:0 \
  --output "$RUN_DIR/vanilla" "${EXTRA[@]}" > "$RUN_DIR/vanilla.log" 2>&1 &
VANILLA_PID=$!
timeout --signal=TERM --kill-after=30s "$LIMIT" env PYTHONPATH="$PROJECT/src" PYTHONUNBUFFERED=1 \
  "$PYTHON" -B closeout_experiment.py worker --bundle "$BUNDLE" --mode guided --device cuda:1 \
  --output "$RUN_DIR/guided" "${EXTRA[@]}" > "$RUN_DIR/guided.log" 2>&1 &
GUIDED_PID=$!
echo "vanilla_pid=$VANILLA_PID guided_pid=$GUIDED_PID timeout_seconds=$LIMIT"
wait "$VANILLA_PID"
VANILLA_EXIT=$?
wait "$GUIDED_PID"
GUIDED_EXIT=$?
printf '{"vanilla":%d,"guided":%d}\n' "$VANILLA_EXIT" "$GUIDED_EXIT" > "$RUN_DIR/worker_exit_codes.json"
if (( VANILLA_EXIT != 0 || GUIDED_EXIT != 0 )); then
  echo "Worker failed: vanilla=$VANILLA_EXIT guided=$GUIDED_EXIT. Inspect $RUN_DIR/*.log; no automatic retry."
  exit 1
fi
env PYTHONPATH="$PROJECT/src" "$PYTHON" -B closeout_experiment.py report --bundle "$BUNDLE" \
  --workers "$RUN_DIR/vanilla" "$RUN_DIR/guided" --output "$RUN_DIR/report" > "$RUN_DIR/report.log" 2>&1
REPORT_EXIT=$?
printf '{"report":%d}\n' "$REPORT_EXIT" > "$RUN_DIR/report_exit_code.json"
echo "Finished: report_exit=$REPORT_EXIT; results=$RUN_DIR/report; logs=$RUN_DIR"
exit "$REPORT_EXIT"
