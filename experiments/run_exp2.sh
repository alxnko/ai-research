#!/usr/bin/env bash
# EXP-2: paired Qwen verification-policy study. See EXP-2_PROTOCOL.md.
#
# Runs 18 control attempts, then 18 treatment attempts, in the registered
# seed and task order. Strictly sequential: one model request at a time, so
# VRAM and decode measurements are not polluted by a second resident model.
set -uo pipefail
cd "$(dirname "$0")/.."

TASKS=(
  backend-auth-token
  backend-idempotency
  debugging-off-by-one
  debugging-money-round
  refactor-parser
  refactor-validators
)
MODEL=qwen3.5:4b-q4_K_M
LOG_DIR=runs/exp2/logs
mkdir -p "$LOG_DIR"
SUMMARY=runs/exp2/attempts.log

run_one() {
  local arm=$1 seed=$2 task=$3
  local dir="runs/exp2/$arm/seed$seed/$task"
  local log="$LOG_DIR/$arm-seed$seed-$task.log"
  if [[ -f "$dir/results.json" ]]; then
    echo "SKIP $arm seed=$seed $task (results.json exists)"
    return 0
  fi
  local t0=$SECONDS
  uv run python -m experiments.exp2_entry \
    --exp2-policy "$arm" \
    --run-dir "$dir" \
    --model "$MODEL" \
    --actuator native_tools \
    --task "$task" \
    --ctx 16384 \
    --ngpu 99 \
    --steps 20 \
    --max-wall 600 \
    --strict-isolation \
    --sampling-profile exp2-coding-v1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --repeat-penalty 1.0 \
    --seed "$seed" >"$log" 2>&1
  local rc=$?
  printf '%s arm=%s seed=%s task=%-22s exit=%d dur=%ss\n' \
    "$(date -Is)" "$arm" "$seed" "$task" "$rc" "$((SECONDS - t0))" | tee -a "$SUMMARY"
}

for arm in control verification; do
  for seed in 42 43 44; do
    for task in "${TASKS[@]}"; do
      run_one "$arm" "$seed" "$task"
    done
  done
done

echo "EXP-2 complete. Per-attempt logs: $LOG_DIR"
