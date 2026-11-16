#!/usr/bin/env bash
# Wait for one training screen to finish successfully, then run RMBench evals
# serially.  A single L40S cannot safely host two simulator evaluators.
set -Eeuo pipefail

[[ $# -ge 4 ]] || {
  echo "Usage: $0 <RMBench-root> <training-screen-name> <final-checkpoint> <eval-config> [...]" >&2
  exit 2
}

RMBENCH_ROOT="$(cd "$1" && pwd)"
TRAINING_SCREEN=$2
FINAL_CHECKPOINT=$3
shift 3
EVAL_CONFIGS=("$@")
STATE_ROOT="${RMBENCH_STATE_ROOT:-${HOME}/RMBench-state}"
LOG_DIR="${STATE_ROOT}/logs"
mkdir -p "${LOG_DIR}"

screen_exists() {
  screen -ls 2>/dev/null | grep -Fq ".${1}"
}

wait_for_evaluation() {
  while screen -ls 2>/dev/null | grep -Eq '[0-9]+\.(smolvla|lte)-eval-'; do
    sleep 30
  done
}

echo "Waiting for training screen: ${TRAINING_SCREEN}"
while screen_exists "${TRAINING_SCREEN}"; do
  sleep 60
done

[[ -f "${FINAL_CHECKPOINT}/model.safetensors" ]] || {
  echo "Training finished without the expected final checkpoint: ${FINAL_CHECKPOINT}" >&2
  exit 1
}

for eval_config in "${EVAL_CONFIGS[@]}"; do
  [[ -f "${eval_config}" ]] || { echo "Evaluation config is missing: ${eval_config}" >&2; exit 1; }
  wait_for_evaluation
  echo "Launching evaluation: ${eval_config}"
  bash "${RMBENCH_ROOT}/scripts/lte/run_eval.sh" "${RMBENCH_ROOT}" "${eval_config}"
  wait_for_evaluation
done

echo "All post-training RMBench evaluations completed successfully."
