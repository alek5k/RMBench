#!/usr/bin/env bash
# Evaluate checkpoints sequentially on one GPU, with one W&B run per checkpoint.
set -Eeuo pipefail

[[ $# -ge 3 && $# -le 4 ]] || {
  echo "Usage: $0 <RMBench-root> <smolvla|lte> <checkpoint-run-root> [episodes]" >&2
  exit 2
}
RMBENCH_ROOT="$(cd "$1" && pwd)"
MODE=$2
CHECKPOINT_ROOT="$(cd "$3" && pwd)"
EPISODES=${4:-100}
WAIT_FOR_CHECKPOINTS=${RMBENCH_WAIT_FOR_CHECKPOINTS:-false}
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "episodes must be a positive integer" >&2; exit 2; }
case "${WAIT_FOR_CHECKPOINTS}" in true|false) ;; *) echo "RMBENCH_WAIT_FOR_CHECKPOINTS must be true or false" >&2; exit 2 ;; esac

wait_for_evaluators() {
  # One L40S should host only one RMBench simulator evaluator at a time,
  # regardless of policy family. The accompanying W&B sidecar shares this
  # name prefix and exits with its evaluator.
  while screen -ls 2>/dev/null | grep -Eq '[0-9]+\.(smolvla|lte)-eval-battery_try-'; do
    sleep 30
  done
}

# Do not contend with an existing evaluator. Each invocation of run_eval.sh
# creates an evaluator and a W&B sidecar; wait for both to exit before the next.
wait_for_evaluators

for checkpoint_step in 20000 40000 60000 80000 100000; do
  # LeRobot stores numeric checkpoint directories with six-digit zero padding.
  printf -v checkpoint_dir '%06d' "${checkpoint_step}"
  checkpoint="${CHECKPOINT_ROOT}/checkpoints/${checkpoint_dir}/pretrained_model"
  while [[ ! -d "${checkpoint}" ]]; do
    [[ "${WAIT_FOR_CHECKPOINTS}" == true ]] || {
      echo "Checkpoint is missing: ${checkpoint}" >&2
      exit 1
    }
    echo "Waiting for checkpoint ${checkpoint_step}: ${checkpoint}"
    sleep 60
  done
  echo "Launching ${MODE} evaluation for checkpoint ${checkpoint_step} (${EPISODES} episodes)."
  bash "${RMBENCH_ROOT}/scripts/lte/run_eval.sh" "${RMBENCH_ROOT}" "${MODE}" "${checkpoint}" "${EPISODES}"
  wait_for_evaluators
done
