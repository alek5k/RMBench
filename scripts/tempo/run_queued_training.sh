#!/usr/bin/env bash
# Pure native/GLA queue process until the predecessor completes and releases the GPU.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_queued_training.sh <release-root> <queue-config> <queue-state>' >&2; exit 2; }
ROOT=$(realpath "$1")
QUEUE=$(realpath "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=/data/RMBench-state
PYTHON="${HOME}/RMBench-state/envs/tempo-smolvla/bin/python"
export SMOVLA_ROOT="${STATE}/deps/smovla"
export PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab:${SMOVLA_ROOT}/src"
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export WANDB_DIR="${STATE}/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "${STATE}/credentials.env"
set +a
cd "$ROOT"
exec "$PYTHON" "$SCRIPT_DIR/queue_training_after_run.py" \
    --root "$ROOT" --config "$QUEUE" --state "$3"
