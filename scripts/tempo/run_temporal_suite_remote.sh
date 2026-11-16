#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_temporal_suite_remote.sh <release-root> <suite.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
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
exec "$PYTHON" "$ROOT/scripts/tempo/queue_temporal_suite.py" --root "$ROOT" --config "$CONFIG" --state "$3"
