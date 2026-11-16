#!/usr/bin/env bash
# New orchestration only; model/trainer code stays pinned to the original release.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_liftqa_baseline_suite_remote.sh <source-root> <suite.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$SCRIPT_DIR/queue_liftqa_baseline_and_evals.py" \
    --root "$ROOT" --config "$CONFIG" --state "$3"
