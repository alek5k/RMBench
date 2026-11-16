#!/usr/bin/env bash
# Wait for the exact predecessor suite; use the existing pinned training/eval envs.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_liftqa_horizon_suite_remote.sh <release> <suite.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$SCRIPT_DIR:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$SCRIPT_DIR/queue_liftqa_seed_repeats.py" \
    --root "$ROOT" --config "$2" --state "$3"
