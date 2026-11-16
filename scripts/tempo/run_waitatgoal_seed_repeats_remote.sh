#!/usr/bin/env bash
# Queue only; no environment installation or mutation of the active runs.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: run_waitatgoal_seed_repeats_remote.sh <release-root> <queue.yaml> <state.json>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$SCRIPT_DIR:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$SCRIPT_DIR/queue_waitatgoal_seed_repeats.py" \
    --root "$ROOT" --config "$CONFIG" --state "$3"
