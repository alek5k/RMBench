#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_cover_blocks_suite_remote.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" WANDB_DIR="$STATE/wandb"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$ROOT/scripts/tempo/queue_cover_blocks_suite.py" \
    --root "$ROOT" --config "$CONFIG" --state "$STATE/cover-blocks-predict50-act50-suite-261109.json"
