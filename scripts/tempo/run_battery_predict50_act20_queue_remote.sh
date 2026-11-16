#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_battery_predict50_act20_queue_remote.sh <source-release> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STAGE=$(dirname "$CONFIG")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$STAGE:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$STAGE/queue_battery_predict50_act20_after_eval.py" \
  --root "$ROOT" --config "$CONFIG" --state "$STATE/tempo-predict50-act20-train-eval-455049d4.json"
