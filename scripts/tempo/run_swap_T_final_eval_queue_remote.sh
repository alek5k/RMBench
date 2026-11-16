#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_swap_T_final_eval_queue_remote.sh <release> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STAGE=$(dirname "$CONFIG")
STATE=/data/RMBench-state
export PYTHONPATH="$STAGE:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab"
export SMOVLA_ROOT="$STATE/deps/smovla" HF_HOME="$STATE/huggingface"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$STAGE/queue_final_rmbench_eval.py" \
  --root "$ROOT" --config "$CONFIG" --state "$STATE/tempo-swap-T-final-eval-b871c40f.json"
