#!/usr/bin/env bash
# Run a standalone evaluation without changing a training queue or its release.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_checkpoint_eval_remote.sh <source-root> <eval.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" WANDB_DIR="$STATE/wandb"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
exec /home/ubuntu/RMBench-state/envs/tempo-smolvla/bin/python \
    "$ROOT/scripts/tempo/evaluate_smolvla.py" --root "$ROOT" --config "$CONFIG"
