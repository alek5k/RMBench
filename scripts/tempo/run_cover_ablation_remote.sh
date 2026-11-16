#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_cover_ablation_remote.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" WANDB_DIR="$STATE/wandb"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
ENV_ROOT="${TEMPO_SMOLVLA_ENV:-$HOME/RMBench-state/envs/tempo-smolvla}"
[[ -x "$ENV_ROOT/bin/python" ]] || ENV_ROOT="$STATE/envs/tempo-smolvla"
exec "$ENV_ROOT/bin/python" "$ROOT/scripts/tempo/run_cover_ablation_brev.py" \
  --root "$ROOT" --config "$CONFIG"
