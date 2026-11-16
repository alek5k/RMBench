#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: setup_rmbench_task_remote.sh <release> <training.yaml> <preparation.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
PREPARATION=$(realpath "$3")
STAGE=$(dirname "$CONFIG")
STATE=/data/RMBench-state
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$STATE/uv-cache" UV_PYTHON_INSTALL_DIR="$STATE/uv-python" UV_LINK_MODE=copy
export SMOVLA_ROOT="$STATE/deps/smovla" HF_HOME="$STATE/huggingface"
export HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
mkdir -p "$STATE/deps" "$STATE/datasets" "$STATE/checkpoints" "$STATE/embedding_cache" "$STATE/wandb"
set -a
source "$STATE/credentials.env"
set +a
bash "$ROOT/scripts/tempo/install_smolvla_environment.sh"
export PYTHONPATH="$STAGE:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
RUN_ID=$("$PYTHON" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["training"]["wandb_run_id"])' "$CONFIG")
[[ "$RUN_ID" =~ ^[a-z0-9]+$ ]]
exec "$PYTHON" "$STAGE/run_rmbench_task.py" --root "$ROOT" --config "$CONFIG" \
  --preparation-config "$PREPARATION" --reference "$STAGE/reference_resolved_config.json" \
  --processors "$STAGE/reference_processors" --state "$STATE/tempo-training-$RUN_ID.json"
