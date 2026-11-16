#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: run_cover_blocks_predict50_act20_remote.sh <release> <training.yaml> [preparation-pid]' >&2; exit 2; }
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
EXTRA=()
if [[ $# -eq 3 ]]; then EXTRA+=(--preparation-pid "$3"); fi
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$STAGE/run_cover_blocks_predict50_act20.py" \
  --root "$ROOT" --config "$CONFIG" \
  --reference "$STATE/checkpoints/battery_try_tempo_predict50_act20_100k_seed0/resolved_config.json" \
  --receipt "$STATE/datasets/cover_blocks_prepared.json" \
  --state "$STATE/tempo-training-9b4a762e.json" "${EXTRA[@]}"
