#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_battery_predict50_act10_remote.sh <release> <training.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Do not mix a system cuBLASLt with the policy environment's packaged CUDA libraries.
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" "$ROOT/scripts/tempo/run_battery_predict50_act10.py" \
  --root "$ROOT" --config "$CONFIG" \
  --reference "$STATE/checkpoints/tempo_smolvla_battery_try_gla_reference_100k/resolved_config.json" \
  --processors "$STATE/checkpoints/tempo_smolvla_battery_try_gla_reference_100k/100000" \
  --state "$STATE/tempo-training-06075719.json"
