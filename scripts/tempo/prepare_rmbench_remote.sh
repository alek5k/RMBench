#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: prepare_rmbench_remote.sh <release-root> <preparation.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export SMOVLA_ROOT="${STATE}/deps/smovla" PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab:${STATE}/deps/smovla/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "${STATE}/credentials.env"
set +a
cd "$ROOT"
exec "${HOME}/RMBench-state/envs/tempo-smolvla/bin/python" scripts/tempo/prepare_rmbench_dataset.py --root "$ROOT" --config "$CONFIG"
