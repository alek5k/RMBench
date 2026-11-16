#!/usr/bin/env bash
# No package changes to the shared policy environment; conversion extras are isolated.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_temporal_objective_trial_remote.sh <release-root> <training-config>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
PYTHON="${HOME}/RMBench-state/envs/tempo-smolvla/bin/python"
export SMOVLA_ROOT="${STATE}/deps/smovla"
export PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab:${SMOVLA_ROOT}/src"
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export WANDB_DIR="${STATE}/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH
set -a
source "${STATE}/credentials.env"
set +a
TOOLS="${STATE}/envs/temporal-dataset-tools"
if ! PYTHONPATH="${TOOLS}:${PYTHONPATH}" "$PYTHON" -c 'import zarr; assert zarr.__version__ == "2.18.7"' 2>/dev/null; then
    "${HOME}/.local/bin/uv" pip install --target "$TOOLS" 'zarr==2.18.7' --override "$ROOT/scripts/tempo/overrides.txt"
fi
cd "$ROOT"
exec "$PYTHON" "$ROOT/scripts/tempo/run_temporal_objective_trial.py" \
    --root "$ROOT" --config "$CONFIG" \
    --reference "$ROOT/config/tempo/references/waitatgoal_5547de5e.json" \
    --defaults "$ROOT/config/tempo/waitatgoal_defaults.yaml" \
    --state "$STATE/tempo-training-b6a941ce.json"
