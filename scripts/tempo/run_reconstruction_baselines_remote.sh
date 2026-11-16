#!/usr/bin/env bash
# Reuse the pinned policy runtime; data/simulator preparation is isolated.
set -Eeuo pipefail
[[ $# -ge 2 && $# -le 3 ]] || { echo 'Usage: run_reconstruction_baselines_remote.sh <source-root> <queue.yaml> [queue-state.json]' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" WANDB_DIR="$STATE/wandb"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" - <<'PY'
import importlib.metadata as m
import torch
for name, expected in {'torch':'2.10.0','transformers':'4.57.1','datasets':'4.1.1','numpy':'1.26.4'}.items():
    assert m.version(name).split('+')[0] == expected, (name, m.version(name))
assert torch.cuda.is_available()
print('Pinned policy environment and GPU verified', flush=True)
PY
# Zarr conversion tools must not change the shared policy environment.
TOOLS="$STATE/envs/temporal-dataset-tools"
if [[ ! -d "$TOOLS/zarr-2.18.7.dist-info" ]]; then
    "$HOME/.local/bin/uv" pip install --python "$PYTHON" --target "$TOOLS" 'numpy==1.26.4' 'zarr==2.18.7'
fi
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$TOOLS:$PYTHONPATH" "$PYTHON" scripts/lte/prepare_temporal_datasets.py \
    --output-root "$STATE/datasets" --datasets waitatgoal_v4
"$PYTHON" - <<'PY'
import json
from pathlib import Path
info=json.loads(Path('/data/RMBench-state/datasets/waitatgoal_v4/meta/info.json').read_text())
assert (info['total_episodes'],info['total_frames'],info['fps'])==(203,14008,10)
assert info['features']['action']['shape']==[2]
print('Full WaitAtGoal dataset verified: 203 episodes / 14008 frames',flush=True)
PY
bash "$ROOT/scripts/tempo/setup_temporal_eval_remote.sh" "$ROOT"
exec "$PYTHON" "$ROOT/scripts/tempo/queue_reconstruction_baselines.py" --root "$ROOT" --config "$CONFIG" \
    --state "${3:-$STATE/waitatgoal-reconstruction-baselines-261110.json}"
