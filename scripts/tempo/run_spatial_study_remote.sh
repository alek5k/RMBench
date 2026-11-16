#!/usr/bin/env bash
# Reuse the already prepared/pinned WaitAtGoal environments without modifying them.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_spatial_study_remote.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" WANDB_DIR="$STATE/wandb"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" - <<'PY'
import importlib.metadata as m
import json
from pathlib import Path
for name, expected in {'torch':'2.10.0','transformers':'4.57.1','datasets':'4.1.1','numpy':'1.26.4'}.items():
    assert m.version(name).split('+')[0] == expected, (name, m.version(name))
info=json.loads(Path('/data/RMBench-state/datasets/waitatgoal_v4/meta/info.json').read_text())
assert (info['total_episodes'],info['total_frames'],info['fps'])==(203,14008,10)
assert info['features']['action']['shape']==[2]
assert Path('/data/RMBench-state/envs/tempo-temporal-eval').is_dir()
print('Pinned runtime and full WaitAtGoal dataset verified; existing GPU jobs untouched',flush=True)
PY
exec "$PYTHON" "$ROOT/scripts/tempo/queue_reconstruction_baselines.py" --root "$ROOT" --config "$CONFIG" \
    --state "$STATE/waitatgoal-spatial-reconstruction-261110.json"
