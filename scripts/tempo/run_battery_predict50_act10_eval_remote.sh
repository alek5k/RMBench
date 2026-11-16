#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_battery_predict50_act10_eval_remote.sh <training-release> <eval.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab"
export SMOVLA_ROOT="$STATE/deps/smovla" HF_HOME="$STATE/huggingface"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" - "$CONFIG" <<'PY'
import json,sys,yaml
from pathlib import Path
recipe=yaml.safe_load(Path(sys.argv[1]).read_text())
checkpoint=Path(recipe['launcher']['checkpoint'])
config=json.loads((checkpoint/'config.json').read_text())
metadata=json.loads((checkpoint/'tempo_run.json').read_text())
assert config['type']=='tempo_smolvla' and config['memory_backend']=='gla'
assert (config['chunk_size'],config['n_action_steps'])==(50,10)
assert metadata['step']==100000 and metadata['training']['wandb_run_id']=='06075719'
assert recipe['wandb']['training_run_id']=='06075719'
assert recipe['evaluation']['episodes']==100
assert (checkpoint/'model.safetensors').stat().st_size>0
print('Verified final 100k checkpoint: source 06075719, predict 50, execute 10.',flush=True)
PY
exec "$PYTHON" "$ROOT/scripts/tempo/evaluate_smolvla.py" --root "$ROOT" --config "$CONFIG"
