#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_cover_blocks_quick_eval_remote.sh <training-release> <eval.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STAGE=$(dirname "$CONFIG")
STATE=/data/RMBench-state
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab"
export SMOVLA_ROOT="$STATE/deps/smovla" HF_HOME="$STATE/huggingface"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" - "$CONFIG" <<'PY'
import json,subprocess,sys,yaml
from pathlib import Path
recipe=yaml.safe_load(Path(sys.argv[1]).read_text())
checkpoint=Path(recipe['launcher']['checkpoint'])
config=json.loads((checkpoint/'config.json').read_text())
metadata=json.loads((checkpoint/'tempo_run.json').read_text())
assert config['type']=='tempo_smolvla' and config['memory_backend']=='gla'
assert (config['chunk_size'],config['n_action_steps'])==(50,20)
step=int(checkpoint.name)
assert step in (30000,70000) and metadata['step']==step
assert metadata['training']['wandb_run_id']=='9b4a762e'
assert metadata['training']['task_name']==recipe['evaluation']['task']=='cover_blocks'
assert recipe['wandb']['training_run_id']=='9b4a762e'
assert recipe['wandb']['group']==metadata['training']['wandb_group']
assert recipe['wandb']['enabled'] and recipe['wandb']['checkpoint_in_name']
assert (recipe['evaluation']['episodes'],recipe['evaluation']['num_workers'])==(10,1)
assert (checkpoint/'model.safetensors').stat().st_size>0
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<16000:raise RuntimeError(f'Insufficient GPU headroom: {free} MiB free')
print(f'Verified source 9b4a762e, Cover Blocks checkpoint {step//1000}k, 50/20, 10 episodes, one worker.',flush=True)
PY
exec "$PYTHON" "$STAGE/evaluate_smolvla.py" --root "$ROOT" --config "$CONFIG"
