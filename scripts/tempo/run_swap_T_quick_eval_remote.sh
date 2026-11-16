#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_swap_T_quick_eval_remote.sh <release> <eval.yaml>' >&2; exit 2; }
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
exec 9>"$STATE/swap_T-30k-quick-eval.lock"
flock -n 9 || { echo 'Swap T 30k quick eval already active' >&2; exit 1; }
cd "$ROOT"
"$PYTHON" - "$CONFIG" "$ROOT" <<'PY'
import json,subprocess,sys,time,yaml
from pathlib import Path
recipe=yaml.safe_load(Path(sys.argv[1]).read_text());root=Path(sys.argv[2])
checkpoint=Path(recipe['launcher']['checkpoint'])
config=json.loads((checkpoint/'config.json').read_text())
metadata=json.loads((checkpoint/'tempo_run.json').read_text())
assert config['type']=='tempo_smolvla' and config['memory_backend']=='gla'
assert (config['chunk_size'],config['n_action_steps'])==(50,50)
assert checkpoint.name=='030000' and metadata['step']==30000
assert metadata['training']['wandb_run_id']==recipe['wandb']['training_run_id']=='b871c40f'
assert metadata['training']['task_name']==recipe['evaluation']['task']=='swap_T'
assert recipe['wandb']['group']==metadata['training']['wandb_group']
assert recipe['wandb']['enabled'] and recipe['wandb']['checkpoint_in_name']
assert (recipe['evaluation']['episodes'],recipe['evaluation']['num_workers'])==(10,1)
assert (checkpoint/'model.safetensors').stat().st_size>0
if Path(recipe['launcher']['output_root']).exists():raise FileExistsError(recipe['launcher']['output_root'])
print('Verified Swap T 30k, source b871c40f, 50/50, 10 episodes, one worker.',flush=True)
print('Waiting for validated simulator repair...',flush=True)
receipt=Path('/data/RMBench-state/swap_T-eval-runtime-ready.json')
deadline=time.monotonic()+7200
while not receipt.exists():
 if time.monotonic()>deadline:raise TimeoutError('Simulator repair exceeded two hours')
 time.sleep(5)
runtime=json.loads(receipt.read_text())
assert runtime['ready'] is True and Path(runtime['source_root'])==root
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
if free<16000:raise RuntimeError(f'Insufficient GPU headroom: {free} MiB free')
print('Simulator repair validated; starting quick eval without pausing training.',flush=True)
PY
exec "$PYTHON" "$STAGE/evaluate_smolvla.py" --root "$ROOT" --config "$CONFIG"
