#!/usr/bin/env bash
# Final Cover Blocks evaluation on the existing, validated simulator environment.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_cover_blocks_final_eval_remote.sh <release-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STAGE=$(dirname "$CONFIG")
STATE=/data/RMBench-state
PYTHON="$HOME/RMBench-state/envs/tempo-smolvla/bin/python"
export PYTHONPATH="$STAGE:$ROOT/scripts/tempo:$ROOT:$ROOT/XPolicyLab"
export SMOVLA_ROOT="$STATE/deps/smovla" HF_HOME="$STATE/huggingface"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" - "$ROOT" "$CONFIG" <<'PY'
from datetime import datetime,timezone
import json,os,subprocess,sys
from pathlib import Path
import yaml
from queue_final_rmbench_eval import validate_evaluation
from queue_training_after_run import verify_checkpoint,predecessor_ready
from evaluate_smolvla import logger_command
root,config=map(Path,sys.argv[1:]);queue=yaml.safe_load(config.read_text())
source=json.loads(Path(queue['predecessor_state']).read_text())
assert predecessor_ready(source,queue['training_run_id'])
assert Path(source['source_root'])==root
training=yaml.safe_load(Path(queue['training_config']).read_text())['training']
recipe=yaml.safe_load((config.parent/queue['eval_config']).read_text())
validate_evaluation(training,recipe)
checkpoint=verify_checkpoint(training,100000)
policy=json.loads((checkpoint/'config.json').read_text())
assert (policy['memory_backend'],policy['chunk_size'],policy['n_action_steps'])==('gla',50,20)
assert (checkpoint/'model.safetensors').stat().st_size>1_000_000_000
launch=recipe['launcher']
code='''import importlib.metadata as m,json
import torch,numpy,sapien,mplib,warp,curobo
expected={'torch':'2.4.1+cu121','numpy':'1.26.4','sapien':'3.0.0b1','mplib':'0.2.1','warp-lang':'1.12.0'}
versions={name:m.version(name) for name in expected}
assert versions==expected,versions
print('Validated simulator versions:',json.dumps(versions))'''
subprocess.run([launch['simulator_python'],'-c',code],env={**os.environ,'CUDA_VISIBLE_DEVICES':''},check=True)
subprocess.run([launch['simulator_python'],'-c',
    "import torch,envs.cover_blocks; assert torch.ones(4,device='cuda').sum().item()==4; print('Cover Blocks CUDA import passed')"],check=True)
env={**os.environ,'WANDB_ENTITY':recipe['wandb']['entity'],'WANDB_PROJECT':recipe['wandb']['project']}
subprocess.run(logger_command(root,launch,recipe['evaluation'],recipe['wandb'],checkpoint,config.parent)+['--check-config'],env=env,check=True)
receipt=Path(queue['runtime_receipt'])
if receipt.exists():raise FileExistsError(receipt)
value={'ready':True,'source_root':str(root),'task':'cover_blocks','checked_at':datetime.now(timezone.utc).isoformat(),
       'simulator_versions_checked':True,'cuda_task_import_checked':True,'wandb_preflight_checked':True,
       'reference_quick_checkpoints':[30000,70000]}
receipt.write_text(json.dumps(value,indent=2)+'\n')
print('Final 100k checkpoint and evaluation runtime verified.',flush=True)
PY
exec "$PYTHON" "$STAGE/queue_final_rmbench_eval.py" \
    --root "$ROOT" --config "$CONFIG" --state "$STATE/tempo-cover-blocks-final-eval-9b4a762e.json"
