#!/usr/bin/env bash
# Isolated simulator overlay; shared policy dependencies are read-only.
set -Eeuo pipefail
[[ $# -ge 2 ]] || {
    echo 'Usage: run_temporal_eval_remote.sh <RMBench release> <eval.yaml> [--smoke] [--smoke-checkpoint PATH]' >&2
    exit 2
}
RMBENCH_ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE_ROOT=/data/RMBench-state
export SMOVLA_ROOT="${STATE_ROOT}/deps/smovla"
export HF_HOME="${STATE_ROOT}/huggingface" WANDB_DIR="${STATE_ROOT}/wandb"
export SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NUMBA_DISABLE_JIT=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
set -a
source "${STATE_ROOT}/credentials.env"
set +a
cd "${RMBENCH_ROOT}"
# Only this bootstrap process uses the training interpreter; rollout execs the
# isolated simulator interpreter. No install operation touches the live env.
exec "${HOME}/RMBench-state/envs/tempo-smolvla/bin/python" - "$RMBENCH_ROOT" "$CONFIG" "${@:3}" <<'PY'
import argparse, json, os, subprocess, sys
from pathlib import Path
import yaml

parser=argparse.ArgumentParser()
parser.add_argument('root',type=Path);parser.add_argument('config',type=Path)
parser.add_argument('--smoke',action='store_true')
parser.add_argument('--smoke-checkpoint',type=Path)
args=parser.parse_args()
root,path,smoke=args.root,args.config,args.smoke or args.smoke_checkpoint is not None
recipe = yaml.safe_load(path.read_text())
launch, ev, wb = recipe['launcher'], recipe['evaluation'], recipe['wandb']
checkpoint = args.smoke_checkpoint or Path(launch['checkpoint'])
metadata = json.loads((checkpoint/'tempo_run.json').read_text())
if metadata.get('training', {}).get('wandb_run_id') != wb['training_run_id']:
    raise ValueError('Checkpoint training run ID mismatch')
if metadata.get('step') != int(checkpoint.name):
    raise ValueError('Checkpoint step mismatch')
shared_python = str(Path(launch['shared_policy_environment'])/'bin/python')
shared_site = subprocess.check_output([shared_python, '-c',
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
eval_python = str(Path(launch['eval_environment'])/'bin/python')
eval_site = subprocess.check_output([eval_python, '-c',
    'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
os.environ['PYTHONPATH'] = os.pathsep.join([
    str(root), str(root/'XPolicyLab'), str(Path(os.environ['SMOVLA_ROOT'])/'src'),
    eval_site, shared_site,
])
output = Path(launch['output_root'])/('smoke.json' if smoke else 'results.json')
if output.exists():
    raise FileExistsError(f'Refusing to overwrite completed evaluation: {output}')
command = [eval_python, str(root/'scripts/lte/eval_lte_smolvla_temporal.py'),
    '--checkpoint', str(checkpoint), '--policy-kind', launch['policy_kind'],
    '--environment', ev['task'], '--temporal-env-root', launch['temporal_env_root'],
    '--episodes', str(1 if smoke else ev['episodes']),
    '--max-steps', str(5 if smoke else ev['max_steps']), '--seed', str(ev['seed']),
    '--prompt', ev['instruction'], '--output', str(output)]
if wb.get('enabled') and not smoke:
    command += ['--wandb-entity', wb['entity'], '--wandb-project', wb['project'],
                '--wandb-run-name', wb['run_name'], '--experiment-id', wb['group'],
                '--training-run-id', wb['training_run_id']]
print('Starting concurrent temporal evaluation:', json.dumps({
    'checkpoint': str(checkpoint), 'episodes': 1 if smoke else ev['episodes'],
    'smoke': smoke, 'output': str(output)}), flush=True)
os.execv(eval_python, command)
PY
