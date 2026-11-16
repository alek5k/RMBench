#!/usr/bin/env bash
# Native control through the shared TEMPO trainer, with no eval or reinstall.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: run_smolvla_train_only_remote.sh <release-root> <training.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
STATE=/data/RMBench-state
PYTHON="${HOME}/RMBench-state/envs/tempo-smolvla/bin/python"
export SMOVLA_ROOT="${STATE}/deps/smovla"
export PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab:${SMOVLA_ROOT}/src"
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export WANDB_DIR="${STATE}/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "${STATE}/credentials.env"
set +a
cd "$ROOT"
exec "$PYTHON" - "$ROOT" "$CONFIG" "$STATE" <<'PY'
from datetime import datetime, timezone
import json, os
from pathlib import Path
import subprocess, sys, time
import yaml

root, config_path, state_root = map(Path, sys.argv[1:])
recipe = yaml.safe_load(config_path.read_text())
training = recipe['training']
if training.get('policy_variant') != 'smolvla' or recipe.get('memory'):
    raise ValueError('This runner requires a native SmolVLA control')
if training.get('wandb_log_checkpoints', False):
    raise ValueError('Checkpoint artifact uploads must stay disabled')
if any(k in training for k in ('memory_backend', 'embedding_cache_dir', 'temporal_camera_keys')):
    raise ValueError('Native baseline cannot configure temporal memory/cache')
run_id = training['wandb_run_id']
if not isinstance(run_id, str) or not run_id.isalnum():
    raise ValueError('Explicit alphanumeric run ID required')
state_path = state_root/f'smolvla-training-{run_id}.json'
if state_path.exists(): raise FileExistsError(state_path)
for output in (training['output_dir'], training['output_dir']+'_smoke'):
    if Path(output).exists(): raise FileExistsError(output)
state = {'training_run_id': run_id, 'training_config': str(config_path),
         'source_root': str(root), 'runner_pid': os.getpid(), 'evaluation_scheduled': False}
def record(stage, **extra):
    state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = state_path.with_suffix('.partial')
    temporary.write_text(json.dumps(state, indent=2)+'\n'); temporary.replace(state_path)
    print(json.dumps(state), flush=True)
try:
    record('waiting_for_idle_gpu')
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader'], text=True).strip():
        time.sleep(5)
    record('regression_tests')
    subprocess.run([sys.executable, '-m', 'unittest',
        'XPolicyLab.policy.TEMPO_SmolVLA.tests.test_baseline',
        'XPolicyLab.policy.TEMPO_SmolVLA.tests.test_training_controls',
        'XPolicyLab.policy.TEMPO_SmolVLA.tests.test_wandb_metadata', '-v'], check=True)
    command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(config_path)]
    record('smoke_test')
    subprocess.run(command+['--smoke', '--smoke-batch-size', str(training['batch_size'])], check=True)
    marker = json.loads((Path(training['output_dir']+'_smoke')/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['memory_shape'] is not None:
        raise RuntimeError('Native checkpoint/inference smoke test failed')
    record('training')
    subprocess.run(command, check=True)
    sys.path.insert(0, str(root/'scripts/tempo'))
    from queue_training_after_run import verify_checkpoint
    checkpoint = verify_checkpoint(training, training['steps'])
    record('complete', checkpoint=str(checkpoint))
except BaseException as error:
    record('failed', error=repr(error)); raise
PY
