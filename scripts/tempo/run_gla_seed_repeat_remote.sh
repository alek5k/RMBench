#!/usr/bin/env bash
# GLA seed repeat or explicitly requested exact historical reproduction.
set -Eeuo pipefail
[[ $# -eq 3 || ( $# -eq 4 && ( "$4" == --exact-reproduction || "$4" == --allow-action-horizon-change ) ) ]] || {
    echo 'Usage: run_gla_seed_repeat_remote.sh <release-root> <training.yaml> <reference-resolved.json> [--exact-reproduction|--allow-action-horizon-change]' >&2
    exit 2
}
MODE=seed_repeat
if [[ $# -eq 4 ]]; then
    if [[ "$4" == --exact-reproduction ]]; then MODE=exact_reproduction; else MODE=action_horizon_study; fi
fi
ROOT=$(realpath "$1")
CONFIG=$(realpath "$2")
REFERENCE=$(realpath "$3")
STATE=/data/RMBench-state
export SMOVLA_ROOT="$STATE/deps/smovla"
export PYTHONPATH="$ROOT:$ROOT/XPolicyLab:$SMOVLA_ROOT/src"
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot"
export WANDB_DIR="$STATE/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "$STATE/credentials.env"
set +a
cd "$ROOT"
exec "$HOME/RMBench-state/envs/tempo-smolvla/bin/python" - "$ROOT" "$CONFIG" "$REFERENCE" "$STATE" "$MODE" <<'PY'
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import subprocess, sys, time
import yaml

root, config_path, reference_path, state_root = map(Path, sys.argv[1:5])
exact_reproduction = sys.argv[5] == 'exact_reproduction'
action_horizon_study = sys.argv[5] == 'action_horizon_study'
recipe = yaml.safe_load(config_path.read_text())
training, memory = recipe['training'], recipe['memory']
reference = json.loads(reference_path.read_text())
previous = reference['training']
if training.get('policy_variant', 'tempo') != 'tempo' or training.get('memory_backend') != 'gla':
    raise ValueError('Expected TEMPO SmolVLA with GLA')
if training.get('wandb_log_checkpoints', False):
    raise ValueError('Checkpoint uploads must remain disabled')
if training.get('validation_fraction', 0) or reference.get('validation'):
    raise ValueError('This seed comparison requires full-data training without holdout')
if memory != reference['policy']['memory']:
    raise ValueError('GLA memory settings differ from actual reference run')
allowed = {'seed', 'reference_wandb_run', 'output_dir', 'run_name', 'wandb_run_id', 'wandb_group'}
if action_horizon_study:
    from XPolicyLab.policy.TEMPO_SmolVLA.action_horizon import action_horizon_overrides
    overrides = action_horizon_overrides(training, reference['policy'])
    if set(overrides) != {'chunk_size', 'n_action_steps'}:
        raise ValueError('Horizon studies must explicitly configure prediction and execution horizons')
    allowed.update(overrides)
differences = {k: [previous.get(k), training.get(k)] for k in previous.keys() | training.keys()
               if previous.get(k) != training.get(k)}
wrong_seed = (training['seed'] != previous['seed'] if exact_reproduction
              else training['seed'] == previous['seed'])
if set(differences) - allowed or wrong_seed:
    raise ValueError(f'Not a seed-only repeat: {differences}')
if training['reference_wandb_run'].split('/')[-1] != previous['wandb_run_id']:
    raise ValueError('Reference W&B identity mismatch')
sys.path.insert(0, str(root / 'scripts/tempo'))
from queue_training_after_run import verify_checkpoint
if not exact_reproduction:
    defaults = yaml.safe_load((config_path.parent / 'waitatgoal_defaults.yaml').read_text())['training']
    for key, expected in defaults.items():
        if training.get(key) != expected:
            raise ValueError(f'WaitAtGoal default mismatch: {key}')
# Exact historical reproductions deliberately retain the actual reference's
# scheduler/warmup, even when different from today's defaults. The comparison
# above rejects any training-parameter changes, including the seed.
run_id = training['wandb_run_id']
if not isinstance(run_id, str) or not run_id.isalnum():
    raise ValueError('Explicit alphanumeric run ID required')
state_path = state_root / f'gla-training-{run_id}.json'
if state_path.exists(): raise FileExistsError(state_path)
for output in (training['output_dir'], training['output_dir'] + '_smoke'):
    if Path(output).exists(): raise FileExistsError(output)
state = {'training_run_id': run_id, 'training_config': str(config_path),
         'source_root': str(root), 'reference_resolved_config': str(reference_path),
         'runner_pid': os.getpid(), 'evaluation_scheduled': False,
         'comparison_mode': sys.argv[5],
         'intended_differences': differences}
def record(stage, **extra):
    state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
    temporary = state_path.with_suffix('.partial')
    temporary.write_text(json.dumps(state, indent=2) + '\n'); temporary.replace(state_path)
    print(json.dumps(state), flush=True)
try:
    record('waiting_for_idle_gpu')
    while subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                  '--format=csv,noheader'], text=True).strip():
        time.sleep(5)
    record('regression_tests')
    modules = ['test_gated_memory', 'test_baseline', 'test_training_controls', 'test_wandb_metadata']
    if action_horizon_study: modules.append('test_action_horizon')
    subprocess.run([sys.executable, '-m', 'unittest',
                    *['XPolicyLab.policy.TEMPO_SmolVLA.tests.' + m for m in modules], '-v'], check=True)
    command = [sys.executable, '-m', 'XPolicyLab.policy.TEMPO_SmolVLA.train', '--config', str(config_path)]
    record('smoke_test')
    subprocess.run(command + ['--smoke', '--smoke-batch-size', str(training['batch_size'])], check=True)
    marker = json.loads((Path(training['output_dir'] + '_smoke') / 'smoke_passed.json').read_text())
    if not marker['finite'] or marker['memory_shape'] != [1, memory['num_tokens'], memory['width']]:
        raise RuntimeError('GLA checkpoint/reload/inference smoke test failed')
    if action_horizon_study and marker['actions_shape'][0] != training['chunk_size']:
        raise ValueError('Smoke inference did not use the requested prediction horizon')
    smoke_root = Path(training['output_dir'] + '_smoke')
    resolved = json.loads((smoke_root / 'resolved_config.json').read_text())
    for key in ('policy', 'parameter_counts', 'pretrained_sha256', 'dataset_frames'):
        expected = reference[key]
        if key == 'policy' and action_horizon_study:
            expected = {**expected, **overrides}
        if resolved[key] != expected:
            raise ValueError(f'Smoke/reference resolved metadata differs: {key}')
    source_checkpoint = Path(previous['output_dir']) / f"{previous['steps']:06d}"
    if not source_checkpoint.is_dir():
        # A cancelled reference can still supply its successfully reloaded smoke
        # checkpoint's identical fitted processors (never unfinished weights).
        source_checkpoint = Path(previous['output_dir'] + '_smoke') / '000003'
    processor_files = list(source_checkpoint.glob('policy_*processor*'))
    if not processor_files:
        raise FileNotFoundError(f'No reference processors in {source_checkpoint}')
    for source in processor_files:
        target = smoke_root / '000003' / source.name
        if hashlib.sha256(source.read_bytes()).digest() != hashlib.sha256(target.read_bytes()).digest():
            raise ValueError(f'Saved normalizer/processor differs: {source.name}')
    record('reference_metadata_verified')
    record('training')
    subprocess.run(command, check=True)
    record('complete', checkpoint=str(verify_checkpoint(training, training['steps'])))
except BaseException as error:
    record('stopped_by_user' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error)); raise
PY
