#!/usr/bin/env python3
"""Local temporal evaluation using an isolated simulator overlay and pinned policy.

Smoke first, then evaluate the same checkpoint with the requested episode count.
No installs, training mutations, uploads, shutdowns, or checkpoint rewrites.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import yaml
from run_liftqa_native64_local import gpu_available


def validate(recipe):
    launch, ev, wb = recipe['launcher'], recipe['evaluation'], recipe['wandb']
    checkpoint = Path(launch['checkpoint'])
    metadata = json.loads((checkpoint / 'tempo_run.json').read_text())
    policy = json.loads((checkpoint / 'config.json').read_text())
    training = metadata['training']
    if launch['execution'] != 'local' or launch['policy_kind'] != policy['type']:
        raise ValueError('Local execution and exact checkpoint policy kind required')
    if (metadata['step'] != int(checkpoint.name) or training['wandb_run_id'] != wb['training_run_id']
            or training['wandb_group'] != wb['group'] or ev['task'] != training['task_name']):
        raise ValueError('Checkpoint step, run, group or task mismatch')
    if wb['run_name'] != training['run_name'] + '_eval' or not wb['enabled']:
        raise ValueError('Evaluation must be named after and linked to its training run')
    if ev['episodes'] != 100 or ev['seed'] != 0 or ev['max_steps'] <= 0:
        raise ValueError('Expected full 100-episode evaluation, fixed seed 0')
    source = Path(launch['temporal_env_root'])
    manifest = json.loads((source / 'ENVIRONMENT_SOURCE.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Pinned environment source differs: ' + relative)
    return metadata


def command(recipe, root, output, *, smoke):
    launch, ev, wb = recipe['launcher'], recipe['evaluation'], recipe['wandb']
    result = [str(Path(launch['eval_environment']) / 'bin/python'),
              str(root / 'scripts/lte/eval_lte_smolvla_temporal.py'),
              '--checkpoint', launch['checkpoint'], '--policy-kind', launch['policy_kind'],
              '--environment', ev['task'], '--temporal-env-root', launch['temporal_env_root'],
              '--episodes', str(1 if smoke else ev['episodes']),
              '--max-steps', str(5 if smoke else ev['max_steps']), '--seed', str(ev['seed']),
              '--prompt', ev['instruction'], '--output', str(output)]
    if not smoke:
        result += ['--wandb-entity', wb['entity'], '--wandb-project', wb['project'],
                   '--wandb-run-name', wb['run_name'], '--experiment-id', wb['group'],
                   '--training-run-id', wb['training_run_id']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text()); metadata = validate(recipe)
    launch, ev, wb = recipe['launcher'], recipe['evaluation'], recipe['wandb']
    if args.validate_only:
        print(json.dumps({'valid': True, 'training_run_id': wb['training_run_id'],
                          'checkpoint_step': metadata['step'], 'episodes': ev['episodes']})); return
    output = Path(launch['output_root']); output.mkdir(parents=True, exist_ok=True)
    with (output / 'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = output / 'evaluation.json'
        if state.exists():
            raise FileExistsError(state)
        for name in ('smoke.json', 'smoke.log', 'results.json', 'eval.log'):
            if (output / name).exists():
                raise FileExistsError(output / name)
        values = {'training_run_id': wb['training_run_id'], 'checkpoint': launch['checkpoint'],
                  'checkpoint_step': metadata['step'], 'episodes': ev['episodes'], 'queue_pid': os.getpid(),
                  'gpu_uuid': launch['gpu_uuid'], 'source_root': str(args.root)}
        def record(stage, **extra):
            values.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            tmp = state.with_suffix('.partial'); tmp.write_text(json.dumps(values, indent=2) + '\n'); tmp.replace(state)
            print(json.dumps(values), flush=True)
        env = dict(os.environ)
        shared_python = str(Path(launch['shared_policy_environment']) / 'bin/python')
        shared_site = subprocess.check_output([shared_python, '-c',
            'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        eval_python = str(Path(launch['eval_environment']) / 'bin/python')
        eval_site = subprocess.check_output([eval_python, '-c',
            'import sysconfig; print(sysconfig.get_paths()["purelib"])'], text=True).strip()
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        selected = [row.split(',')[0].strip() for row in inventory.splitlines()
                    if row.split(',')[1].strip() == launch['gpu_uuid']]
        if selected != [str(launch['mujoco_egl_device_id'])]:
            raise ValueError('CUDA/EGL physical GPU selection does not match requested UUID')
        env.update(PYTHONPATH=os.pathsep.join([str(args.root), str(args.root / 'XPolicyLab'),
                                              str(Path(launch['smovla_root']) / 'src'), eval_site, shared_site]),
                   # Pinned Robosuite requires numeric visibility for its EGL
                   # guard. Resolve/verify it from the requested UUID first.
                   SMOVLA_ROOT=launch['smovla_root'], CUDA_VISIBLE_DEVICES=selected[0], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   HF_HOME=launch['hf_home'], HF_HUB_OFFLINE='1', WANDB_DIR=str(output),
                   MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(launch['mujoco_egl_device_id']),
                   SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', NUMBA_DISABLE_JIT='1',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp')
        env.pop('LD_LIBRARY_PATH', None)
        try:
            if not gpu_available(launch['gpu_uuid']):
                raise RuntimeError('Selected GPU is occupied; nothing stopped')
            with Path('/tmp/tempo-local-' + launch['gpu_uuid'] + '.lock').open('a') as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for smoke in (True, False):
                    result = output / ('smoke.json' if smoke else 'results.json')
                    with (output / ('smoke.log' if smoke else 'eval.log')).open('x') as log:
                        record('smoke_evaluating' if smoke else 'evaluating')
                        child = subprocess.Popen(command(recipe, args.root, result, smoke=smoke),
                                                 cwd=args.root, env=env, stdout=log, stderr=subprocess.STDOUT)
                        record(values['stage'], child_pid=child.pid)
                        if child.wait():
                            raise RuntimeError('Evaluation subprocess failed: ' + str(child.returncode))
                    report = json.loads(result.read_text())
                    if (report['training_run_id'], report['checkpoint_step'], report['episodes']) != (
                            '' if smoke else wb['training_run_id'], metadata['step'], 1 if smoke else ev['episodes']):
                        raise ValueError('Incomplete or mismatched evaluation result')
                    if smoke:
                        record('smoke_passed', smoke_verified=True, child_pid=None)
                record('complete', child_pid=None, successes=report['successes'],
                       success_rate=report['success_rate'], result=str(result))
        except BaseException as error:
            record('failed', error=repr(error)); raise


if __name__ == '__main__':
    main()
