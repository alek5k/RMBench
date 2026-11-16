#!/usr/bin/env python3
"""Smoke then evaluate every verified local TEMPO checkpoint, sequentially.

Original checkpoints are read-only. Portable views rebind only VLM/tokenizer
paths; tensors and all numerical/model settings remain unchanged. No cloud use.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def portable_checkpoint(source, destination, vlm_assets, run_id, step, *, expected_horizon=10):
    """Symlink original weights/statistics and rewrite only two machine-local paths."""
    metadata = json.loads((source/'tempo_run.json').read_text())
    if metadata['step'] != step or metadata['training']['wandb_run_id'] != run_id:
        raise ValueError('Checkpoint step/source identity mismatch')
    config = json.loads((source/'config.json').read_text())
    if (config['type'], config['memory_backend'], config['chunk_size'], config['n_action_steps']) != ('tempo_smolvla', 'gla', expected_horizon, expected_horizon):
        raise ValueError(f'Expected trained horizon-{expected_horizon} TEMPO GLA checkpoint')
    portable = deepcopy(config);portable['vlm_model_name'] = str(vlm_assets)
    processor = json.loads((source/'policy_preprocessor.json').read_text())
    portable_pre = deepcopy(processor);tokenizers = 0
    for item in portable_pre['steps']:
        if item['registry_name'] == 'tokenizer_processor':
            item['config']['tokenizer_name'] = str(vlm_assets);tokenizers += 1
    if tokenizers != 1:raise ValueError('Expected exactly one saved tokenizer')
    if destination.exists():
        if (json.loads((destination/'config.json').read_text()) != portable or
            json.loads((destination/'policy_preprocessor.json').read_text()) != portable_pre):
            raise ValueError('Existing portable checkpoint differs')
        for item in source.iterdir():
            if item.is_file() and item.name not in ('config.json', 'policy_preprocessor.json', 'optimizer.pt'):
                if (destination/item.name).resolve() != item.resolve():raise ValueError('Existing portable tensor link differs')
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for item in source.iterdir():
        if item.is_file() and item.name not in ('config.json', 'policy_preprocessor.json', 'optimizer.pt'):
            (destination/item.name).symlink_to(item.resolve())
    (destination/'config.json').write_text(json.dumps(portable, indent=2)+'\n')
    (destination/'policy_preprocessor.json').write_text(json.dumps(portable_pre, indent=2)+'\n')
    return destination


def completed_result(directory, run_id, step, episodes):
    paths = list(directory.glob('*/completed.json'))
    if len(paths) != 1:raise ValueError('Missing/ambiguous evaluation result')
    result = json.loads(paths[0].read_text())
    if (result['training_run_id'], result['checkpoint_step'], result['episodes']) != (run_id, step, episodes):
        raise ValueError('Evaluation result source/step/episode mismatch')
    return result


def skipped_checkpoint(directory, run_id, step, episodes):
    """Explicit user-authorized skips are separate from completed evaluation results."""
    path = directory/'skipped.json'
    if not path.exists():return None
    result = json.loads(path.read_text())
    if ((result.get('training_run_id'), result.get('checkpoint_step'), result.get('requested_episodes'))
            != (run_id, step, episodes) or result.get('status') != 'skipped_by_user'
            or result.get('user_authorized') is not True or not result.get('reason')):
        raise ValueError('Invalid/mismatched checkpoint skip authorization')
    if not 0 <= result['episodes_completed'] < episodes:
        raise ValueError('Skipped evaluation must remain explicitly incomplete')
    return result


def evaluation_environment(recipe, parent=None):
    environment = dict(os.environ if parent is None else parent,
                       CUDA_VISIBLE_DEVICES=recipe['cuda_visible_devices'], SMOVLA_ROOT=recipe['smovla_root'],
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
                       PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
                       DISABLE_VK_LAYER_VALVE_steam_overlay_1='1', DISABLE_VK_LAYER_VALVE_steam_fossilize_1='1')
    # Resolve the matching packaged CUDA libraries, not system cuBLASLt 12.9.
    environment.pop('LD_LIBRARY_PATH', None)
    return environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--resume', action='store_true', help='Resume only a stopped/failed sweep; preserve prior results.')
    args = parser.parse_args();recipe = yaml.safe_load(args.config.read_text())
    root = Path(recipe['source_root']);checkpoints = Path(recipe['checkpoint_root'])
    output = Path(recipe['output_root']);vlm = Path(recipe['vlm_assets'])
    run_id = recipe['training_run_id'];steps = recipe['checkpoint_steps']
    for path in [root/'XPolicyLab/setup_policy_server.py', root/'scripts/eval_policy_xpolicylab.py',
                 root/'XPolicyLab/utils/wait_for_policy_server.sh', root/'assets/objects/objaverse/list.json']:
        if not path.is_file():raise FileNotFoundError(f'Pinned source/assets not ready: {path}')
    if steps != list(range(10000, 100001, 10000)):raise ValueError('Expected all ten training checkpoints')
    verification = json.loads(checkpoints.with_name(checkpoints.name+'.verified.json').read_text())
    if not verification['verified'] or verification['training_run_id'] != run_id:
        raise ValueError('Missing verified local backup')
    for filename in ('config.json', 'tokenizer_config.json'):
        if not (vlm/filename).is_file():raise FileNotFoundError(vlm/filename)
    output.mkdir(parents=True, exist_ok=True);state_path = output/'state.json'
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            if not args.resume:raise FileExistsError(state_path)
            state = json.loads(state_path.read_text())
            if state['training_run_id'] != run_id or state['stage'] != 'failed':raise ValueError('Only this failed sweep can resume')
            state.setdefault('previous_failures', []).append({'error': state.pop('error', None), 'updated': state['updated']})
            state['queue_pid'] = os.getpid()
        else:state = {'queue_pid': os.getpid(), 'training_run_id': run_id, 'execution': 'local',
                 'cuda_visible_devices': recipe['cuda_visible_devices'], 'checkpoint_steps': steps, 'completed': []}
        def record(stage, **extra):
            state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
            partial = state_path.with_suffix('.partial');partial.write_text(json.dumps(state, indent=2)+'\n')
            partial.replace(state_path);print(json.dumps(state), flush=True)
        try:
            record('preparing_portable_checkpoints')
            portable = {step: portable_checkpoint(checkpoints/f'{step:06d}', Path(recipe['portable_root'])/f'{step:06d}', vlm, run_id, step)
                        for step in steps}
            configs = output/'configs';configs.mkdir(exist_ok=True)
            environment = evaluation_environment(recipe)
            launcher = Path(__file__).with_name('evaluate_smolvla.py')
            def evaluate(step, *, smoke=False):
                target = output/('smoke' if smoke else f'{step:06d}')
                if list(target.glob('*/completed.json')):
                    return completed_result(target, run_id, step, recipe['smoke_episodes'] if smoke else recipe['evaluation']['episodes'])
                settings = {key: deepcopy(recipe[key]) for key in ('launcher', 'evaluation', 'wandb')}
                settings['launcher'].update(checkpoint=str(portable[step]), output_root=str(target),
                                             smovla_vlm_path=str(vlm), smovla_root=recipe['smovla_root'])
                if smoke:
                    settings['evaluation']['episodes'] = recipe['smoke_episodes'];settings['wandb']['enabled'] = False
                config = configs/('smoke.yaml' if smoke else f'{step:06d}.yaml')
                config.write_text(yaml.safe_dump(settings, sort_keys=False))
                command = [sys.executable, str(launcher), '--root', str(root), '--config', str(config)]
                with (output/('smoke.log' if smoke else f'{step:06d}.log')).open('a') as log:
                    subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                return completed_result(target, run_id, step, settings['evaluation']['episodes'])
            record('smoke_evaluation', checkpoint_step=100000)
            smoke = evaluate(100000, smoke=True);record('smoke_passed', smoke=smoke)
            for step in steps:
                skipped = skipped_checkpoint(output/f'{step:06d}', run_id, step, recipe['evaluation']['episodes'])
                if skipped is not None:
                    state['skipped'] = [r for r in state.get('skipped', []) if r['checkpoint_step'] != step]+[skipped]
                    record('checkpoint_skipped', checkpoint_step=step)
                    continue
                record('evaluating', checkpoint_step=step)
                result = evaluate(step)
                state['completed'] = [r for r in state['completed'] if r['checkpoint_step'] != step]+[result]
                record('checkpoint_complete', checkpoint_step=step)
            record('complete')
        except BaseException as error:
            record('failed', error=repr(error));raise


if __name__ == '__main__':main()
