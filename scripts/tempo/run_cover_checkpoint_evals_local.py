#!/usr/bin/env python3
"""Sequential Cover Blocks checkpoint evals on an idle local GPU; no training writes."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

from run_local_checkpoint_evals import completed_result, evaluation_environment


def verify_checkpoints(recipe):
    """Require the requested trained identities, horizon, queries and shared group."""
    root = Path(recipe['checkpoint_root'])
    configs = []
    for step in recipe['checkpoint_steps']:
        checkpoint = root/f'{step:06d}'
        for name in ('model.safetensors', 'config.json', 'tempo_run.json',
                     'policy_preprocessor.json', 'policy_postprocessor.json'):
            if not (checkpoint/name).is_file() or (checkpoint/name).stat().st_size == 0:
                raise FileNotFoundError(checkpoint/name)
        metadata = json.loads((checkpoint/'tempo_run.json').read_text())
        config = json.loads((checkpoint/'config.json').read_text())
        training = metadata['training']
        if (metadata['step'], training['wandb_run_id'], training['task_name'], training['wandb_group']) != (
                step, recipe['training_run_id'], 'cover_blocks', recipe['wandb']['group']):
            raise ValueError('Wrong checkpoint identity, task, step or W&B group')
        if (config['type'], config['memory_backend'], config['chunk_size'], config['n_action_steps'],
                config['memory']['history_queries'], config['memory']['num_tokens'],
                config['temporal_objective']) != ('tempo_smolvla', 'gla', 50, 50, 128, 4, 'joint'):
            raise ValueError('Expected joint 128-lag-query TEMPO, four read tokens, predict50/act50')
        if not Path(config['vlm_model_name']).is_dir():
            raise FileNotFoundError(config['vlm_model_name'])
        configs.append(config)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError('Checkpoint policy configurations differ')
    if recipe['wandb']['training_run_id'] != recipe['training_run_id']:
        raise ValueError('Eval logger has the wrong training identity')
    return configs[0]


def prepare_source(recipe):
    """Copy the exact training release; attach existing simulator assets only."""
    source, target = Path(recipe['source_root']), Path(recipe['evaluation_source'])
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite an evaluation release: {target}')
    shutil.copytree(source, target, symlinks=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    checked = 0
    for path in source.rglob('*'):
        relative = path.relative_to(source)
        if '__pycache__' in relative.parts or path.suffix == '.pyc' or path.is_symlink() or not path.is_file():
            continue
        with path.open('rb') as original, (target/relative).open('rb') as copied:
            if hashlib.file_digest(original, 'sha256').digest() != hashlib.file_digest(copied, 'sha256').digest():
                raise ValueError(f'Copied source differs: {relative}')
        checked += 1
    for name in ('objects', 'embodiments'):
        asset = Path(recipe['assets_root'])/name
        if not asset.is_dir():
            raise FileNotFoundError(asset)
        (target/'assets'/name).symlink_to(asset.resolve(), target_is_directory=True)
    for relative in ('assets/objects/003_cover', 'assets/objects/objaverse/list.json',
                     'assets/embodiments/aloha-agilex'):
        if not (target/relative).exists():
            raise FileNotFoundError(target/relative)
    return target, checked


def verify_gpu_idle(uuid):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    matches = [line.split(',')[0].strip() for line in rows.splitlines() if line.split(',')[-1].strip() == uuid]
    if matches != ['1']:
        raise ValueError(f'Expected the second local GPU, got {matches}')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                         '--format=csv,noheader'], text=True)
    if any(line.split(',')[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError('GPU 1 is occupied; no existing process will be interrupted')


def evaluation_recipe(recipe, step, *, smoke=False):
    settings = {key: deepcopy(recipe[key]) for key in ('launcher', 'evaluation', 'wandb')}
    target = Path(recipe['output_root'])/('smoke' if smoke else f'{step:06d}')
    settings['launcher'].update(checkpoint=str(Path(recipe['checkpoint_root'])/f'{step:06d}'),
                                output_root=str(target), smovla_root=recipe['smovla_root'])
    if smoke:
        settings['evaluation'].update(episodes=recipe['smoke_episodes'], num_workers=1)
        settings['wandb']['enabled'] = False
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    if recipe['checkpoint_steps'] != [100000, 200000] or recipe['evaluation']['task'] != 'cover_blocks':
        raise ValueError('Expected the requested Cover Blocks 100k and 200k checkpoint evals')
    output = Path(recipe['output_root'])
    output.mkdir(parents=True, exist_ok=False)
    state = {'queue_pid': os.getpid(), 'execution': 'local', 'instance_action': 'none',
             'training_run_id': recipe['training_run_id'], 'checkpoint_steps': recipe['checkpoint_steps'],
             'gpu_uuid': recipe['cuda_visible_devices'], 'evaluation_scheduled': True, 'completed': []}

    def record(stage, **extra):
        state.update(stage=stage, updated=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = output/'state.partial'
        temporary.write_text(json.dumps(state, indent=2)+'\n')
        temporary.replace(output/'state.json')
        print(json.dumps(state), flush=True)

    try:
        with (Path('/tmp')/f"tempo-local-{recipe['cuda_visible_devices']}.lock").open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record('verifying_checkpoints')
            verify_checkpoints(recipe)
            verify_gpu_idle(recipe['cuda_visible_devices'])
            record('preparing_evaluation_source')
            root, count = prepare_source(recipe)
            record('source_verified', source_root=str(root), source_files_sha256_verified=count)
            environment = evaluation_environment(recipe)
            environment.update(TMPDIR='/tmp', TMP='/tmp', TEMP='/tmp', PYTHONDONTWRITEBYTECODE='1')
            launcher = Path(__file__).with_name('evaluate_smolvla.py')
            (output/'configs').mkdir()

            def evaluate(step, *, smoke=False):
                label = 'smoke' if smoke else f'{step:06d}'
                settings = evaluation_recipe(recipe, step, smoke=smoke)
                config_path = output/'configs'/f'{label}.yaml'
                config_path.write_text(yaml.safe_dump(settings, sort_keys=False))
                record('smoke_evaluation' if smoke else 'evaluating', checkpoint_step=step)
                with (output/f'{label}.log').open('w') as log:
                    subprocess.run([sys.executable, str(launcher), '--root', str(root), '--config', str(config_path)],
                                   cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                return completed_result(Path(settings['launcher']['output_root']), recipe['training_run_id'],
                                        step, settings['evaluation']['episodes'])

            smoke = evaluate(recipe['checkpoint_steps'][0], smoke=True)
            record('smoke_passed', smoke=smoke)
            for step in recipe['checkpoint_steps']:
                result = evaluate(step)
                state['completed'].append(result)
                record('checkpoint_complete')
            record('complete')
    except BaseException as error:
        record('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
