#!/usr/bin/env python3
"""Wait, download all checkpoints with SHA verification, then evaluate locally.

There are deliberately no Brev stop/delete or W&B artifact-upload operations.
The remote training state, rather than elapsed time, gates checkpoint download.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path,PurePosixPath
import re
import shlex
import subprocess
import sys
import time
import yaml
from export_checkpoint_manifest import digest


def verify_tree(root, entries):
    root=Path(root).resolve()
    for entry in entries:
        relative=PurePosixPath(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe inventory path')
        path=root/relative
        if not path.resolve().is_relative_to(root) or path.is_symlink():raise ValueError('Unsafe local file')
        if path.stat().st_size!=entry['size'] or digest(path)!=entry['sha256']:
            raise ValueError(f'Checksum mismatch: {path}')


def transfer(instance, remote, local):
    if not re.fullmatch(r'/[A-Za-z0-9_./-]+',remote):raise ValueError('Unsafe remote path')
    subprocess.run(['rsync','-aL','--partial','--checksum','--protect-args',f'{instance}:{remote}',str(local)],check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    recipe_path=args.config.resolve()
    recipe=yaml.safe_load(recipe_path.read_text())
    host=recipe['instance']
    if not re.fullmatch('[A-Za-z0-9-]+',host):raise ValueError('Invalid instance name')
    root=recipe_path.parents[2]
    local=(recipe_path.parent/recipe['local_directory']).resolve()
    local.mkdir(parents=True,exist_ok=True)
    status_path=local/'followup_status.json'
    if status_path.is_file() and json.loads(status_path.read_text()).get('stage')=='complete':return
    status={'training_run_id':recipe['training_run_id'],'instance':host,'instance_action':'none'}
    def record(stage,**extra):
        status.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        temporary=status_path.with_suffix('.partial')
        temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(status_path)
        print(json.dumps(status),flush=True)
    try:
        record('waiting_for_training')
        while True:
            query=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=20',host,
                                  shlex.join(['cat',recipe['remote_state']])],capture_output=True,text=True)
            if query.returncode:
                print('Training state temporarily unavailable; retaining instance and retrying.',flush=True)
            else:
                state=json.loads(query.stdout)
                if state.get('training_run_id')!=recipe['training_run_id']:raise ValueError('Remote run ID mismatch')
                if state.get('stage')=='failed':raise RuntimeError(state.get('error','Remote training failed'))
                if state.get('stage')=='awaiting_local_download':break
            time.sleep(recipe.get('poll_seconds',30))
        record('building_remote_inventory')
        command=[recipe['remote_python'],'-','--state',recipe['remote_state'],'--run-id',recipe['training_run_id'],
                 '--bootstrap',recipe['remote_bootstrap']]
        result=subprocess.run(['ssh','-T',host,shlex.join(command)],
                              input=(root/'scripts/tempo/export_checkpoint_manifest.py').read_text(),
                              capture_output=True,text=True,check=True)
        manifest=json.loads(result.stdout)
        (local/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        record('downloading_checkpoints',steps=manifest['steps'])
        transfer(host,manifest['checkpoints']['root']+'/',str(local)+'/')
        assets=local/'vlm_assets';assets.mkdir(exist_ok=True)
        transfer(host,manifest['vlm_assets']['root']+'/',str(assets)+'/')
        archive=local/'source-bundle.tar.zst'
        transfer(host,manifest['source_bundle']['path'],archive)
        record('verifying_downloads')
        verify_tree(local,manifest['checkpoints']['files'])
        verify_tree(assets,manifest['vlm_assets']['files'])
        if digest(archive)!=manifest['source_bundle']['sha256']:raise ValueError('Source bundle checksum mismatch')
        (local/'download_verified.json').write_text(json.dumps({'training_run_id':recipe['training_run_id'],
            'steps':manifest['steps'],'verified':True},indent=2)+'\n')
        source=local/'source'
        source.mkdir(exist_ok=True)
        subprocess.run(['tar','--zstd','-xf',str(archive),'-C',str(source),'--no-same-owner'],check=True)
        record('download_verified_instance_unchanged')
        evaluation=yaml.safe_load((recipe_path.parent/recipe['evaluation_config']).read_text())
        launch,ev,wb=evaluation['launcher'],evaluation['evaluation'],evaluation['wandb']
        training=yaml.safe_load((recipe_path.parent/recipe['training_config']).read_text())['training']
        final=local/f"{training['steps']:06d}"
        if final!=Path(launch['checkpoint']) or wb['training_run_id']!=recipe['training_run_id']:
            raise ValueError('Local evaluation checkpoint/source mismatch')
        environment=dict(os.environ,SMOVLA_ROOT=str((recipe_path.parent/recipe['smovla_root']).resolve()),
                         PYTHONPATH=str(source/'RMBench'),SDL_VIDEODRIVER='dummy',SDL_AUDIODRIVER='dummy',
                         TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        command=[launch['policy_python'],str(root/'scripts/lte/eval_lte_smolvla_temporal.py'),
                 '--checkpoint',str(final),'--policy-kind','tempo_smolvla','--environment',ev['task'],
                 '--temporal-env-root',launch['temporal_env_root'],'--episodes',str(ev['episodes']),
                 '--max-steps',str(ev['max_steps']),'--seed',str(ev['seed']),'--vlm-assets',str(assets),
                 '--prompt',ev['instruction'],'--output',str(Path(launch['output_root'])/'results.json')]
        if wb.get('enabled'):
            command.extend(['--wandb-entity',wb['entity'],'--wandb-project',wb['project'],
                            '--wandb-run-name',wb['run_name'],'--experiment-id',wb['group'],
                            '--training-run-id',wb['training_run_id']])
        record('evaluating_locally')
        subprocess.run(command,env=environment,cwd=source/'RMBench',check=True)
        record('complete')
    except BaseException as error:
        record('failed',error=repr(error));raise


if __name__=='__main__':main()
