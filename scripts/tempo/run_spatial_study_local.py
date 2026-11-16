#!/usr/bin/env python3
"""Matched spatial tests: smoke BOTH, then one train+eval per local GPU.

No cloud writes, distributed optimizer, checkpoint upload, or machine shutdown.
Existing results/checkpoints are never overwritten. Each worker owns one GPU;
the training seeds, batch size and native LR schedule remain unchanged.
"""
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
from spatial_study import validate_recipe as validate_spatial, check_smoke
from queue_reconstruction_baselines import validate_eval
from queue_training_after_run import verify_checkpoint


def record(path, **values):
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(values, updated=datetime.now(timezone.utc).isoformat())
    tmp = path.with_suffix('.partial'); tmp.write_text(json.dumps(old, indent=2) + '\n'); tmp.replace(path)
    print(json.dumps(old), flush=True)


def load_studies(queue, directory):
    if queue['execution'] != 'local': raise ValueError('Only explicitly local jobs supported')
    reference = yaml.safe_load((directory / queue['reference_config']).read_text())
    defaults = yaml.safe_load((directory / queue['defaults_config']).read_text())
    state_root = Path(queue['state_root'])
    base_root = Path('/mnt/wdblack/tempo/waitatgoal-spatial-261110')
    input_root = Path(queue.get('input_state_root', queue['state_root']))
    ssd_root = Path('/home/sydney1/Repos/RMBench/logs/waitatgoal-spatial-261110-rerun1')
    if (not (state_root.is_relative_to(base_root) or state_root == ssd_root)
            or '..' in state_root.parts or input_root != base_root):
        raise ValueError('Unexpected local study directory')
    studies = []
    if [r['mode'] for r in queue['runs']] != ['spatial4', 'native64']:
        raise ValueError('Exactly spatial4 and native64 required')
    if len({r['gpu_uuid'] for r in queue['runs']}) != 2:
        raise ValueError('Independent GPUs required; no distributed batch-size change')
    for item in queue['runs']:
        path, evpath = directory / item['training_config'], directory / item['eval_config']
        recipe = yaml.safe_load(path.read_text()); evaluation = yaml.safe_load(evpath.read_text())
        canonical = deepcopy(recipe); t = recipe['training']; native = item['mode'] == 'native64'
        expected_dataset = input_root / 'datasets/waitatgoal_v4'
        expected_cache = input_root / ('embedding_cache/smolvla_waitatgoal_camera1_native64' if native
                                      else 'embedding_cache/smolvla_waitatgoal_camera1_pool2')
        expected_output = state_root / 'checkpoints' / f"waitatgoal_tempo_{item['mode']}_horizon10_10k_seed1"
        if (Path(t['dataset_root']), Path(t['embedding_cache_dir']), Path(t['output_dir'])) != (
                expected_dataset, expected_cache, expected_output):
            raise ValueError('Incorrect local dataset/cache/checkpoint paths')
        # Rebind ONLY machine paths before applying the existing strict reference
        # guard; do not weaken numerical/provenance/objective comparisons.
        for key in ('dataset_root', 'embedding_cache_dir'):
            canonical['training'][key] = t[key].replace(str(input_root), '/data/RMBench-state')
        canonical['training']['output_dir'] = t['output_dir'].replace(str(state_root), '/data/RMBench-state')
        validate_spatial(canonical, reference, item['mode'], defaults)
        validate_eval(t, evaluation)
        launch = evaluation['launcher']
        if (launch['execution'] != 'local' or launch['eval_environment'] != str(Path(queue['python']).parent.parent)
                or launch['temporal_env_root'] != str(input_root / 'deps/temporal-eval')):
            raise ValueError('Evaluation must use the prepared local interpreter and pinned simulator')
        studies.append((item, path, recipe, evpath, evaluation))
    return studies


def environment(queue, root, gpu):
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=gpu, SMOVLA_ROOT=queue['smovla_root'], HF_HOME=queue['hf_home'],
               HF_LEROBOT_HOME=str(Path(queue['state_root'])/'datasets/lerobot'),
               WANDB_DIR=str(Path(queue['state_root'])/'wandb'), WANDB_ENTITY='uts_robot_lab',
               PYTHONPATH=os.pathsep.join([str(root/'scripts/tempo'),str(root),str(root/'XPolicyLab'),
                                         str(Path(queue['smovla_root'])/'src')]),
               OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
               SDL_VIDEODRIVER='dummy', SDL_AUDIODRIVER='dummy', NUMBA_DISABLE_JIT='1')
    env.pop('LD_LIBRARY_PATH', None)
    return env


def eval_command(queue, root, evaluation, *, smoke):
    launch, ev, wb = evaluation['launcher'], evaluation['evaluation'], evaluation['wandb']
    checkpoint = Path(launch['checkpoint'])
    if smoke: checkpoint = checkpoint.parent.with_name(checkpoint.parent.name+'_smoke') / '000003'
    metadata = json.loads((checkpoint/'tempo_run.json').read_text())
    if metadata['training']['wandb_run_id'] != wb['training_run_id'] or metadata['step'] != int(checkpoint.name):
        raise ValueError('Checkpoint identity/step mismatch')
    output = Path(launch['output_root']) / ('smoke.json' if smoke else 'results.json')
    if output.exists(): raise FileExistsError(output)
    command = [queue['python'],str(root/'scripts/lte/eval_lte_smolvla_temporal.py'),
               '--checkpoint',str(checkpoint),'--policy-kind','tempo_smolvla',
               '--environment',ev['task'],'--temporal-env-root',launch['temporal_env_root'],
               '--episodes',str(1 if smoke else ev['episodes']),
               '--max-steps',str(5 if smoke else ev['max_steps']),
               '--seed',str(ev['seed']),'--prompt',ev['instruction'],'--output',str(output)]
    if wb['enabled'] and not smoke:
        command += ['--wandb-entity',wb['entity'],'--wandb-project',wb['project'],
                    '--wandb-run-name',wb['run_name'],'--experiment-id',wb['group'],
                    '--training-run-id',wb['training_run_id']]
    return command, output


def worker(args, queue, study):
    item, path, recipe, _, evaluation = study
    mode = item['mode']; t = recipe['training']; state = Path(queue['state_root'])/(mode+'.json')
    env = environment(queue,args.root,item['gpu_uuid'])
    def run(command): subprocess.run(command,cwd=args.root,env=env,check=True)
    try:
        if args.phase == 'smoke':
            record(state,stage='smoke_training',training_run_id=t['wandb_run_id'],gpu_uuid=item['gpu_uuid'])
            run([queue['python'],'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path),
                 '--smoke','--smoke-batch-size','8'])
            resolved = check_smoke(recipe)
            record(state,stage='smoke_evaluating',parameter_counts=resolved['parameter_counts'])
            command,output=eval_command(queue,args.root,evaluation,smoke=True);run(command)
            result=json.loads(output.read_text())
            if result['episodes']!=1 or result['checkpoint_step']!=3:raise ValueError('Eval smoke mismatch')
            record(state,stage='smoke_passed',smoke_verified=True)
        else:
            old=json.loads(state.read_text())
            if old['stage']!='smoke_passed' or old['training_run_id']!=t['wandb_run_id']:
                raise ValueError('Full-batch smoke must pass before training')
            record(state,stage='training',training_config=str(path))
            run([queue['python'],'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path)])
            checkpoint=verify_checkpoint(t,t['steps'])
            record(state,stage='evaluating',checkpoint=str(checkpoint))
            command,output=eval_command(queue,args.root,evaluation,smoke=False);run(command)
            result=json.loads(output.read_text())
            if (result['training_run_id'],result['checkpoint_step'],result['episodes'],result['checkpoint']) != (
                    t['wandb_run_id'],10000,100,str(checkpoint)):
                raise ValueError('Final evaluation mismatch')
            record(state,stage='complete',result=str(output),success_rate=result['success_rate'])
            if queue.get('archive_root'):
                # Training and eval have exited: this archival copy owns no GPU
                # context and cannot delay either policy's evaluation.
                archive=Path(queue['archive_root'])/mode
                record(state,stage='archiving',archive_root=str(archive),evaluation_complete=True)
                archive.mkdir(parents=True,exist_ok=False)
                subprocess.run(['rsync','-a','--partial',str(Path(t['output_dir']))+'/',
                                str(archive/'checkpoints')+'/'],check=True)
                shutil.copyfile(output,archive/'results.json')
                record(state,stage='complete',archive_complete=True)
    except BaseException as error:
        record(state,stage='failed',phase=args.phase,error=repr(error));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--mode',choices=('spatial4','native64'));p.add_argument('--phase',choices=('smoke','train-eval'))
    p.add_argument('--validate-only',action='store_true');args=p.parse_args()
    queue=yaml.safe_load(args.config.read_text());studies=load_studies(queue,args.config.parent)
    if args.validate_only:print(json.dumps({'valid':True,'gpus':[r[0]['gpu_uuid'] for r in studies]}));return
    if args.mode:
        if not args.phase:raise ValueError('Worker phase required')
        worker(args,queue,next(s for s in studies if s[0]['mode']==args.mode));return
    root=Path(queue['state_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'local-study.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=root/'local-study.json'
        if state.exists():raise FileExistsError(state)
        for item,_,recipe,_,evaluation in studies:
            for candidate in (recipe['training']['output_dir'],recipe['training']['output_dir']+'_smoke',
                              evaluation['launcher']['output_root'],root/(item['mode']+'.json')):
                if Path(candidate).exists():raise FileExistsError(candidate)
        if shutil.disk_usage(root).free<20*2**30:raise RuntimeError('Require 20GiB free; no deletion performed')
        # Display/remote-desktop allocations are not training jobs. All other
        # compute processes on either selected GPU prevent accidental overlap.
        processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                                           '--format=csv,noheader'],text=True)
        for row in processes.splitlines():
            gpu,_,name=[v.strip() for v in row.split(',',2)]
            if gpu in {r[0]['gpu_uuid'] for r in studies} and name!='/usr/libexec/gnome-remote-desktop-daemon':
                raise RuntimeError('Selected GPU is occupied: '+row)
        inputs=Path(queue.get('input_state_root', queue['state_root']))
        info=inputs/'datasets/waitatgoal_v4/meta/info.json'
        if hashlib.sha256(info.read_bytes()).hexdigest()!='f474f42c06c7ec6e698271ae5381d64e7d8af99b3f1f1e902339fcfab978121e':
            raise ValueError('Reference dataset info hash mismatch')
        envsource=inputs/'deps/temporal-eval/ltedp/envs/waitatgoal/waitatgoal_env.py'
        if hashlib.sha256(envsource.read_bytes()).hexdigest()!='eccfb8d5862b05265ae5461f2f26508ebdc3c0c3d0aad5bd84627ea3c63d117f':
            raise ValueError('Reference simulator hash mismatch')
        (root/'logs').mkdir(exist_ok=True);(root/'wandb').mkdir(exist_ok=True)
        record(state,stage='smoking_both',source_root=str(args.root),cloud_duplicate_removed=True,
               run_ids=[s[2]['training']['wandb_run_id'] for s in studies],instance_action='none',
               evaluation_scheduled=True)
        try:
            for phase in ('smoke','train-eval'):
                children=[]
                for study in studies:
                    mode=study[0]['mode'];log=(root/'logs'/(mode+'-'+phase+'.log')).open('x')
                    command=[queue['python'],str(Path(__file__).resolve()),'--root',str(args.root),
                             '--config',str(args.config),'--mode',mode,'--phase',phase]
                    child=subprocess.Popen(command,cwd=args.root,env=environment(queue,args.root,study[0]['gpu_uuid']),
                                           stdout=log,stderr=subprocess.STDOUT)
                    log.close();children.append((mode,child))
                failures=[]
                for mode,child in children:
                    if child.wait()!=0:failures.append(mode)
                if failures:raise RuntimeError(phase+' failed: '+','.join(failures))
                record(state,stage='training_and_evaluating' if phase=='smoke' else 'complete',both_smokes_passed=True)
        except BaseException as error:
            record(state,stage='failed',error=repr(error));raise


if __name__=='__main__':main()
