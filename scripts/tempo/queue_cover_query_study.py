#!/usr/bin/env python3
"""Cover Blocks lag-query ablations after both WaitAtGoal train/eval runs.

No current jobs are interrupted; all datasets, checkpoints and logs are retained.
Only the history decoder's sampled time-lag query count changes (32 then 64),
not the four learned memory-read queries, memory capacity or loss weight.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from queue_cover_blocks_suite import validate_evaluation
from queue_training_after_run import verify_checkpoint
from run_rmbench_task import validate_dataset


def validate_recipe(recipe, reference, queries):
    training, old = recipe['training'], reference['training']
    allowed = {'reference_wandb_run','steps','output_dir','wandb_run_id','wandb_group','run_name','wandb_tags'}
    changed = {key for key in training.keys() | old.keys() if training.get(key) != old.get(key)}
    if changed - allowed:
        raise ValueError('Unexpected training drift: ' + str(changed - allowed))
    if recipe['memory'] != {**reference['memory'], 'history_queries': queries}:
        raise ValueError('Only reconstruction time-lag query count may change')
    if (training['task_name'],training['steps'],training['batch_size'],training['seed'],
            training['chunk_size'],training['n_action_steps']) != ('cover_blocks',100000,8,0,50,50):
        raise ValueError('Expected Cover Blocks 100k, batch8, seed0, 50/50')
    if (training.get('temporal_objective'), training.get('train_language_model'),
            training['reference_wandb_run']) != ('joint',False,'uts_robot_lab/RMBench/8d2f6a10'):
        raise ValueError('Expected reference-matched joint loss with frozen backbone')
    if training.get('history_reconstruction_target','history') != 'history':
        raise ValueError('This study reconstructs time-specific history, not mean/current targets')
    if training.get('wandb_log_checkpoints',False) or training.get('validation_fraction',0):
        raise ValueError('No checkpoint artifacts or holdout in this comparison')
    if 'reconstruction-effectiveness' not in training.get('wandb_tags',()):
        raise ValueError('Missing study tag')
    for key in ('output_dir','wandb_run_id','wandb_group','run_name'):
        if training[key] == old[key]:raise ValueError('Fresh identity required: ' + key)


def predecessor_complete(state, expected_ids, result_paths):
    if state.get('run_ids') != expected_ids:
        raise ValueError('WaitAtGoal predecessor identities changed')
    if state.get('stage') in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
        raise RuntimeError('WaitAtGoal predecessor failed/cancelled; not starting new jobs')
    if state.get('stage') != 'complete':return False
    if {item['training_run_id'] for item in state['completed']} != set(expected_ids):
        raise ValueError('Incomplete WaitAtGoal study completion marker')
    for run_id,path in zip(expected_ids,result_paths,strict=True):
        result=json.loads(Path(path).read_text())
        if (result.get('training_run_id'),result.get('checkpoint_step'),result.get('episodes')) != (run_id,10000,100):
            raise ValueError('Incomplete/wrong WaitAtGoal final evaluation')
    return True


def check_smoke(recipe, reference, processors):
    import torch
    from safetensors.torch import load_file
    training=recipe['training'];root=Path(training['output_dir']+'_smoke')
    marker=json.loads((root/'smoke_passed.json').read_text())
    if not marker['finite'] or marker['actions_shape'] != [50,14] or marker['memory_shape'] != [1,4,64]:
        raise ValueError('Full-batch save/reload/action/memory smoke failed')
    actual=json.loads((root/'resolved_config.json').read_text())
    expected=copy.deepcopy(reference['policy'])
    expected.update(history_reconstruction_target='history',history_reconstruction_diagnostics=False)
    expected['memory']=recipe['memory']
    if actual['policy'] != expected:
        differences={key:[expected.get(key),actual['policy'].get(key)]
                     for key in expected.keys() | actual['policy'].keys() if expected.get(key)!=actual['policy'].get(key)}
        raise ValueError('Unexpected resolved policy drift: ' + str(differences))
    if (actual['parameter_counts'] != reference['parameter_counts'] or
            actual['pretrained_sha256'] != reference['pretrained_sha256'] or actual['dataset_frames'] != 51077
            or actual['objective_routing']['mode'] != 'joint'):
        raise ValueError('Pretrained weights/counts/dataset/objective mismatch')
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):
        config=json.loads((root/'000003'/name).read_text())
        if config != json.loads((processors/name).read_text()):raise ValueError('Processor config changed')
        for item in config['steps']:
            if 'state_file' not in item:continue
            first=load_file(str(root/'000003'/item['state_file']))
            second=load_file(str(processors/item['state_file']))
            if first.keys()!=second.keys():raise ValueError('Normalizer tensor keys changed')
            for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','state'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();queue=yaml.safe_load(args.config.read_text());directory=args.config.parent
    reference=yaml.safe_load((directory/queue['reference_config']).read_text())
    studies=[]
    if len(queue['runs'])!=2:raise ValueError('Exactly two query-count ablations required')
    for item,queries in zip(queue['runs'],(32,64),strict=True):
        path,evpath=directory/item['training_config'],directory/item['eval_config']
        recipe,evaluation=yaml.safe_load(path.read_text()),yaml.safe_load(evpath.read_text())
        validate_recipe(recipe,reference,queries);validate_evaluation(recipe['training'],evaluation)
        studies.append((path,recipe,evpath,evaluation))
    for key in ('wandb_run_id','output_dir','wandb_group','run_name'):
        if len({recipe['training'][key] for _,recipe,_,_ in studies})!=2:raise ValueError('Overlapping study identities')
    if args.validate_only:
        print(json.dumps({'valid':True,'time_lag_queries':[32,64],'memory_read_queries':4,
                          'steps_each':100000,'objectives':['joint','joint']}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for _,recipe,_,evaluation in studies:
            for path in (recipe['training']['output_dir'],recipe['training']['output_dir']+'_smoke',
                         evaluation['launcher']['output_root'],evaluation['launcher']['output_root']+'_smoke'):
                if Path(path).exists():raise FileExistsError(path)
        state={'queue_pid':os.getpid(),'source_root':str(args.root),'completed':[],
               'run_ids':[recipe['training']['wandb_run_id'] for _,recipe,_,_ in studies],
               'predecessor_state':queue['predecessor_state'],'time_lag_queries':[32,64],
               'memory_read_queries':4,'evaluation_scheduled':True,'instance_action':'none'}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            tmp=args.state.with_suffix('.partial');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(args.state)
            print(json.dumps(state),flush=True)
        def run(command,**kwargs):subprocess.run(command,cwd=args.root,check=True,**kwargs)
        def idle():
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(queue['poll_seconds'])
        try:
            record('waiting_for_waitatgoal_train_and_evals')
            while not predecessor_complete(json.loads(Path(queue['predecessor_state']).read_text()),
                    queue['predecessor_run_ids'],queue['predecessor_results']):
                time.sleep(queue['poll_seconds'])
            idle()
            if shutil.disk_usage('/data').free < 80*2**30:raise RuntimeError('Require 80GiB free; no deletion performed')
            resolved=json.loads(Path(queue['reference_resolved_config']).read_text())
            if resolved['training'] != reference['training']:raise ValueError('Reference resolved config mismatch')
            pinned=Path('/data/RMBench-releases/e363a647604d798fa69e/RMBench')
            for name in ('envs/_base_task.py','envs/cover_blocks.py','scripts/eval_policy_xpolicylab.py',
                         'data/_download.py','XPolicyLab/scripts/transform_lerobot_v30_format.py','XPolicyLab/utils/process_data.py'):
                if hashlib.sha256((args.root/name).read_bytes()).digest()!=hashlib.sha256((pinned/name).read_bytes()).digest():
                    raise ValueError('Pinned benchmark/conversion source changed: '+name)
            record('preparing_cover_blocks')
            preparation=directory/queue['preparation_config']
            run([sys.executable,str(args.root/'scripts/tempo/prepare_rmbench_dataset.py'),
                 '--root',str(args.root),'--config',str(preparation)],env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'})
            p=yaml.safe_load(preparation.read_text())['dataset'];receipt=json.loads(Path(p['receipt']).read_text())
            data=validate_dataset(studies[0][1]['training'],receipt)
            if (data['total_episodes'],data['total_frames'],data['fps'])!=(50,51077,15):raise ValueError('Wrong full Cover dataset')
            if receipt['max_episode_frames']>2048:raise ValueError('Full history would exceed configured limit')
            record('validating_simulator_runtime',dataset=data)
            simulator=studies[0][3]['launcher']['simulator_python']
            run([simulator,'-c',"import importlib.metadata as m; import torch,numpy,sapien,mplib,warp,curobo,envs.cover_blocks; "
                 "assert {k:m.version(k) for k in ('torch','numpy','sapien','mplib','warp-lang')} == "
                 "{'torch':'2.4.1+cu121','numpy':'1.26.4','sapien':'3.0.0b1','mplib':'0.2.1','warp-lang':'1.12.0'}; "
                 "assert torch.ones(4,device='cuda').sum().item()==4; print('Cover Blocks runtime verified')"])
            record('regression_tests')
            tests=('test_tempo','test_shared_replay','test_gated_memory','test_baseline','test_training_controls',
                   'test_wandb_metadata','test_temporal_task','test_action_horizon','test_validation','test_objectives',
                   'test_backbone_training','test_reconstruction_targets')
            run([sys.executable,'-m','unittest',*['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+name for name in tests]])
            for index,(path,recipe,evpath,evaluation) in enumerate(studies):
                idle();training=recipe['training']
                record('smoke_training',experiment=index+1,training_run_id=training['wandb_run_id'],
                       history_queries=recipe['memory']['history_queries'])
                command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path)]
                run(command+['--smoke','--smoke-batch-size',str(training['batch_size'])])
                check_smoke(recipe,resolved,Path(queue['reference_processors']))
                smoke=copy.deepcopy(evaluation);smoke['launcher']['output_root']+='_smoke'
                smoke_path=args.state.parent/'queues/cover-query-study-smokes'/f"{training['wandb_run_id']}.yaml"
                smoke_path.parent.mkdir(parents=True,exist_ok=True)
                if smoke_path.exists():raise FileExistsError(smoke_path)
                smoke_path.write_text(yaml.safe_dump(smoke,sort_keys=False))
                record('smoke_evaluation')
                run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),
                     '--config',str(smoke_path),'--checkpoint',str(Path(training['output_dir']+'_smoke')/'000003'),
                     '--episodes','2','--no-wandb'])
                record('smoke_passed',smoke_verified=True)
            for index,(path,recipe,evpath,evaluation) in enumerate(studies):
                idle();training=recipe['training']
                record('training',experiment=index+1,training_run_id=training['wandb_run_id'],
                       training_config=str(path),history_queries=recipe['memory']['history_queries'],smoke_verified=True)
                run([sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(path)])
                checkpoint=verify_checkpoint(training,100000)
                saved=json.loads((checkpoint/'config.json').read_text())
                if saved['memory']['history_queries']!=recipe['memory']['history_queries'] or saved['temporal_objective']!='joint':
                    raise ValueError('Saved final checkpoint query/objective mismatch')
                record('evaluating',checkpoint=str(checkpoint))
                run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),'--config',str(evpath)])
                results=list(Path(evaluation['launcher']['output_root']).glob('*/completed.json'))
                if len(results)!=1:raise ValueError('Missing/ambiguous final eval result')
                result=json.loads(results[0].read_text())
                if (result['training_run_id'],result['checkpoint_step'],result['episodes'],result['checkpoint'])!=(
                        training['wandb_run_id'],100000,100,str(checkpoint)):
                    raise ValueError('Wrong/incomplete final evaluation')
                state['completed'].append({'training_run_id':training['wandb_run_id'],'history_queries':recipe['memory']['history_queries'],
                                           'result':str(results[0]),'success_rate':result['success_rate']})
                record('experiment_complete')
            record('complete')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
