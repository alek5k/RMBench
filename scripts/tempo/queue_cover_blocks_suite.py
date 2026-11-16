#!/usr/bin/env python3
"""Three fresh Cover Blocks joint-loss train/eval experiments, strictly sequential."""
import argparse
import copy
from datetime import datetime,timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import yaml
from queue_training_after_run import verify_checkpoint


EXPECTED=((200000,1.,False),(100000,5.,False),(100000,1.,True))


def validate_training(recipe, reference, index):
    training,old=recipe['training'],reference['training']
    allowed={'reference_wandb_run','steps','n_action_steps','temporal_objective','train_language_model',
             'output_dir','wandb_run_id','wandb_group','run_name','wandb_tags'}
    delta={key:[old.get(key),training.get(key)] for key in old.keys()|training.keys() if old.get(key)!=training.get(key)}
    if set(delta)-allowed:raise ValueError(f'Unexpected reference drift: {set(delta)-allowed}')
    steps,weight,llm=EXPECTED[index]
    if (training['task_name'],training['steps'],training['chunk_size'],training['n_action_steps'],
        training['batch_size'],training['seed'])!=('cover_blocks',steps,50,50,8,0):
        raise ValueError('Wrong task/duration/horizon/batch/seed')
    if training.get('temporal_objective')!='joint':raise ValueError('All new runs must use joint loss, never reconstruction_only')
    if training.get('train_language_model') is not llm:raise ValueError('Wrong language-model training mode')
    expected_memory={**reference['memory'],'history_loss_weight':weight}
    if recipe['memory']!=expected_memory:raise ValueError('Unexpected memory architecture/loss drift')
    if training['reference_wandb_run']!='uts_robot_lab/RMBench/9b4a762e':raise ValueError('Wrong reference identity')
    if training.get('wandb_log_checkpoints',False) or training.get('validation_fraction',0):
        raise ValueError('No checkpoint uploads or holdout')
    for key in ('output_dir','wandb_run_id','wandb_group'):
        if training[key]==old[key]:raise ValueError('Fresh identity required: '+key)
    return delta


def validate_evaluation(training,evaluation):
    old=evaluation['evaluation']
    expected={'task':'cover_blocks','env_cfg':'aloha_agilex','action_type':'joint','episodes':100,
              'num_workers':4,'seed':0,'task_config':'demo_clean','instruction_type':'seen',
              'expert_check':True,'frequency':30,'max_seed_attempts':5000}
    if old!=expected:raise ValueError('Evaluation settings changed')
    wb=evaluation['wandb'];launch=evaluation['launcher']
    if (not wb['enabled'] or wb['training_run_id']!=training['wandb_run_id'] or
        wb['group']!=training['wandb_group'] or wb['run_name']!=training['run_name']+'_eval' or
        Path(launch['checkpoint'])!=Path(training['output_dir'])/f"{training['steps']:06d}" or
        launch['policy_name']!='TEMPO_SmolVLA'):
        raise ValueError('Evaluation identity/checkpoint mismatch')


def predecessor_complete(state,run_id,checkpoint):
    if state.get('training_run_id')!=run_id:raise ValueError('Predecessor identity changed')
    if state.get('stage') in ('failed','cancelled','stopped_by_user'):raise RuntimeError('Predecessor evaluation failed')
    if state.get('stage')!='complete':return False
    result=json.loads(Path(state['result']).read_text())
    if (result.get('training_run_id'),result.get('checkpoint_step'),result.get('episodes'),result.get('checkpoint'))!=(
            run_id,100000,100,str(checkpoint)):
        raise ValueError('Incomplete/wrong predecessor result')
    return True


def checkpoint_budget(recipes,reference_checkpoint,reference_counts):
    size=sum(p.stat().st_size for p in reference_checkpoint.iterdir() if p.is_file())
    # Conservative FP32 Adam moments for the ENTIRE old frozen VLM, although only
    # its text model is unfrozen. Includes one complete smoke checkpoint per run.
    total=10*2**30
    for recipe in recipes:
        t=recipe['training'];extra=8*reference_counts['frozen_vlm'] if t['train_language_model'] else 0
        total+=(math.ceil(t['steps']/t['save_interval'])+1)*(size+extra)
    return total


def check_smoke(training,reference,memory):
    from safetensors.torch import load_file
    import torch
    root=Path(training['output_dir']+'_smoke')
    smoke=json.loads((root/'smoke_passed.json').read_text())
    if not smoke['finite'] or smoke['actions_shape']!=[50,14] or smoke['memory_shape']!=[1,4,64]:
        raise ValueError('Train/save/reload/inference smoke failed')
    actual=json.loads((root/'resolved_config.json').read_text())
    expected=copy.deepcopy(reference['policy'])
    expected.update(n_action_steps=50,temporal_objective='joint',train_expert_only=not training['train_language_model'],freeze_vision_encoder=True)
    expected['memory']=copy.deepcopy(memory)
    if actual['policy']!=expected:raise ValueError('Resolved model/reference mismatch')
    if actual['objective_routing']['mode']!='joint':raise ValueError('Wrong loss routing in actual smoke')
    if actual['pretrained_sha256']!=reference['pretrained_sha256'] or actual['dataset_frames']!=51077:
        raise ValueError('Wrong pretrained weights/dataset')
    counts=actual['parameter_counts'];old=reference['parameter_counts']
    if not training['train_language_model']:
        if counts!=old:raise ValueError('Frozen-backbone parameter counts changed')
    elif not (counts['trainable_vlm']>0 and counts['frozen_vlm']+counts['trainable_vlm']==old['frozen_vlm'] and
              counts['trainable_expert']==old['trainable_expert'] and counts['trainable_total']==old['trainable_total']+counts['trainable_vlm']):
        raise ValueError('LLM unfreezing parameter counts inconsistent')
    original=Path(reference['training']['output_dir'])/'100000'
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):
        processor=json.loads((root/'000003'/name).read_text())
        if processor!=json.loads((original/name).read_text()):raise ValueError('Saved processor settings changed')
        for step in processor['steps']:
            if 'state_file' not in step:continue
            first=load_file(str(root/'000003'/step['state_file']));second=load_file(str(original/step['state_file']))
            if first.keys()!=second.keys():raise ValueError('Normalizer keys changed')
            for key in first:torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
    return actual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','state'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args();queue=yaml.safe_load(args.config.read_text());directory=args.config.parent
    reference=yaml.safe_load((directory/queue['reference_config']).read_text())
    recipes=[yaml.safe_load((directory/item['training_config']).read_text()) for item in queue['runs']]
    if len(recipes)!=3:raise ValueError('Exactly three ordered experiments required')
    identities=[]
    for index,(recipe,item) in enumerate(zip(recipes,queue['runs'],strict=True)):
        validate_training(recipe,reference,index)
        validate_evaluation(recipe['training'],yaml.safe_load((directory/item['eval_config']).read_text()))
        identities.append(recipe['training']['wandb_run_id'])
    if len(set(identities))!=3:raise ValueError('Run IDs must be unique')
    if args.validate_only:print(json.dumps({'valid':True,'run_ids':identities,'objectives':['joint']*3}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for recipe,item in zip(recipes,queue['runs'],strict=True):
            t=recipe['training'];ev=yaml.safe_load((directory/item['eval_config']).read_text())
            for p in (Path(t['output_dir']),Path(t['output_dir']+'_smoke'),Path(ev['launcher']['output_root'])):
                if p.exists():raise FileExistsError(p)
        state={'queue_pid':__import__('os').getpid(),'source_root':str(args.root),'run_ids':identities,
               'completed':[],'instance_action':'none','evaluation_scheduled':True,'temporal_objective':'joint'}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            partial=args.state.with_suffix('.partial');partial.write_text(json.dumps(state,indent=2)+'\n');partial.replace(args.state)
            print(json.dumps(state),flush=True)
        def run(command):subprocess.run(command,cwd=args.root,check=True)
        def idle():
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(queue['poll_seconds'])
        try:
            record('waiting_for_cover_blocks_eval')
            while not predecessor_complete(json.loads(Path(queue['predecessor_state']).read_text()),
                                           queue['predecessor_training_run_id'],Path(queue['predecessor_checkpoint'])):
                time.sleep(queue['poll_seconds'])
            record('waiting_for_idle_gpu');idle()
            resolved=json.loads(Path(queue['reference_resolved_config']).read_text())
            if resolved['training']!=reference['training']:raise ValueError('Reference config drift')
            required=checkpoint_budget(recipes,Path(queue['reference_checkpoint']),resolved['parameter_counts'])
            available=shutil.disk_usage(Path(recipes[0]['training']['output_dir']).parent).free
            if available<required:raise RuntimeError(f'Insufficient suite disk: {available} available, {required} required; no deletion performed')
            record('regression_tests',disk_required_bytes=required,disk_available_bytes=available)
            tests=('test_tempo','test_shared_replay','test_gated_memory','test_baseline','test_training_controls',
                   'test_wandb_metadata','test_temporal_task','test_action_horizon','test_validation','test_objectives','test_backbone_training')
            run([sys.executable,'-m','unittest',*['XPolicyLab.policy.TEMPO_SmolVLA.tests.'+name for name in tests]])
            # Validate ALL requested modes (especially LLM unfreezing) before
            # committing to the long first production run. Smokes never log W&B.
            smoke_reports=[]
            for index,(recipe,item) in enumerate(zip(recipes,queue['runs'],strict=True)):
                t=recipe['training'];idle()
                command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(directory/item['training_config'])]
                record('smoke_test',experiment=index+1,training_run_id=t['wandb_run_id'])
                run(command+['--smoke','--smoke-batch-size',str(t['batch_size'])])
                actual=check_smoke(t,resolved,recipe['memory'])
                smoke_reports.append(actual)
                record('smoke_passed',parameter_counts=actual['parameter_counts'],smoke_verified=True)
            for index,(recipe,item) in enumerate(zip(recipes,queue['runs'],strict=True)):
                t=recipe['training'];idle()
                command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(directory/item['training_config'])]
                record('training',experiment=index+1,training_run_id=t['wandb_run_id'],
                       parameter_counts=smoke_reports[index]['parameter_counts'],smoke_verified=True)
                run(command)
                checkpoint=verify_checkpoint(t,t['steps'])
                saved=json.loads((checkpoint/'config.json').read_text())
                if saved['temporal_objective']!='joint' or (saved['chunk_size'],saved['n_action_steps'])!=(50,50):
                    raise ValueError('Final checkpoint objective/horizon drift')
                record('evaluating',checkpoint=str(checkpoint))
                run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),'--root',str(args.root),'--config',str(directory/item['eval_config'])])
                ev=yaml.safe_load((directory/item['eval_config']).read_text())
                results=list(Path(ev['launcher']['output_root']).glob('*/completed.json'))
                if len(results)!=1:raise ValueError('Missing/ambiguous eval completion')
                result=json.loads(results[0].read_text())
                if (result['training_run_id'],result['checkpoint_step'],result['episodes'],result['checkpoint'])!=(t['wandb_run_id'],t['steps'],100,str(checkpoint)):
                    raise ValueError('Evaluation incomplete/wrong checkpoint')
                state['completed'].append({'training_run_id':t['wandb_run_id'],'result':str(results[0]),'success_rate':result['success_rate']})
                record('experiment_complete')
            record('complete')
        except BaseException as error:record('failed',error=repr(error));raise


if __name__=='__main__':main()
