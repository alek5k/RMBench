#!/usr/bin/env python3
"""Repeat both 12k policies at train seeds 1/2 after seed-0 training and eval.

Materializes ordinary trainer/evaluator YAML on the training host. Reuses the
pinned trainer, simulator, pretrained weights and frozen cache, with no package
installation, checkpoint uploads, deletion or instance lifecycle changes.
"""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import yaml
from queue_training_after_run import (
    evaluation_ready, predecessor_ready, validate_defaults, validate_match, verify_checkpoint,
)
from queue_temporal_suite import validate_evaluation


def build_plan(recipe, directory):
    """Pure config expansion; only train seed and identity/output metadata change."""
    defaults=yaml.safe_load((directory/recipe['defaults_config']).read_text())
    templates={}
    for kind,paths in recipe['templates'].items():
        templates[kind]={key:yaml.safe_load((directory/value).read_text()) for key,value in paths.items()}
        t=templates[kind]['training']['training']
        validate_defaults(t,defaults)
        if (t['steps'],t['save_interval'],t['seed'])!=(12000,12000,0):
            raise ValueError('Expected seed-0 12k training references')
    validate_match(templates['smolvla']['training']['training'],templates['tempo']['training']['training'])
    if recipe['evaluation_seed']!=0:raise ValueError('Keep the reference evaluation seed 0')
    expected=[('smolvla',1),('tempo',1),('smolvla',2),('tempo',2)]
    if [(r['policy'],r['seed']) for r in recipe['runs']]!=expected:
        raise ValueError('Expected exactly the seed-1 pair then seed-2 pair')
    plan=[];identities=set();outputs=set()
    for item in recipe['runs']:
        kind,seed,rid=item['policy'],item['seed'],item['run_id']
        if type(seed) is not int or not isinstance(rid,str) or not re.fullmatch(r'[a-z0-9]{8}',rid):
            raise ValueError('Explicit integer training seed and fresh run ID required')
        source=templates[kind]
        training=copy.deepcopy(source['training']);t=training['training'];original=dict(t)
        t.update(seed=seed,wandb_run_id=rid,reference_wandb_run=f"uts_robot_lab/{t['wandb_project']}/{original['wandb_run_id']}",
                 run_name=f"{t['run_name']}_seed{seed}",wandb_group=f"{t['wandb_group']}_seed{seed}",
                 output_dir=f"{t['output_dir']}_seed{seed}")
        # Mask only the explicitly requested seed while checking every other
        # shared training parameter with the existing strict comparison.
        validate_match(original,{**t,'seed':original['seed']})
        validate_defaults(t,defaults)
        if training.get('memory')!=source['training'].get('memory'):
            raise ValueError('Memory architecture changed')
        evaluation=copy.deepcopy(source['evaluation'])
        evaluation['launcher']['checkpoint']=str(Path(t['output_dir'])/'012000')
        evaluation['launcher']['output_root']+=f'_seed{seed}'
        evaluation['wandb'].update(training_run_id=rid,group=t['wandb_group'],
                                    run_name=f"{evaluation['wandb']['run_name']}_seed{seed}")
        if evaluation['evaluation']['seed']!=recipe['evaluation_seed']:
            raise ValueError('Evaluation seed must remain the same as seed-0 references')
        validate_evaluation(t,evaluation)
        if rid in identities or rid in {v['training']['training']['wandb_run_id'] for v in templates.values()}:
            raise ValueError('Duplicate/reference run ID')
        identities.add(rid)
        for output in [t['output_dir'],t['output_dir']+'_smoke',evaluation['launcher']['output_root']]:
            if output in outputs:raise ValueError('Duplicate output path')
            outputs.add(output)
        label=f'{kind}_seed{seed}'
        plan.append({'policy':kind,'seed':seed,'run_id':rid,'training':training,'evaluation':evaluation,
                     'training_path':str(Path(recipe['runtime_dir'])/(label+'_training.yaml')),
                     'evaluation_path':str(Path(recipe['runtime_dir'])/(label+'_eval.yaml')),
                     'log':str(Path(recipe['log_dir'])/(label+'.log'))})
    return plan,templates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    recipe=yaml.safe_load(args.config.read_text())
    plan,templates=build_plan(recipe,args.config.parent)
    for kind,path in recipe['reference_states'].items():
        state=json.loads(Path(path).read_text())
        expected=templates[kind]['training']
        if state['training_run_id']!=expected['training']['wandb_run_id']:
            raise ValueError('Actual reference run ID mismatch')
        if yaml.safe_load(Path(state['training_config']).read_text())!=expected:
            raise ValueError('Actual reference config differs from staged template')
    predecessor=Path(recipe['predecessor_state']);predecessor_eval=Path(recipe['predecessor_eval_state'])
    reference=templates['tempo']['training']['training']
    predecessor_ready(json.loads(predecessor.read_text()),reference['wandb_run_id'])
    evaluation_ready(json.loads(predecessor_eval.read_text()),reference)
    if args.validate_only:
        print(json.dumps({'valid':True,'steps_each':12000,'eval_seed':0,'eval_episodes_each':100,
                          'runs':[{k:r[k] for k in ('policy','seed','run_id')} for r in plan]}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for run in plan:
            for output in (run['training']['training']['output_dir'],run['training']['training']['output_dir']+'_smoke',
                           run['evaluation']['launcher']['output_root'],run['training_path'],run['evaluation_path']):
                if Path(output).exists():raise FileExistsError(output)
        status={'queue_pid':os.getpid(),'predecessor_training_run_id':reference['wandb_run_id'],
                'predecessor_state':str(predecessor),'predecessor_eval_state':str(predecessor_eval),
                'evaluation_seed':0,'evaluation_episodes':100,'instance_action':'none','completed':[],
                'plan':[{k:r[k] for k in ('policy','seed','run_id','training_path','evaluation_path','log')} for r in plan]}
        def record(stage,**extra):
            status.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temporary=args.state.with_suffix('.partial');temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(args.state)
            print(json.dumps(status),flush=True)
        try:
            for run in plan:
                for key,source in [('training_path','training'),('evaluation_path','evaluation')]:
                    path=Path(run[key]);path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('x') as output:yaml.safe_dump(run[source],output,sort_keys=False)
            record('waiting_for_predecessor_training')
            while not predecessor_ready(json.loads(predecessor.read_text()),reference['wandb_run_id']):
                time.sleep(recipe['poll_seconds'])
            verify_checkpoint(reference,12000)
            record('waiting_for_predecessor_evaluation')
            while not evaluation_ready(json.loads(predecessor_eval.read_text()),reference):
                time.sleep(recipe['poll_seconds'])
            for run in plan:
                t=run['training']['training'];current={k:run[k] for k in ('policy','seed','run_id','log')}
                record('waiting_for_idle_gpu',current=current)
                while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                    time.sleep(recipe['poll_seconds'])
                log=Path(run['log']);log.parent.mkdir(parents=True,exist_ok=True)
                with log.open('a') as output:
                    def execute(command):
                        subprocess.run(command,cwd=args.root,stdout=output,stderr=subprocess.STDOUT,check=True)
                    command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',run['training_path']]
                    record('smoke_test',current=current)
                    execute(command+['--smoke','--smoke-batch-size',str(t['batch_size'])])
                    marker=json.loads((Path(t['output_dir']+'_smoke')/'smoke_passed.json').read_text())
                    memory=None if run['policy']=='smolvla' else [1,run['training']['memory']['num_tokens'],run['training']['memory']['width']]
                    if not marker['finite'] or marker['memory_shape']!=memory:raise RuntimeError('Repeat smoke test failed')
                    record('training',current=current);execute(command)
                    verify_checkpoint(t,12000)
                    record('evaluating',current=current)
                    execute(['bash',str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'),str(args.root),run['evaluation_path']])
                result_path=Path(run['evaluation']['launcher']['output_root'])/'results.json'
                result=json.loads(result_path.read_text())
                evaluation_ready({'training_run_id':run['run_id'],'stage':'complete','result':str(result_path)},t)
                if result['seed']!=0:raise ValueError('Completed eval used a different seed')
                status['completed'].append({**current,'successes':result['successes'],
                    'success_rate':result['success_rate'],'result':str(result_path)})
                record('repeat_complete',current=current)
            record('complete',current=None)
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
