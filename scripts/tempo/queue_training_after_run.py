#!/usr/bin/env python3
"""Wait for successful training completion and an idle GPU, then train a matched policy.

Runs in screen on the training host. No simulator, download, shutdown, deletion,
checkpoint upload, or package/environment mutation is performed by this queue.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def validate_schedule_changes(target,changes):
    """Only explicitly declared LR knobs may differ in a queued schedule study."""
    allowed={'learning_rate','minimum_lr','warmup_steps','scheduler','scheduler_decay_steps'}
    if set(changes)-allowed:raise ValueError('Only LR schedule changes may be declared')
    for key,expected in changes.items():
        if target.get(key)!=expected:raise ValueError(f'Declared schedule change mismatch: {key}')
    return set(changes)


def validate_duration_changes(target,changes):
    if set(changes)-{'steps','save_interval'}:raise ValueError('Only duration/checkpoint changes may be declared')
    for key,expected in changes.items():
        if type(expected) is not int or expected<=0 or target.get(key)!=expected:
            raise ValueError(f'Declared duration change mismatch: {key}')
    return set(changes)


def validate_defaults(training,defaults):
    if training.get('task_name')!='waitatgoal':raise ValueError('WaitAtGoal defaults applied to another task')
    for key,expected in defaults['training'].items():
        if training.get(key)!=expected:raise ValueError(f'WaitAtGoal default mismatch: {key}')


def validate_holdout_changes(target,changes):
    allowed={'validation_fraction','validation_interval','validation_split_seed','validation_seed',
             'validation_batch_size','validation_num_workers'}
    if set(changes)-allowed:raise ValueError('Only validation settings may be declared')
    for key,expected in changes.items():
        if target.get(key)!=expected:raise ValueError(f'Declared validation change mismatch: {key}')
    return set(changes)


def validate_match(reference,baseline,training_changes=None,duration_changes=None,holdout_changes=None):
    allowed={'policy_variant','reference_wandb_run','wandb_run_id','wandb_group','run_name','output_dir',
             'memory_backend','gla_gate_normalizer','temporal_camera_keys','embedding_cache_dir','embedding_cache_batch_size'}
    allowed|=validate_schedule_changes(baseline,training_changes or {})
    allowed|=validate_duration_changes(baseline,duration_changes or {})
    allowed|=validate_holdout_changes(baseline,holdout_changes or {})
    differences={key:[reference.get(key),baseline.get(key)] for key in reference.keys()|baseline.keys()
                 if reference.get(key)!=baseline.get(key)}
    if set(differences)-allowed:
        raise ValueError(f'Shared training settings differ: {sorted(set(differences)-allowed)}')
    variant=baseline.get('policy_variant','tempo')
    if variant not in ('smolvla','tempo'):raise ValueError('Expected SmolVLA or TEMPO SmolVLA')
    if variant=='tempo' and baseline.get('memory_backend')!='gla':raise ValueError('Expected GLA backend')
    if baseline.get('wandb_run_id')==reference.get('wandb_run_id'):raise ValueError('Run IDs must differ')
    if baseline['output_dir']==reference['output_dir']:raise ValueError('Output paths must differ')
    if variant=='smolvla' and any(key in baseline for key in ('memory_backend','embedding_cache_dir','temporal_camera_keys')):
        raise ValueError('Baseline recipe must not configure temporal modules/cache')
    if baseline.get('wandb_log_checkpoints',False):raise ValueError('Checkpoint uploads must remain disabled')
    return differences


def validate_gla_reference(reference,target,training_changes=None,duration_changes=None,holdout_changes=None):
    """Only duration and run identity may differ from the previous GLA recipe."""
    if target.get('memory')!=reference.get('memory') or not target.get('memory'):
        raise ValueError('GLA memory settings differ from reference')
    allowed={'steps','output_dir','wandb_run_id','wandb_group','run_name','reference_wandb_run'}
    a,b=reference['training'],target['training']
    allowed|=validate_schedule_changes(b,training_changes or {})
    allowed|=validate_duration_changes(b,duration_changes or {})
    allowed|=validate_holdout_changes(b,holdout_changes or {})
    differences={key for key in a.keys()|b.keys() if a.get(key)!=b.get(key)}
    if differences-allowed:raise ValueError(f'GLA reference settings differ: {sorted(differences-allowed)}')


def verify_checkpoint(training,step):
    checkpoint=Path(training['output_dir'])/f'{step:06d}'
    for name in ('model.safetensors','optimizer.pt','config.json','tempo_run.json',
                 'policy_preprocessor.json','policy_postprocessor.json'):
        path=checkpoint/name
        if not path.is_file() or not path.stat().st_size:raise FileNotFoundError(path)
    metadata=json.loads((checkpoint/'tempo_run.json').read_text())
    if metadata['step']!=step or metadata['training']['wandb_run_id']!=training['wandb_run_id']:
        raise ValueError('Checkpoint step/run identity mismatch')
    return checkpoint


def predecessor_ready(state,run_id):
    if state.get('training_run_id')!=run_id:raise ValueError('Predecessor run ID mismatch')
    if state.get('stage') in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
        raise RuntimeError(state.get('error','Predecessor failed or was cancelled'))
    return state.get('stage') in ('awaiting_local_download','complete')


def evaluation_ready(state,training):
    """Require a successful, complete evaluation of this exact final checkpoint."""
    if state.get('training_run_id')!=training['wandb_run_id']:
        raise ValueError('Predecessor evaluation run ID mismatch')
    if state.get('stage') in ('failed','cancelled','stopped_by_user','cancelled_by_user'):
        raise RuntimeError(state.get('error','Predecessor evaluation failed or was cancelled'))
    if state.get('stage')!='complete':return False
    result=json.loads(Path(state['result']).read_text())
    checkpoint=Path(training['output_dir'])/f"{training['steps']:06d}"
    if (result.get('training_run_id')!=training['wandb_run_id'] or
        result.get('checkpoint_step')!=training['steps'] or
        result.get('episodes')!=100 or Path(result['checkpoint']).resolve()!=checkpoint.resolve()):
        raise ValueError('Predecessor evaluation result mismatch or incomplete episodes')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    recipe=yaml.safe_load(args.config.read_text())
    predecessor_path=Path(recipe['predecessor_state'])
    predecessor=json.loads(predecessor_path.read_text())
    if predecessor['training_run_id']!=recipe['predecessor_training_run_id']:
        raise ValueError('Predecessor run ID mismatch')
    reference=yaml.safe_load(Path(predecessor['training_config']).read_text())['training']
    baseline_path=(args.config.parent/recipe['training_config']).resolve()
    baseline_recipe=yaml.safe_load(baseline_path.read_text())
    baseline=baseline_recipe['training']
    defaults_path=args.config.parent/recipe['defaults_config'] if recipe.get('defaults_config') else None
    if defaults_path:validate_defaults(baseline,yaml.safe_load(defaults_path.read_text()))
    if recipe.get('require_native_warmup'):
        if baseline.get('scheduler')!='smolvla' or baseline.get('warmup_steps',0)<=0:
            raise ValueError('Native SmolVLA scheduler with positive warmup is required')
    training_changes=recipe.get('training_changes',{})
    duration_changes=recipe.get('duration_changes',{})
    holdout_changes=recipe.get('holdout_changes',{})
    native=baseline.get('policy_variant','tempo')=='smolvla'
    if native and baseline_recipe.get('memory'):raise ValueError('Native baseline must not configure TEMPO memory')
    if not native:
        gla_reference=yaml.safe_load((args.config.parent/recipe['reference_gla_config']).read_text())
        validate_gla_reference(gla_reference,baseline_recipe,training_changes,duration_changes,holdout_changes)
    differences=validate_match(reference,baseline,training_changes,duration_changes,holdout_changes)
    eval_state_path=Path(recipe['predecessor_eval_state']) if recipe.get('predecessor_eval_state') else None
    if eval_state_path:
        evaluation_ready(json.loads(eval_state_path.read_text()),reference)
    if args.validate_only:
        print(json.dumps({'matching_shared_settings':True,'intended_differences':differences},indent=2));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for output in (baseline['output_dir'],baseline['output_dir']+'_smoke'):
            if Path(output).exists():raise FileExistsError(output)
        status={'predecessor_training_run_id':recipe['predecessor_training_run_id'],
                'training_run_id':baseline['wandb_run_id'],'training_config':str(baseline_path),
                'queue_pid':os.getpid(),'intended_differences':differences}
        if eval_state_path:status['predecessor_eval_state']=str(eval_state_path)
        if defaults_path:status['defaults_config']=str(defaults_path)
        def record(stage,**extra):
            status.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            temporary=args.state.with_suffix('.partial')
            temporary.write_text(json.dumps(status,indent=2)+'\n');temporary.replace(args.state)
            print(json.dumps(status),flush=True)
        try:
            record('waiting_for_predecessor_training')
            while not predecessor_ready(json.loads(predecessor_path.read_text()),recipe['predecessor_training_run_id']):
                time.sleep(recipe.get('poll_seconds',30))
            verify_checkpoint(reference,reference['steps'])
            if eval_state_path:
                record('waiting_for_predecessor_evaluation')
                while not evaluation_ready(json.loads(eval_state_path.read_text()),reference):
                    time.sleep(recipe.get('poll_seconds',30))
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                time.sleep(recipe.get('poll_seconds',30))
            command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(baseline_path)]
            record('baseline_smoke_test' if native else 'gla_smoke_test')
            subprocess.run(command+['--smoke','--smoke-batch-size',str(baseline['batch_size'])],cwd=args.root,check=True)
            smoke=json.loads((Path(baseline['output_dir']+'_smoke')/'smoke_passed.json').read_text())
            expected_memory=None if native else [1,baseline_recipe['memory']['num_tokens'],baseline_recipe['memory']['width']]
            if not smoke['finite'] or smoke['memory_shape']!=expected_memory:
                raise RuntimeError('Policy smoke test failed or memory shape does not match recipe')
            if recipe.get('require_same_validation_split'):
                source=json.loads((Path(reference['output_dir'])/'validation_split.json').read_text())
                target=json.loads((Path(baseline['output_dir']+'_smoke')/'validation_split.json').read_text())
                if source!=target:raise ValueError('Native/TEMPO holdout or normalization provenance differs')
                record('validation_split_verified',validation_split_id=target['split_id'])
            record('training_baseline' if native else 'training_gla')
            subprocess.run(command,cwd=args.root,check=True)
            for step in range(baseline['save_interval'],baseline['steps']+1,baseline['save_interval']):
                verify_checkpoint(baseline,step)
            record('complete',checkpoint=str(Path(baseline['output_dir'])/f"{baseline['steps']:06d}"))
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
