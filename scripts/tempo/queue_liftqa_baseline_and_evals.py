#!/usr/bin/env python3
"""Train matched LiftQA SmolVLA before releasing the existing three-eval suite.

A cooperative barrier in the old eval launcher holds only its future evals.
Neither its supervisor nor its active training child is stopped or replaced.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from queue_temporal_suite import validate_evaluation, check_liftqa_dataset
from queue_training_after_run import validate_match, verify_checkpoint


def atomic_json(path, value):
    temporary = path.with_suffix('.handoff.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def validate_recipe(directory, recipe, predecessor):
    if predecessor['training_run_id'] != recipe['predecessor_training_run_id']:
        raise ValueError('Predecessor identity mismatch')
    reference = yaml.safe_load(Path(predecessor['training_config']).read_text())['training']
    path = directory / recipe['training_config']
    baseline = yaml.safe_load(path.read_text()); training = baseline['training']
    validate_match(reference, training)
    if baseline.get('memory') or training.get('policy_variant') != 'smolvla':
        raise ValueError('Expected native baseline without memory')
    if (training['task_name'], training['steps'], training['save_interval']) != ('liftqa', 20000, 10000):
        raise ValueError('Expected LiftQA 20k with 10k local checkpoints')
    evaluations = []
    for item in recipe['evaluations']:
        source = yaml.safe_load((directory/item['training_config']).read_text())['training']
        eval_path = directory/item['eval_config']
        evaluation = yaml.safe_load(eval_path.read_text())
        validate_evaluation(source, evaluation)
        if evaluation['wandb']['group'] != source['wandb_group']:
            raise ValueError('Evaluation group mismatch')
        evaluations.append((source, eval_path, evaluation))
    expected = [('waitatgoal','smolvla'), ('waitatgoal','tempo_smolvla'),
                ('liftqa','tempo_smolvla'), ('liftqa','smolvla')]
    if [(e['evaluation']['task'],e['launcher']['policy_kind']) for _,_,e in evaluations] != expected:
        raise ValueError('Expected all four task/policy evaluations in scheduled order')
    return reference, path, training, evaluations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    recipe = yaml.safe_load(args.config.read_text())
    old_path = Path(recipe['predecessor_state'])
    predecessor = json.loads(old_path.read_text())
    reference, training_path, training, evaluations = validate_recipe(args.config.parent, recipe, predecessor)
    if args.validate_only:
        print(json.dumps({'valid':True,'training_run_id':training['wandb_run_id'],'evaluations':4}));return
    args.state.parent.mkdir(parents=True,exist_ok=True)
    with args.state.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.state.exists():raise FileExistsError(args.state)
        for output in (training['output_dir'],training['output_dir']+'_smoke'):
            if Path(output).exists():raise FileExistsError(output)
        for _,_,ev in evaluations:
            if (Path(ev['launcher']['output_root'])/'results.json').exists():
                raise FileExistsError(ev['launcher']['output_root'])
        handoff = json.loads(Path(recipe['eval_handoff']).read_text())
        if (Path(handoff['replacement_state']).resolve()!=args.state.resolve() or
            handoff['old_supervisor_pid']!=predecessor['queue_pid']):
            raise ValueError('Eval barrier identity mismatch')
        state={'training_run_id':training['wandb_run_id'],'training_config':str(training_path),
               'predecessor_training_run_id':reference['wandb_run_id'],'queue_pid':os.getpid(),
               'source_root':str(args.root),'instance_action':'none','evaluations_completed':[],
               'release_evaluations':False,'predecessor_state':str(old_path)}
        def record(stage,**extra):
            state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
            atomic_json(args.state,state);print(json.dumps(state),flush=True)
        def run(cmd):subprocess.run(cmd,cwd=args.root,check=True)
        def pause():time.sleep(recipe.get('poll_seconds',10))
        try:
            record('waiting_for_liftqa_gla_training')
            while True:
                old=json.loads(old_path.read_text())
                if old['training_run_id']!=reference['wandb_run_id']:raise ValueError('Predecessor changed')
                if old['stage'] in ('failed','cancelled','stopped_by_user'):
                    raise RuntimeError('Predecessor failed or was cancelled')
                if old['stage']=='evaluating_final_checkpoints' and Path(handoff['barrier_receipt']).exists():
                    receipt=json.loads(Path(handoff['barrier_receipt']).read_text())
                    if receipt['old_supervisor_pid']!=predecessor['queue_pid']:
                        raise ValueError('Barrier receipt supervisor mismatch')
                    break
                pause()
            # The original supervisor reaped a successful trainer and verified
            # both full checkpoints before reaching its first eval launcher.
            for step in (10000,20000):verify_checkpoint(reference,step)
            record('waiting_for_idle_gpu')
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():pause()
            check_liftqa_dataset(training['dataset_root'])
            record('preparing_simulators')
            run(['bash',str(args.root/'scripts/tempo/setup_temporal_eval_remote.sh'),str(args.root)])
            trainer=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(training_path)]
            record('baseline_smoke_training')
            run(trainer+['--smoke','--smoke-batch-size',str(training['batch_size'])])
            marker=json.loads((Path(training['output_dir']+'_smoke')/'smoke_passed.json').read_text())
            if not marker['finite'] or marker['memory_shape'] is not None:raise RuntimeError('Native baseline smoke failed')
            baseline_eval=evaluations[-1][1]
            record('baseline_smoke_evaluation')
            run(['bash',str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'),str(args.root),str(baseline_eval),
                 '--smoke-checkpoint',str(Path(training['output_dir']+'_smoke')/'000003')])
            record('training_liftqa_smolvla')
            run(trainer)
            for step in (10000,20000):verify_checkpoint(training,step)
            record('waiting_for_original_three_evaluations',release_evaluations=True)
            # Original supervisor performs the first three; never duplicate them.
            while json.loads(old_path.read_text())['stage'] not in ('complete','failed'):pause()
            failures=[]
            for index,(source,path,evaluation) in enumerate(evaluations):
                try:
                    verify_checkpoint(source,source['steps'])
                    if index==3:
                        record('evaluating_liftqa_smolvla')
                        run(['bash',str(args.root/'scripts/tempo/run_temporal_eval_remote.sh'),str(args.root),str(path)])
                    result=json.loads((Path(evaluation['launcher']['output_root'])/'results.json').read_text())
                    if result['episodes']!=100 or result['training_run_id']!=source['wandb_run_id']:
                        raise ValueError('Incomplete/mismatched evaluation')
                    state['evaluations_completed'].append({'training_run_id':source['wandb_run_id'],
                        'task':source['task_name'],'checkpoint_step':source['steps'],'success_rate':result['success_rate']})
                    record('evaluating_final_checkpoints')
                except Exception as error:
                    failures.append({'config':str(path),'error':repr(error)})
                    record('evaluating_final_checkpoints',evaluation_failures=failures)
            if failures:raise RuntimeError(f'{len(failures)} evaluations failed; all four were attempted')
            record('complete')
        except BaseException as error:
            record('failed',error=repr(error));raise


if __name__=='__main__':main()
