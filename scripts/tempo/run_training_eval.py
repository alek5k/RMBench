#!/usr/bin/env python3
"""Run smoke checks, full SmolVLA/TEMPO training and final-checkpoint evaluation."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--training-config',required=True,type=Path)
    parser.add_argument('--eval-config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--skip-smoke',action='store_true',help='Use only after this exact source/config passed its smoke test.')
    parser.add_argument('--smoke-eval-episodes',type=int,default=0,
                        help='Before production training, evaluate the 3-update smoke checkpoint without W&B logging.')
    args=parser.parse_args()
    if args.smoke_eval_episodes < 0:
        parser.error('--smoke-eval-episodes must be nonnegative')
    training=yaml.safe_load(args.training_config.read_text())['training']
    evaluation=yaml.safe_load(args.eval_config.read_text())
    local_eval=evaluation['launcher'].get('execution','remote')=='local'
    checkpoint=Path(training['output_dir'])/f"{training['steps']:06d}"
    if not local_eval and Path(evaluation['launcher']['checkpoint'])!=checkpoint:
        raise ValueError('Evaluation must select this training run final checkpoint')
    if local_eval and Path(evaluation['launcher']['checkpoint']).name!=checkpoint.name:
        raise ValueError('Local evaluation must select this training run final checkpoint step')
    if evaluation['wandb']['training_run_id']!=training['wandb_run_id']:
        raise ValueError('Training and evaluation W&B run IDs differ')
    if args.state.exists():raise FileExistsError(args.state)
    state={'training_run_id':training['wandb_run_id'],'training_config':str(args.training_config),
           'eval_config':str(args.eval_config),'checkpoint':str(checkpoint),'runner_pid':os.getpid()}
    args.state.parent.mkdir(parents=True,exist_ok=True)
    def record(stage,**extra):
        state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        temp=args.state.with_suffix('.partial')
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
        print(json.dumps(state),flush=True)
    def run(command):
        subprocess.run(command,cwd=args.root,check=True)
    command=[sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train','--config',str(args.training_config)]
    try:
        if args.skip_smoke:
            marker=Path(training['output_dir']+'_smoke')/'smoke_passed.json'
            if not marker.is_file():raise FileNotFoundError(marker)
        else:
            record('smoke_test')
            run(command+['--smoke','--smoke-batch-size',str(training['batch_size'])])
        if args.smoke_eval_episodes and not local_eval:
            record('preparing_smoke_simulator')
            run(['bash',str(args.root/'scripts/tempo/setup_smolvla_eval.sh'),str(args.root)])
            record('smoke_evaluation')
            run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),
                 '--root',str(args.root),'--config',str(args.eval_config),
                 '--checkpoint',str(Path(training['output_dir']+'_smoke')/'000003'),
                 '--episodes',str(args.smoke_eval_episodes),'--no-wandb'])
        record('training')
        run(command)
        if json.loads((checkpoint/'tempo_run.json').read_text())['step']!=training['steps']:
            raise RuntimeError('Training did not produce the requested final checkpoint')
        if local_eval:
            record('awaiting_local_download',instance_action='none',evaluation_execution='local')
            return
        record('preparing_simulator')
        run(['bash',str(args.root/'scripts/tempo/setup_smolvla_eval.sh'),str(args.root)])
        record('evaluating')
        run([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),
             '--root',str(args.root),'--config',str(args.eval_config)])
        record('complete')
    except BaseException as error:
        record('failed',error=repr(error));raise


if __name__=='__main__':main()
