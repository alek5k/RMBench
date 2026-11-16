#!/usr/bin/env python3
"""Persistently wait for the GLA run, evaluate its final checkpoint, then train a matched baseline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml


def validate_match(reference, baseline):
    allowed = {'policy_variant','memory_backend','gla_gate_normalizer','temporal_camera_keys',
               'embedding_cache_dir','embedding_cache_batch_size','output_dir','run_name'}
    differences = {k:[reference.get(k),baseline.get(k)] for k in reference.keys()|baseline.keys()
                   if reference.get(k)!=baseline.get(k)}
    unexpected = set(differences)-allowed
    if unexpected: raise ValueError(f'Baseline training parameters differ: {unexpected}')
    if baseline.get('policy_variant')!='smolvla': raise ValueError('Expected a native SmolVLA baseline')
    return differences


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--training-pid',required=True,type=int)
    parser.add_argument('--reference-config',required=True,type=Path)
    parser.add_argument('--baseline-config',required=True,type=Path)
    parser.add_argument('--eval-config',required=True,type=Path)
    parser.add_argument('--state',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    reference=yaml.safe_load(args.reference_config.read_text())['training']
    baseline=yaml.safe_load(args.baseline_config.read_text())['training']
    differences=validate_match(reference,baseline)
    checkpoint=Path(reference['output_dir'])/f"{reference['steps']:06d}"
    if Path(yaml.safe_load(args.eval_config.read_text())['launcher']['checkpoint']) != checkpoint:
        raise ValueError('Evaluation must use the reference run final checkpoint')
    if args.validate_only:
        print(json.dumps({'matching_parameters':True,'intended_differences':differences,'checkpoint':str(checkpoint)},indent=2));return
    if args.state.exists(): raise FileExistsError(f'Queue already exists: {args.state}')
    args.state.parent.mkdir(parents=True,exist_ok=True)
    state={'reference_training_run_id':'5lhbf4r8','training_pid':args.training_pid,
           'checkpoint':str(checkpoint),'baseline_config':str(args.baseline_config),
           'intended_differences':differences}
    def record(stage,**extra):
        state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        temporary=args.state.with_suffix('.partial')
        temporary.write_text(json.dumps(state,indent=2)+'\n'); temporary.replace(args.state)
        print(json.dumps(state),flush=True)
    def command(argv):
        subprocess.run(argv,cwd=args.root,check=True)
    try:
        record('waiting_for_gla_training')
        proc=Path(f'/proc/{args.training_pid}/stat')
        def started():
            try: return proc.read_text().split()[21]
            except FileNotFoundError: return None
        initial=started()
        while initial is not None and started()==initial:
            time.sleep(30)
        marker=checkpoint/'tempo_run.json'
        if not marker.is_file() or json.loads(marker.read_text())['step']!=reference['steps']:
            raise RuntimeError(f'Training stopped before its final checkpoint: {checkpoint}')
        for name in ('model.safetensors','config.json','optimizer.pt'):
            if not (checkpoint/name).is_file(): raise FileNotFoundError(checkpoint/name)
        record('preparing_simulator')
        command(['bash',str(args.root/'scripts/tempo/setup_smolvla_eval.sh'),str(args.root)])
        record('evaluating_gla_final_checkpoint')
        command([sys.executable,str(args.root/'scripts/tempo/evaluate_smolvla.py'),
                 '--root',str(args.root),'--config',str(args.eval_config)])
        record('baseline_smoke_test')
        command([sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train',
                 '--config',str(args.baseline_config),'--smoke'])
        record('training_baseline')
        command([sys.executable,'-m','XPolicyLab.policy.TEMPO_SmolVLA.train',
                 '--config',str(args.baseline_config)])
        record('complete')
    except BaseException as error:
        record('failed',error=repr(error))
        raise


if __name__=='__main__':main()
