#!/usr/bin/env python3
"""Complete two interrupted seeds without discarding 98 real episode results."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import yaml

from evaluate_smolvla import logger_command
from run_cover_checkpoint_evals_local import (evaluation_recipe, prepare_source,
                                             verify_checkpoints, verify_gpu_idle)
from run_local_checkpoint_evals import completed_result, evaluation_environment

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
EPISODE = re.compile(r'cover_blocks \| episode=(\d+) \| worker=(\d+) \| seed=(\d+) \| success=(True|False)')


def episode_records(text):
    records = [dict(episode=int(a), worker=int(b), seed=int(c), success=d == 'True')
               for a,b,c,d in EPISODE.findall(ANSI.sub('', text))]
    if len({r['episode'] for r in records}) != len(records) or len({r['seed'] for r in records}) != len(records):
        raise ValueError('Duplicate episode IDs or seeds; refusing to merge')
    return records


def missing_seeds(records):
    expected = set(range(100000,100100))
    if len(records) != 98 or {r['seed'] for r in records}-expected:
        raise ValueError('Expected exactly the 98 previously completed episodes')
    missing = sorted(expected-{r['seed'] for r in records})
    if missing != [100001,100002] or sorted(set(range(100))-{r['episode'] for r in records}) != [2,3]:
        raise ValueError('Interrupted seed/episode identities changed')
    return missing


def merge_records(previous, recovered):
    missing = missing_seeds(previous)
    if sorted(r['seed'] for r in recovered) != missing:
        raise ValueError('Recovery evaluated different seeds')
    result = deepcopy(previous)
    for episode,seed in zip([2,3],missing,strict=True):
        row = next(r for r in recovered if r['seed'] == seed)
        result.append({**row,'episode':episode})
    if len({r['episode'] for r in result}) != 100 or len({r['seed'] for r in result}) != 100:
        raise ValueError('Merged evaluation is not 100 unique episodes/seeds')
    return result


def canonical_log(records):
    lines=[];successes=0
    for total,row in enumerate(records,1):
        successes += row['success']
        lines.append(f"cover_blocks | episode={row['episode']} | worker={row['worker']} | seed={row['seed']} | "
                     f"success={row['success']} | batch success: {successes}/{total} => {successes/total*100:.1f}%")
    lines.append(f'Final batch success rate: {successes}/{len(records)} = {successes/len(records)*100:.1f}%')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();repair=yaml.safe_load(args.config.read_text())
    recipe=yaml.safe_load(Path(repair['original_config']).read_text())
    output=Path(repair['output_root']);output.mkdir(parents=True,exist_ok=True)
    if (output/'state.json').exists():raise FileExistsError(output/'state.json')
    state={'stage':'starting','queue_pid':os.getpid(),'training_run_id':recipe['training_run_id'],
           'execution':'local','instance_action':'none','completed':[],'checkpoint_steps':[100000,200000]}
    def record(stage,**extra):
        state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        temporary=output/'state.partial';temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(output/'state.json');print(json.dumps(state),flush=True)
    try:
        with (Path('/tmp')/f"tempo-local-{recipe['cuda_visible_devices']}.lock").open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            verify_gpu_idle(recipe['cuda_visible_devices']);verify_checkpoints(recipe)
            original=Path(repair['original_simulator_log'])
            previous=episode_records(original.read_text(errors='replace'));missing=missing_seeds(previous)
            record('preparing_source',preserved_episodes=len(previous),missing_seeds=missing)
            recipe.update(output_root=str(output),evaluation_source=repair['evaluation_source'])
            root,count=prepare_source(recipe)
            repo=Path(__file__).resolve().parents[2]
            # Model, vision, normalization and simulator environment files stay
            # pinned. Only the benchmark transport/explicit-seed entry is updated.
            original_script=root/'scripts/eval_policy_xpolicylab.py'
            old_hash=hashlib.sha256(original_script.read_bytes()).hexdigest()
            shutil.copy2(repo/'scripts/eval_policy_xpolicylab.py',original_script)
            provenance={'original_log':str(original),'original_log_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),
                        'preserved_episodes':previous,'missing_seeds':missing,'source_files_verified_before_patch':count,
                        'runtime_patch':{'file':'scripts/eval_policy_xpolicylab.py','before_sha256':old_hash,
                                         'after_sha256':hashlib.sha256(original_script.read_bytes()).hexdigest()}}
            (output/'recovery_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            recipe['evaluation']['num_workers']=repair['num_workers']
            recipe['launcher'].update(websocket=repair['websocket'],progress_timeout_seconds=repair['progress_timeout_seconds'],
                                      wandb_logger_script=str(repo/'scripts/lte/log_eval_wandb.py'))
            environment=evaluation_environment(recipe)
            environment.update(TMPDIR='/tmp',TMP='/tmp',TEMP='/tmp',PYTHONDONTWRITEBYTECODE='1')
            (output/'configs').mkdir(exist_ok=True)
            launcher=Path(__file__).with_name('evaluate_smolvla.py')
            def evaluate(settings,label):
                config=output/'configs'/f'{label}.yaml';config.write_text(yaml.safe_dump(settings,sort_keys=False))
                with (output/f'{label}.log').open('w') as log:
                    subprocess.run([sys.executable,str(launcher),'--root',str(root),'--config',str(config)],cwd=root,
                                   env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
                return completed_result(Path(settings['launcher']['output_root']),recipe['training_run_id'],
                                        int(Path(settings['launcher']['checkpoint']).name),settings['evaluation']['episodes'])
            settings=evaluation_recipe(recipe,100000)
            settings['evaluation'].update(episodes=2,num_workers=1,seed_start=missing[0],max_seed_attempts=2)
            settings['wandb']['enabled']=False
            settings['launcher']['output_root']=str(output/'missing_100k_seeds')
            record('recovering_missing_episodes',checkpoint_step=100000)
            recovery=evaluate(settings,'missing_100k_seeds')
            recovered=episode_records((Path(recovery['output'])/'simulator.log').read_text(errors='replace'))
            merged=merge_records(previous,recovered)
            merged_dir=output/'100000/merged';merged_dir.mkdir(parents=True)
            (merged_dir/'simulator.log').write_text(canonical_log(merged))
            (merged_dir/'episodes.json').write_text(json.dumps(merged,indent=2)+'\n')
            merged_settings=evaluation_recipe(recipe,100000)
            merged_settings['wandb']['resume_run_id']=repair['original_wandb_run_id']
            command=logger_command(root,merged_settings['launcher'],merged_settings['evaluation'],merged_settings['wandb'],
                                   Path(merged_settings['launcher']['checkpoint']),merged_dir)
            logger_env={**environment,'PYTHONPATH':f'{root}:{root}/XPolicyLab',
                        'WANDB_ENTITY':recipe['wandb']['entity'],'WANDB_PROJECT':recipe['wandb']['project']}
            record('logging_recovered_100k',checkpoint_step=100000)
            with (merged_dir/'wandb.log').open('w') as log:
                subprocess.run(command+['--once'],cwd=root,env=logger_env,stdout=log,stderr=subprocess.STDOUT,check=True)
            result={'checkpoint':merged_settings['launcher']['checkpoint'],'checkpoint_step':100000,'episodes':100,
                    'successes':sum(r['success'] for r in merged),'success_rate':sum(r['success'] for r in merged)/100,
                    'training_run_id':recipe['training_run_id'],'output':str(merged_dir),
                    'recovered':True,'preserved_episodes':98,'recovery_provenance':str(output/'recovery_provenance.json')}
            (merged_dir/'completed.json').write_text(json.dumps(result,indent=2)+'\n')
            state['completed'].append(result);record('checkpoint_complete',checkpoint_step=100000)
            record('evaluating',checkpoint_step=200000)
            result=evaluate(evaluation_recipe(recipe,200000),'200000')
            state['completed'].append(result);record('complete',checkpoint_step=200000)
    except BaseException as error:
        record('failed',error=repr(error));raise


if __name__=='__main__':main()
