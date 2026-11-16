#!/usr/bin/env python3
"""Validate a same-prefix simulator copy, then run a small CUDA import probe."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import yaml
from evaluate_smolvla import logger_command


def fingerprint(root):
    result={'files':0,'bytes':0,'symlinks':0}
    for p in root.rglob('*'):
        if '__pycache__' in p.relative_to(root).parts:continue
        info=p.lstat()
        if stat.S_ISREG(info.st_mode):result['files']+=1;result['bytes']+=info.st_size
        elif stat.S_ISLNK(info.st_mode):result['symlinks']+=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('root','manifest','eval-config','receipt','state'):
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--transfer-pid',action='append',type=int,default=[])
    args=parser.parse_args();state={'ready':False,'source_root':str(args.root)}
    def record(stage,**extra):
        state.update(stage=stage,updated=datetime.now(timezone.utc).isoformat(),**extra)
        temp=args.state.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.state)
        print(json.dumps(state),flush=True)
    try:
        record('waiting_for_runtime_copy')
        deadline=time.monotonic()+7200
        for pid in args.transfer_pid:
            while Path(f'/proc/{pid}/cmdline').exists():
                try:command=Path(f'/proc/{pid}/cmdline').read_bytes()
                except FileNotFoundError:break
                if not command.startswith(b'tar\x00'):break
                if time.monotonic()>deadline:raise TimeoutError('Simulator copy exceeded two hours')
                time.sleep(5)
        record('validating_copy')
        manifest=json.loads(args.manifest.read_text())
        for path,expected in manifest['roots'].items():
            actual=fingerprint(Path(path))
            if actual!=expected:raise ValueError(f'Incomplete runtime copy {path}: {actual} != {expected}')
        for path,expected in manifest['critical_sha256'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError(f'Binary hash mismatch: {path}')
        for name in ('objects','embodiments'):
            link=args.root/'assets'/name
            if not link.exists():link.symlink_to(Path('/data/RMBench-state/assets')/name,target_is_directory=True)
        if not (args.root/'assets/objects/007_T_block').is_dir():raise FileNotFoundError('Swap T object assets')
        recipe=yaml.safe_load(args.eval_config.read_text());launch=recipe['launcher']
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1',
             'PYTHONPATH':f'{args.root}:{args.root}/XPolicyLab',
             'WANDB_ENTITY':recipe['wandb']['entity'],'WANDB_PROJECT':recipe['wandb']['project']}
        env.pop('LD_LIBRARY_PATH',None)
        record('cpu_import_checks')
        subprocess.run([launch['simulator_python'],str(args.root/'scripts/update_embodiment_config_path.py')],
                       cwd=args.root,env=env,check=True)
        code='''import json,importlib.metadata as m
import torch,numpy,sapien,mplib,warp,curobo
versions={k:m.version(k) for k in ('torch','numpy','sapien','mplib','warp-lang')}
assert versions['torch']=='2.4.1+cu121'
assert versions['numpy']=='1.26.4'
assert versions['sapien']=='3.0.0b1'
assert versions['mplib']=='0.2.1'
assert versions['warp-lang']=='1.12.0'
assert not torch.cuda.is_available()
print('Validated simulator versions:',json.dumps(versions))'''
        subprocess.run([launch['simulator_python'],'-c',code],cwd=args.root,env=env,check=True)
        # CuRobo constructs a CUDA Pose in a method's default argument at import
        # time. Test that dependency in a separate, short-lived GPU process;
        # hiding CUDA during the task import produces a false setup failure.
        record('gpu_import_check')
        gpu_code='''import torch,envs.swap_T
assert torch.cuda.is_available()
probe=torch.ones(4,device='cuda')
assert probe.sum().item()==4
torch.cuda.synchronize()
print('Swap T simulator/CuRobo CUDA import probe passed; no rollout launched.')'''
        subprocess.run([launch['simulator_python'],'-c',gpu_code],cwd=args.root,
                       env={**env,'CUDA_VISIBLE_DEVICES':'0'},check=True)
        # Validate the logger on the saved smoke checkpoint without creating a W&B run.
        smoke=Path(launch['checkpoint']).parent.with_name(Path(launch['checkpoint']).parent.name+'_smoke')/'000003'
        command=logger_command(args.root,launch,recipe['evaluation'],recipe['wandb'],smoke,args.state.parent)
        subprocess.run(command+['--check-config'],cwd=args.root,env=env,check=True)
        record('ready',ready=True,source_bundle_id=json.loads((args.root/'SOURCE_BUNDLE.json').read_text())['bundle_id'],
               verified_manifest=str(args.manifest),cuda_import_probe=True,rollout_launched=False)
        temp=args.receipt.with_suffix('.partial');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(args.receipt)
    except BaseException as error:
        record('failed',error=repr(error));raise


if __name__=='__main__':main()
