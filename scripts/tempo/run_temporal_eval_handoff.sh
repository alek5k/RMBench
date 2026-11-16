#!/usr/bin/env bash
# Deployment-only gate around the unmodified temporal eval launcher.
# Only the identified old supervisor waits. New smoke/evals bypass it.
set -Eeuo pipefail
exec /home/shadeform/RMBench-state/envs/tempo-smolvla/bin/python - "$@" <<'PY'
import json,os,sys,time
from pathlib import Path
marker=Path('/data/RMBench-state/temporal-eval-handoff.json')
handoff=json.loads(marker.read_text())
parent=os.getppid()
if parent==handoff['old_supervisor_pid']:
    fields=Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()
    if fields[19]!=handoff['old_supervisor_start']:raise RuntimeError('Supervisor PID reused')
    receipt=Path(handoff['barrier_receipt'])
    temporary=receipt.with_suffix('.partial')
    temporary.write_text(json.dumps({'old_supervisor_pid':parent,'barrier_pid':os.getpid()})+'\n')
    temporary.replace(receipt)
    print('Existing three evaluations deferred until matched LiftQA SmolVLA completes.',flush=True)
    state_path=Path(handoff['replacement_state'])
    while True:
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if state.get('release_evaluations'):break
            if state.get('stage')=='failed':raise RuntimeError('Replacement schedule failed; evaluations remain deferred')
        time.sleep(5)
os.execv('/bin/bash',['bash',handoff['original_eval_wrapper'],*sys.argv[1:]])
PY
