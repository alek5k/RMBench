#!/usr/bin/env bash
# CPU-only early preparation; GPU cache extraction remains in the train queue.
set -Eeuo pipefail
[[ $# -eq 1 ]] || { echo 'Usage: prepare_liftqa_remote.sh <release-root>' >&2; exit 2; }
ROOT=$(realpath "$1")
STATE=/data/RMBench-state
PYTHON="${HOME}/RMBench-state/envs/tempo-smolvla/bin/python"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export HF_HOME="${STATE}/huggingface" HF_LEROBOT_HOME="${STATE}/datasets/lerobot"
export SMOVLA_ROOT="${STATE}/deps/smovla" PYTHONPATH="${ROOT}:${ROOT}/XPolicyLab:${STATE}/deps/smovla/src"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "${STATE}/credentials.env"
set +a
cd "$ROOT"
"$PYTHON" scripts/lte/prepare_temporal_datasets.py --output-root "$STATE/datasets" --datasets liftqa_v2
"$PYTHON" - "$ROOT" "$STATE" <<'PY'
from datetime import datetime,timezone
import json,sys
from pathlib import Path
import numpy as np
import torch
import zarr
sys.path.insert(0,str(Path(sys.argv[1])/'scripts/tempo'))
from queue_temporal_suite import check_liftqa_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from XPolicyLab.policy.TEMPO_SmolVLA.train import select_input_features

assert not torch.cuda.is_available(), 'CPU preparation must not use the training GPU'
state=Path(sys.argv[2]);root=state/'datasets/liftqa_v2'
summary=check_liftqa_dataset(root)
source=zarr.open(str(state/'datasets/raw/lift_qa_v2.zarr'),mode='r')
ends=np.asarray(source['meta']['episode_ends'])
lengths=np.diff(np.r_[0,ends])
assert summary['total_episodes']==len(ends) and summary['total_frames']==int(ends[-1])
assert int(lengths.max())<=2048, 'Episodes must fit complete causal histories'
dataset=LeRobotDataset('local/liftqa_v2',root=root,
    delta_timestamps={'action':[i/10 for i in range(50)]},video_backend='torchcodec')
for index in (0,len(dataset)-1):
    row=dataset[index]
    assert tuple(row['action'].shape)==(50,3)
    assert tuple(row['observation.state'].shape)==(3,)
    assert tuple(row['observation.image'].shape)==(3,144,144)
    assert all(torch.isfinite(row[k]).all() for k in ('action','observation.state','observation.image'))
assert set(select_input_features(dataset_to_policy_features(dataset.meta.features)))=={
    'observation.state','observation.images.camera1'}
for key in ('action','observation.state'):
    stats=dataset.meta.stats[key]
    assert np.isfinite(stats['mean']).all() and np.isfinite(stats['std']).all()
receipt={'complete':True,'dataset':'local/liftqa_v2',**summary,'max_episode_frames':int(lengths.max()),
         'cpu_only':True,'sample_decode_passed':True,'gpu_cache_built':False,
         'updated':datetime.now(timezone.utc).isoformat()}
(state/'datasets/liftqa_v2_prepared.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
