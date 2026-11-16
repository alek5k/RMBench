#!/usr/bin/env bash
# Fresh install from pinned public sources; no copied environments or model caches.
set -Eeuo pipefail
[[ $# -eq 2 ]] || { echo 'Usage: setup_cover_ablation_fresh.sh <source-root> <queue.yaml>' >&2; exit 2; }
ROOT=$(realpath "$1")
QUEUE=$(realpath "$2")
STATE=/data/RMBench-state
umask 077
mkdir -p "$STATE/logs" "$STATE/bootstrap" "$STATE/envs" "$STATE/datasets" "$STATE/assets" "$STATE/references"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$STATE/uv-cache" UV_PYTHON_INSTALL_DIR="$STATE/uv-python" UV_LINK_MODE=copy
export HF_HOME="$STATE/huggingface" HF_LEROBOT_HOME="$STATE/datasets/lerobot" SMOVLA_ROOT="$STATE/deps/smovla"
export TEMPO_SMOLVLA_ENV="$STATE/envs/tempo-smolvla" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 TMPDIR=/tmp TMP=/tmp TEMP=/tmp
unset LD_LIBRARY_PATH
set -a
source "$STATE/credentials.env"
set +a
python3 - <<'PY'
import shutil
if shutil.disk_usage('/data').free < 100*2**30:
    raise RuntimeError('Fresh runtime, data and 20 checkpoints require 100 GiB free; no deletion performed')
PY
sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get update
sudo -n env DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y python3-venv python3-pip python3-yaml ffmpeg ninja-build \
  gcc-12 g++-12 make cmake pkg-config libgl1 libegl1 libglib2.0-0 libvulkan1 rsync zstd screen unzip
# CUDA 12.1 rejects Ubuntu 24.04's default GCC 13. Keep its host compiler <=12.
export CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 CUDAHOSTCXX=/usr/bin/g++-12 MAX_JOBS=4
if ! command -v uv >/dev/null 2>&1; then
  curl -fsSL https://astral.sh/uv/0.9.7/install.sh -o "$STATE/bootstrap/uv-install.sh"
  UV_NO_MODIFY_PATH=1 bash "$STATE/bootstrap/uv-install.sh"
fi
if [[ ! -x "$HOME/miniforge3/bin/conda" ]]; then
  [[ ! -e "$HOME/miniforge3" ]] || { echo 'Existing incomplete Miniforge retained; refusing overwrite' >&2; exit 1; }
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o "$STATE/bootstrap/Miniforge3-Linux-x86_64.sh"
  bash "$STATE/bootstrap/Miniforge3-Linux-x86_64.sh" -b -p "$HOME/miniforge3"
fi
if [[ ! -e "$STATE/envs/rmbench-eval" && ! -L "$STATE/envs/rmbench-eval" ]]; then
  ln -s "$HOME/miniforge3/envs/rmbench-eval" "$STATE/envs/rmbench-eval"
fi
[[ "$(readlink -f "$STATE/envs/rmbench-eval")" == "$HOME/miniforge3/envs/rmbench-eval" ]]
TRAIN_CONFIG=$(python3 - "$QUEUE" <<'PY'
from pathlib import Path
import sys,yaml
p=Path(sys.argv[1]);print(p.parent/yaml.safe_load(p.read_text())['training_config'])
PY
)
printf 'Installing policy environment and official Cover Blocks data; log: %s\n' "$STATE/logs/cover-balanced-policy-setup.log"
bash "$ROOT/scripts/tempo/setup_smolvla_training.sh" "$ROOT" "$TRAIN_CONFIG" \
  > "$STATE/logs/cover-balanced-policy-setup.log" 2>&1 &
policy_setup_pid=$!
printf 'Installing separate pinned simulator; log: %s\n' "$STATE/logs/cover-balanced-simulator-setup.log"
bash "$ROOT/scripts/tempo/setup_smolvla_eval.sh" "$ROOT" \
  > "$STATE/logs/cover-balanced-simulator-setup.log" 2>&1 &
simulator_setup_pid=$!
set +e
wait "$policy_setup_pid"; policy_status=$?
wait "$simulator_setup_pid"; simulator_status=$?
set -e
[[ "$policy_status" -eq 0 && "$simulator_status" -eq 0 ]] || {
  printf 'Setup failed: policy=%s simulator=%s; retained component logs\n' "$policy_status" "$simulator_status" >&2; exit 1;
}
"$TEMPO_SMOLVLA_ENV/bin/python" - <<'PY'
import torch,importlib.metadata as m
assert torch.cuda.is_available() and 'L40S' in torch.cuda.get_device_name()
assert m.version('numpy')=='1.26.4' and m.version('transformers')=='4.57.1'
assert torch.__version__=='2.10.0+cu126'
print('Pinned policy CUDA environment verified',flush=True)
PY
"$STATE/envs/rmbench-eval/bin/python" - <<'PY'
import torch,numpy,sapien,mplib,curobo,warp,importlib.metadata as m
assert torch.__version__=='2.4.1+cu121'
assert {k:m.version(k) for k in ('numpy','sapien','mplib','warp-lang')} == {
 'numpy':'1.26.4','sapien':'3.0.0b1','mplib':'0.2.1','warp-lang':'1.12.0'}
assert torch.ones(4,device='cuda').sum().item()==4
print('Pinned simulator CUDA environment verified',flush=True)
PY
printf 'Fresh setup complete; starting smoke/train/eval queue.\n'
