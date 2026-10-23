#!/usr/bin/env bash
# Provision RMBench's simulator client separately from the SmolVLA policy server.
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 <RMBench-root>" >&2; exit 2; }
RMBENCH_ROOT="$(cd "$1" && pwd)"
CONDA_BASE="${HOME}/miniforge3"
EVAL_ENV="rmbench-eval"
CUDA_CHANNEL="nvidia/label/cuda-12.1.1"
CUROBO_REF="v0.7.8"

[[ -x "${CONDA_BASE}/bin/conda" ]] || { echo "Miniforge is missing at ${CONDA_BASE}" >&2; exit 1; }
[[ -d "${RMBENCH_ROOT}/XPolicyLab" ]] || { echo "RMBench checkout is missing XPolicyLab" >&2; exit 1; }

source "${CONDA_BASE}/etc/profile.d/conda.sh"
if ! conda env list | awk 'NR > 2 {print $1}' | grep -Fxq "${EVAL_ENV}"; then
  conda create -y -n "${EVAL_ENV}" python=3.10 pip
fi
conda activate "${EVAL_ENV}"

# The policy server stays in `smolvla`. Keep the simulator ABI coherent with
# PyTorch CUDA 12.1 and build CuRobo against this environment's toolkit.
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.4.1' 'torchvision==0.19.1'
python -m pip install -r <(grep -vE '^(torch|torchvision)' "${RMBENCH_ROOT}/scripts/requirements.txt")
# mplib 0.2.1 is not NumPy-2 compatible.  XPolicyLab only requires NumPy
# >=1.23 and OpenCV >=4.8, so pin the compatible common intersection here.
python -m pip install 'numpy==1.26.4' 'opencv-python-headless==4.10.0.84'
# The remaining XPolicyLab transport dependencies do not conflict with the
# simulator stack.  Install them explicitly so the editable install below can
# safely use --no-deps.
python -m pip install 'websockets>=14.0' 'msgpack>=1.0.8' 'msgpack-numpy>=0.4.8'

# The versioned NVIDIA label prevents Conda from resolving a mixture of CUDA
# 12 and 13 development headers.
conda install -y --override-channels -c "${CUDA_CHANNEL}" cuda-toolkit=12.1.1
export CUDA_HOME="${CONDA_PREFIX}"
export CC=/usr/bin/g++
export CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.9"  # L40S (Ada)
export MAX_JOBS="${MAX_JOBS:-4}"

if ! command -v g++ >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  command -v sudo >/dev/null 2>&1 || { echo "g++ and ninja are required to build CuRobo" >&2; exit 1; }
  sudo -n apt-get update
  sudo -n apt-get install -y build-essential ninja-build
fi
[[ -x "${CUDA_HOME}/bin/nvcc" ]] || { echo "CUDA 12.1 nvcc is missing" >&2; exit 1; }
[[ -f "${CUDA_HOME}/include/thrust/complex.h" ]] || { echo "CUDA 12.1 Thrust headers are missing" >&2; exit 1; }

CUROBO_ROOT="${RMBENCH_ROOT}/envs/curobo"
if [[ ! -d "${CUROBO_ROOT}/.git" ]]; then
  git clone --branch "${CUROBO_REF}" --depth 1 https://github.com/NVlabs/curobo.git "${CUROBO_ROOT}"
fi
git -C "${CUROBO_ROOT}" fetch --depth 1 origin "${CUROBO_REF}"
git -C "${CUROBO_ROOT}" checkout --detach FETCH_HEAD
python -m pip install -e "${CUROBO_ROOT}" --no-build-isolation
python -m pip install 'warp-lang==1.12.0' 'setuptools==69.5.1'
# All runtime dependencies were installed and pinned above.  Avoid allowing
# this editable convenience package to replace the simulator's NumPy/OpenCV.
python -m pip install -e "${RMBENCH_ROOT}/XPolicyLab" --no-deps

# Match RMBench's upstream installer patches without touching the policy env.
SAPIEN_ROOT="$(python -m pip show sapien | awk '/^Location:/{print $2}')/sapien"
MPLIB_ROOT="$(python -m pip show mplib | awk '/^Location:/{print $2}')/mplib"
# RMBench's upstream installer makes URDF reads explicit UTF-8.  Keep the
# same narrow, idempotent fix here (rather than modifying the policy env).
sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' \
  "${SAPIEN_ROOT}/wrapper/urdf_loader.py"
sed -i -E 's/(if np.linalg.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "${MPLIB_ROOT}/planner.py"

# Assets are intentionally not part of the training bootstrap, but simulator
# evaluation requires them. Source credentials only if they were staged.
if [[ ! -f "${RMBENCH_ROOT}/assets/objects/objaverse/list.json" ]]; then
  credential_file="${RMBENCH_ROOT}/scripts/lte/.env"
  [[ -r "${credential_file}" ]] && { set -a; source "${credential_file}"; set +a; }
  (cd "${RMBENCH_ROOT}" && bash scripts/_download_assets.sh)
fi

cd "${RMBENCH_ROOT}"
python -m pip check
python - <<'PY'
import curobo
import sapien
import warp
import envs
print("RMBench evaluation environment is ready")
PY
