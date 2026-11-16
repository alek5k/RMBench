#!/usr/bin/env bash
set -Eeuo pipefail
[[ $# -eq 2 || $# -eq 3 ]] || { echo 'Usage: run_smolvla_remote.sh <RMBench release> <config> [eval-config]' >&2; exit 2; }
RMBENCH_ROOT="$(realpath "$1")"
CONFIG="$(realpath "$2")"
STATE_ROOT=/data/RMBench-state
ENV_ROOT="${HOME}/RMBench-state/envs/tempo-smolvla"
export PATH="${ENV_ROOT}/bin:${HOME}/.local/bin:${PATH}"
export SMOVLA_ROOT="${STATE_ROOT}/deps/smovla"
export PYTHONPATH="${RMBENCH_ROOT}:${RMBENCH_ROOT}/XPolicyLab:${SMOVLA_ROOT}/src"
export HF_HOME="${STATE_ROOT}/huggingface" HF_LEROBOT_HOME="${STATE_ROOT}/datasets/lerobot"
export WANDB_DIR="${STATE_ROOT}/wandb" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
set -a
source "${STATE_ROOT}/credentials.env"
set +a
cd "${RMBENCH_ROOT}"
"${ENV_ROOT}/bin/python" -m unittest XPolicyLab.policy.TEMPO_SmolVLA.tests.test_tempo XPolicyLab.policy.TEMPO_SmolVLA.tests.test_shared_replay XPolicyLab.policy.TEMPO_SmolVLA.tests.test_gated_memory XPolicyLab.policy.TEMPO_SmolVLA.tests.test_baseline XPolicyLab.policy.TEMPO_SmolVLA.tests.test_training_controls XPolicyLab.policy.TEMPO_SmolVLA.tests.test_wandb_metadata XPolicyLab.policy.TEMPO_SmolVLA.tests.test_temporal_task -v
if [[ $# -eq 3 ]]; then
    EVAL_CONFIG="$(realpath "$3")"
    RUN_ID="$("${ENV_ROOT}/bin/python" - "${CONFIG}" <<'PY'
import sys,yaml
value=yaml.safe_load(open(sys.argv[1]))['training']['wandb_run_id']
assert isinstance(value,str) and value.isalnum(), 'Explicit alphanumeric wandb_run_id required'
print(value)
PY
)"
    exec "${ENV_ROOT}/bin/python" "${RMBENCH_ROOT}/scripts/tempo/run_training_eval.py" \
        --root "${RMBENCH_ROOT}" --training-config "${CONFIG}" --eval-config "${EVAL_CONFIG}" \
        --state "${STATE_ROOT}/smolvla-train-eval-${RUN_ID}.json" --smoke-eval-episodes 2
fi
"${ENV_ROOT}/bin/python" -m XPolicyLab.policy.TEMPO_SmolVLA.preflight --config "${CONFIG}"
"${ENV_ROOT}/bin/python" -m XPolicyLab.policy.TEMPO_SmolVLA.benchmark_shared --config "${CONFIG}"
"${ENV_ROOT}/bin/python" -m XPolicyLab.policy.TEMPO_SmolVLA.train --config "${CONFIG}" --smoke
exec "${ENV_ROOT}/bin/python" -m XPolicyLab.policy.TEMPO_SmolVLA.train --config "${CONFIG}"
