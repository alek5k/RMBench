#!/usr/bin/env bash
# Train a temporal LTE recipe, then evaluate its final checkpoint on the
# temporal environment(s) selected by launcher.task. Evaluations run only when
# training exits cleanly.
set -Eeuo pipefail

[[ $# -eq 3 ]] || {
  echo "Usage: $0 <RMBench-root> <lte> <training-config.yaml>" >&2
  exit 2
}
RMBENCH_ROOT=$1
MODE=$2
CONFIG_FILE=$3
[[ "${MODE}" == lte || "${MODE}" == smolvla ]] || {
  echo "Post-training temporal evaluation supports only SmolVLA or LTE SmolVLA." >&2
  exit 2
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_READER="${SCRIPT_DIR}/training_config.py"
config_get() { python3 "${CONFIG_READER}" "${CONFIG_FILE}" get "$1"; }
config_arg() {
  local key=$1
  python3 - "${CONFIG_READER}" "${CONFIG_FILE}" "${key}" <<'PY'
import importlib.util
import sys
from pathlib import Path

reader_path, config_path, key = sys.argv[1:]
spec = importlib.util.spec_from_file_location("training_config", reader_path)
assert spec and spec.loader
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)
config = reader.load_config(Path(config_path))
try:
    value = config["args"][key]
except KeyError as error:
    raise SystemExit(f"Missing trainer argument: {key}") from error
print(reader.scalar(value, key, Path(config_path)))
PY
}

[[ "$(config_get post_train_temporal_evals)" == true ]] || {
  echo "Recipe does not request post-training temporal evaluations." >&2
  exit 2
}
case "$(config_get task)" in
  waitatgoal) TEMPORAL_EVALS=(waitatgoal) ;;
  liftqa) TEMPORAL_EVALS=(liftqa) ;;
  waitatgoal_liftqa) TEMPORAL_EVALS=(waitatgoal liftqa) ;;
  *)
    echo "Post-training temporal evaluation requires launcher.task=waitatgoal, liftqa, or waitatgoal_liftqa." >&2
    exit 2
    ;;
esac
if [[ "${MODE}" == lte ]]; then
  POLICY_KIND=lte_smolvla
else
  POLICY_KIND=smolvla
fi

# Preserve the normal trainer's behaviour and only proceed after a successful
# exit. This means a failed train can never be mistaken for an evaluated run.
bash "${SCRIPT_DIR}/run_training.sh" "${RMBENCH_ROOT}" "${MODE}" "${CONFIG_FILE}"

OUTPUT_DIR="$(config_arg output_dir)"
STEPS="$(config_arg steps)"
WANDB_ENABLED="$(config_arg wandb.enable)"
WANDB_ENTITY="$(config_arg wandb.entity)"
WANDB_PROJECT="$(config_arg wandb.project)"
JOB_NAME="$(config_arg job_name)"
EXPERIMENT_ID="$(config_get experiment_id)"
[[ "${STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid final step count: ${STEPS}" >&2; exit 2; }
printf -v STEP_DIR '%06d' "${STEPS}"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/${STEP_DIR}/pretrained_model"
[[ -f "${CHECKPOINT}/model.safetensors" ]] || {
  echo "Final checkpoint is missing after successful training: ${CHECKPOINT}" >&2
  exit 1
}

TEMPORAL_ENV_ROOT="${LTE_DP_TEMPORAL_ENV_ROOT:-${HOME}/LTE-DP}"
[[ -f "${TEMPORAL_ENV_ROOT}/ltedp/envs/waitatgoal/waitatgoal_env.py" ]] || {
  echo "LTE-DP temporal environment source is missing: ${TEMPORAL_ENV_ROOT}" >&2
  exit 1
}
[[ "${WANDB_ENABLED}" == true ]] || {
  echo "Post-training temporal evaluations require wandb.enable=true." >&2
  exit 2
}
[[ -n "${WANDB_ENTITY}" && -n "${WANDB_PROJECT}" && -n "${EXPERIMENT_ID}" ]] || {
  echo "Post-training temporal evaluations require W&B entity, project, and launcher.experiment_id." >&2
  exit 2
}

CREDENTIAL_FILE="${RMBENCH_CREDENTIAL_FILE:-${RMBENCH_ROOT}/scripts/lte/.env}"
[[ -r "${CREDENTIAL_FILE}" ]] && { set -a; source "${CREDENTIAL_FILE}"; set +a; }
source "${RMBENCH_ROOT}/XPolicyLab/policy/SmolVLA/conda_init.sh"
smolvla_activate_conda "$(config_get conda_env)"
export PYTHONPATH="${RMBENCH_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_DIR="${WANDB_DIR:-${RMBENCH_STATE_ROOT:-${HOME}/RMBench-state}/wandb}"
RESULT_ROOT="${RMBENCH_STATE_ROOT:-${HOME}/RMBench-state}/eval_result/${JOB_NAME}"

run_eval() {
  local environment=$1
  local result="${RESULT_ROOT}/${environment}_${STEP_DIR}_100eps.json"
  echo "Starting post-training ${environment} evaluation from ${CHECKPOINT}"
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NUMBA_DISABLE_JIT=1 SDL_VIDEODRIVER=dummy \
    python "${SCRIPT_DIR}/eval_lte_smolvla_temporal.py" \
      --checkpoint "${CHECKPOINT}" \
      --policy-kind "${POLICY_KIND}" \
      --environment "${environment}" \
      --temporal-env-root "${TEMPORAL_ENV_ROOT}" \
      --episodes 100 \
      --output "${result}" \
      --wandb-entity "${WANDB_ENTITY}" \
      --wandb-project "${WANDB_PROJECT}" \
      --wandb-run-name "${environment}_${JOB_NAME}_eval_${STEPS}" \
      --experiment-id "${EXPERIMENT_ID}"
}

for environment in "${TEMPORAL_EVALS[@]}"; do
  run_eval "${environment}"
done
