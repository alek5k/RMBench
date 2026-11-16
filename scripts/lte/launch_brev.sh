#!/usr/bin/env bash
# Build and upload one immutable RMBench/XPolicyLab source bundle to Brev.
set -Eeuo pipefail

[[ $# -eq 3 ]] || { echo "Usage: $0 <brev-instance> <smolvla|lte> <training-config.yaml>" >&2; exit 2; }
INSTANCE=$1
MODE=$2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RMBENCH_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE=$3
[[ "${CONFIG_FILE}" == /* ]] || CONFIG_FILE="$(pwd)/${CONFIG_FILE}"
CONFIG_FILE="$(realpath "${CONFIG_FILE}")"
CREDENTIAL_FILE="${SCRIPT_DIR}/.env"
CONFIG_READER="${SCRIPT_DIR}/training_config.py"
XPOLICYLAB_ROOT="${XPOLICYLAB_ROOT:-${RMBENCH_ROOT}/XPolicyLab}"
if [[ ! -e "${XPOLICYLAB_ROOT}/.git" ]]; then
  XPOLICYLAB_ROOT="$(cd -- "${RMBENCH_ROOT}/.." && pwd)/XPolicyLab"
fi
case "${MODE}" in smolvla|lte) ;; *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;; esac
command -v brev >/dev/null 2>&1 || { echo "brev CLI is required" >&2; exit 1; }
command -v zstd >/dev/null 2>&1 || { echo "zstd is required" >&2; exit 1; }
[[ -r "${CONFIG_FILE}" ]] || { echo "Training config not readable: ${CONFIG_FILE}" >&2; exit 1; }
[[ -e "${XPOLICYLAB_ROOT}/.git" ]] || { echo "XPolicyLab Git worktree not found: ${XPOLICYLAB_ROOT}" >&2; exit 1; }
python3 "${CONFIG_READER}" "${CONFIG_FILE}" validate

BUILD_DIR="$(mktemp -d -p /tmp rmbench-source-bundle-XXXXXX)"
trap 'rm -rf "${BUILD_DIR}"' EXIT
BUILD_JSON="$(python3 "${SCRIPT_DIR}/build_source_bundle.py" \
  --rmbench-root "${RMBENCH_ROOT}" \
  --xpolicylab-root "${XPOLICYLAB_ROOT}" \
  --recipe "${CONFIG_FILE}" \
  --output-dir "${BUILD_DIR}")"
eval "$(python3 - "${BUILD_JSON}" <<'PY'
import json, shlex, sys
result = json.loads(sys.argv[1])
for key in ("archive", "checksum", "manifest", "bundle_id"):
    print(f"{key.upper()}={shlex.quote(result[key])}")
PY
)"

BOOTSTRAP_DIR=.lte-bootstrap
REMOTE_ARCHIVE="${BOOTSTRAP_DIR}/$(basename "${ARCHIVE}")"
REMOTE_CHECKSUM="${BOOTSTRAP_DIR}/$(basename "${CHECKSUM}")"
brev exec "${INSTANCE}" "mkdir -p '${BOOTSTRAP_DIR}' && chmod 700 '${BOOTSTRAP_DIR}'"
brev copy "${ARCHIVE}" "${INSTANCE}:${REMOTE_ARCHIVE}"
brev copy "${CHECKSUM}" "${INSTANCE}:${REMOTE_CHECKSUM}"
brev copy "${SCRIPT_DIR}/setup_remote.sh" "${INSTANCE}:${BOOTSTRAP_DIR}/setup_remote.sh"
if [[ -r "${CREDENTIAL_FILE}" ]]; then
  brev copy "${CREDENTIAL_FILE}" "${INSTANCE}:${BOOTSTRAP_DIR}/credentials.env"
fi
brev exec "${INSTANCE}" "test ! -f '${BOOTSTRAP_DIR}/credentials.env' || chmod 600 '${BOOTSTRAP_DIR}/credentials.env'; chmod 700 '${BOOTSTRAP_DIR}/setup_remote.sh'; nohup bash '${BOOTSTRAP_DIR}/setup_remote.sh' '${BOOTSTRAP_DIR}' '${MODE}' '${BUNDLE_ID}' > '${BOOTSTRAP_DIR}/setup-${BUNDLE_ID}.log' 2>&1 < /dev/null & echo 'Remote setup PID:' \$! '(log: ~/${BOOTSTRAP_DIR}/setup-${BUNDLE_ID}.log)'"
printf 'Bundle: %s\nManifest: %s\nArchive: %s\n' "${BUNDLE_ID}" "${MANIFEST}" "${ARCHIVE}"
