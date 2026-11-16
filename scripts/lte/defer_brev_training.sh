#!/usr/bin/env bash
# Start an already-uploaded source bundle only after a named screen session has
# finished.  Used to serialize expensive evaluation and training on one GPU.
set -Eeuo pipefail

[[ $# -eq 4 ]] || {
  echo "Usage: $0 <screen-to-wait-for> <bootstrap-dir> <smolvla|lte> <bundle-id>" >&2
  exit 2
}

WAIT_SCREEN=$1
BOOTSTRAP_DIR=$2
MODE=$3
BUNDLE_ID=$4
case "${MODE}" in smolvla|lte) ;; *) echo "Invalid mode: ${MODE}" >&2; exit 2 ;; esac

screen_exists() {
  screen -ls 2>/dev/null | grep -Fq ".${1}"
}

echo "Waiting for screen session: ${WAIT_SCREEN}"
while screen_exists "${WAIT_SCREEN}"; do
  sleep 60
done

echo "Prerequisite session ended; starting ${MODE} bundle ${BUNDLE_ID}."
exec bash "${BOOTSTRAP_DIR}/setup_remote.sh" "${BOOTSTRAP_DIR}" "${MODE}" "${BUNDLE_ID}"
