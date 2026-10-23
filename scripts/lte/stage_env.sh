#!/usr/bin/env bash
# Copy a credential .env beside the SmolVLA Brev scripts with restrictive permissions.
set -Eeuo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 /absolute/path/to/.env" >&2; exit 2; }
SOURCE_ENV=$1
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DESTINATION_ENV="${SCRIPT_DIR}/.env"

[[ -f "${SOURCE_ENV}" && -r "${SOURCE_ENV}" ]] || {
  echo "Credential file is not a readable regular file: ${SOURCE_ENV}" >&2
  exit 1
}

install -m 600 "${SOURCE_ENV}" "${DESTINATION_ENV}"
echo "Staged credentials at ${DESTINATION_ENV} (mode 600; gitignored)."
