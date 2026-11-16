#!/usr/bin/env bash
# Minimal fresh-instance tools; model/environment setup stays in its own script.
set -Eeuo pipefail
[[ $# -eq 3 ]] || { echo 'Usage: bootstrap_rmbench_task_remote.sh <release> <training.yaml> <preparation.yaml>' >&2; exit 2; }
STAGE=$(dirname "$(realpath "$2")")
export PATH="$HOME/.local/bin:$PATH"
exec 9>/data/RMBench-state/task-bootstrap.lock
flock -n 9 || { echo 'Another task bootstrap is active' >&2; exit 1; }
if ! command -v ffmpeg >/dev/null; then
  # Some fresh images cannot reach archive.ubuntu.com over HTTP. Retain the
  # original apt config and use its official HTTPS endpoint only if verified.
  if grep -q 'http://archive.ubuntu.com/ubuntu' /etc/apt/sources.list &&
     ! curl -fsSI --max-time 8 http://archive.ubuntu.com/ubuntu/dists/jammy/InRelease >/dev/null 2>&1 &&
     curl -fsSI --max-time 8 https://archive.ubuntu.com/ubuntu/dists/jammy/InRelease >/dev/null 2>&1; then
    sudo -n cp -n /etc/apt/sources.list /etc/apt/sources.list.tempo-backup
    sudo -n sed -i 's|http://archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list
  fi
  sudo -n apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 update
  sudo -n env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 install -y ffmpeg
fi
if ! command -v uv >/dev/null; then
  curl --fail --location --retry 3 https://astral.sh/uv/install.sh | sh
fi
exec bash "$STAGE/setup_rmbench_task_remote.sh" "$@"
