#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python_bin="${PYTHON:-.venv/bin/python}"
if ! command -v -- "$python_bin" >/dev/null || [[ ! -f frontend/dist/index.html ]]; then
  echo 'Run bash scripts/setup_demo.sh before starting VoiceFocus.' >&2
  exit 1
fi

export OPENBLAS_NUM_THREADS="${VOICEFOCUS_BLAS_THREADS:-1}"
exec "$python_bin" -m uvicorn backend.app.main:app \
  --host "${VOICEFOCUS_HOST:-127.0.0.1}" --port "${VOICEFOCUS_PORT:-8000}" \
  --workers 1 --no-access-log
