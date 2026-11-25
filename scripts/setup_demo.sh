#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -n "${PYTHON:-}" ]]; then
  python_bin="$PYTHON"
else
  python3 -m venv .venv
  python_bin=.venv/bin/python
fi
"$python_bin" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else "VoiceFocus requires Python 3.11 or 3.12.")'

export PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
"$python_bin" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchaudio==2.7.1
PYTHON="$python_bin" bash scripts/setup_model.sh spk_emb_100
PYTHON="$python_bin" bash scripts/setup_model.sh spk_emb_causal_100
CC="${CC:-cc}" PYTHON="$python_bin" bash scripts/setup_harmony.sh
npm --prefix frontend ci --no-audit --no-fund
npm --prefix frontend run build
