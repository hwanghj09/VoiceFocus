#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python}"
"$python_bin" -c 'import torch, torchaudio'
"$python_bin" -m pip install -r backend/requirements.txt -r backend/requirements-harmony.txt

mkdir -p .voicefocus/vendor .voicefocus/models
vendor=.voicefocus/vendor/unmixx
if [[ ! -d "$vendor/.git" ]]; then
  git clone https://github.com/jihoojung0106/unmixx.git "$vendor"
fi
git -C "$vendor" checkout --detach 8e750521b5942f4717656cac86a23cf0bd90dea5

weights=.voicefocus/models/singer-identity.ts
if [[ ! -f "$weights" ]]; then
  part=$(mktemp .voicefocus/models/singer-identity.XXXXXX)
  trap 'rm -f -- "$part"' EXIT
  curl --fail --location --retry 3 --output "$part" \
    'https://huggingface.co/BernardoTorres/singer-identity/resolve/55e09cdfcd733b1c01f2b94f316a0280995a909e/contrastive/model.ts'
  "$python_bin" -c 'from pathlib import Path; from sys import argv; from backend.app.models.harmony import verify_file, SINGER_SHA256; verify_file(Path(argv[1]), SINGER_SHA256)' "$part"
  mv "$part" "$weights"
fi
"$python_bin" - <<'PY'
from backend.app.models.harmony import CONFIG_SHA256, SINGER_SHA256, SOURCE, SPLITTER_SHA256, WEIGHTS, verify_file
verify_file(WEIGHTS, SINGER_SHA256)
verify_file(SOURCE / "ckpt" / "best.ckpt", SPLITTER_SHA256)
verify_file(SOURCE / "ckpt" / "conf.yml", CONFIG_SHA256)
print("VoiceFocus harmony models are ready.")
PY
