#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python}"
"$python_bin" -m pip install 'ai-edge-litert==2.1.6'

mkdir -p .voicefocus/models
weights=.voicefocus/models/yamnet.tflite
if [[ ! -f "$weights" ]]; then
  part=$(mktemp .voicefocus/models/yamnet.XXXXXX)
  trap 'rm -f -- "$part"' EXIT
  curl --fail --location --retry 3 --output "$part" \
    'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/android/lite-model_yamnet_classification_tflite_1.tflite'
  "$python_bin" -c 'from pathlib import Path; from sys import argv; from backend.app.models.routing import verified_weights; verified_weights(Path(argv[1]))' "$part"
  mv "$part" "$weights"
fi
"$python_bin" -c 'from backend.app.models.routing import verified_weights, WEIGHTS; verified_weights(WEIGHTS)'
echo 'VoiceFocus audio routing is ready.'
