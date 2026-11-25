#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${PYTHON:-python}"
case "${1:-spk_emb_100}" in
  spk_emb_100|spk_emb_causal_100) model="${1:-spk_emb_100}" ;;
  *) echo "Usage: $0 [spk_emb_100|spk_emb_causal_100]" >&2; exit 2 ;;
esac

"$python_bin" -c 'import torch, torchaudio; assert torch.__version__.split("+")[0] == torchaudio.__version__.split("+")[0] == "2.7.1"'
"$python_bin" -m pip install -r backend/requirements.txt -r backend/requirements-model.txt
PYTHON="$python_bin" bash scripts/setup_routing.sh
# REAL-TSE's wheel omits namespace packages used by inference.
checkout() {
  local name="$1" url="$2" revision="$3" directory
  directory=".voicefocus/vendor/$name"
  if [[ ! -d "$directory/.git" ]]; then
    mkdir -p .voicefocus/vendor
    git clone "$url" "$directory"
  fi
  git -C "$directory" checkout --detach "$revision"
}
checkout wesep-real-tse https://github.com/REAL-TSE/wesep-real-tse.git d9ed77d788e9bc8ca08d15854defe404e4593773
checkout wespeaker https://github.com/wenet-e2e/wespeaker.git 8f53b6485d9f88a207bd17e7f8dba899495ec794

if [[ -n "${VOICEFOCUS_CHECKPOINT_ARCHIVE:-}" ]]; then
  archive="$VOICEFOCUS_CHECKPOINT_ARCHIVE"
else
  mkdir -p .voicefocus
  archive=.voicefocus/real-tse-ckpt.zip
  if [[ ! -f "$archive" ]]; then
    part=$(mktemp .voicefocus/real-tse-ckpt.XXXXXX)
    trap 'rm -f -- "$part"' EXIT
    curl --fail --location --retry 3 --output "$part" \
      'https://drive.usercontent.google.com/download?id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC&export=download&confirm=t'
    "$python_bin" scripts/install_checkpoint.py "$part" "$model"
    mv "$part" "$archive"
    exit 0
  fi
fi
"$python_bin" scripts/install_checkpoint.py "$archive" "$model"
