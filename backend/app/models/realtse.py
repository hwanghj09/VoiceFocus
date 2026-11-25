from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = ROOT / ".voicefocus" / "models" / "spk_emb_100"
VENDOR_DIR = ROOT / ".voicefocus" / "vendor"
MODEL_ID = "real-tse-bsrnn-spk-emb"
MODEL_NAME = "REAL-TSE BSRNN"
MODEL_VERSION = "spk_emb_100"
CHECKPOINT_FILES = ("config.yaml", "avg_model.pt")


def checkpoint_exists(directory: Path) -> bool:
    return all((directory / name).is_file() for name in CHECKPOINT_FILES)
