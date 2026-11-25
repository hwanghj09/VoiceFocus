"""Verify and install the selected files from the official REAL-TSE release."""
import argparse
from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ARCHIVE_SHA256 = "0a7e44050b1235db209181d7cda29ac93487fbb981ec768d3062b3243c49ce98"
MODELS = ("spk_emb_100", "spk_emb_causal_100")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("model", choices=MODELS)
    args = parser.parse_args()
    with args.archive.open("rb") as archive_file:
        actual = file_digest(archive_file, "sha256").hexdigest()
    if actual != ARCHIVE_SHA256:
        parser.error("The archive SHA-256 does not match the official checkpoint bundle.")

    root = Path(__file__).resolve().parents[1] / ".voicefocus" / "models"
    destination = root / args.model
    destination.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) as archive, TemporaryDirectory(dir=root) as temporary:
        staging = Path(temporary)
        for name in ("avg_model.pt", "config.yaml"):
            with archive.open(f"{args.model}/{name}") as source, (staging / name).open("wb") as output:
                copyfileobj(source, output)
        for name in ("avg_model.pt", "config.yaml"):
            (staging / name).replace(destination / name)
    print(f"Installed {args.model} in {destination}")


if __name__ == "__main__":
    main()
