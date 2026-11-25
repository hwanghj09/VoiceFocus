"""Compare cosine similarity of enrollment and independent speaker utterances."""
import argparse
import csv
from pathlib import Path
from statistics import median

import numpy as np

from backend.app.audio.wav import read_audio
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    extractor = Extractor(args.model)
    embeddings = {}
    for speaker in ("a", "b"):
        for index in range(6):
            name = f"{speaker}-{index}"
            vector = extractor.embedding(read_audio(args.source / f"{name}.wav", min_seconds=1))
            embeddings[name] = vector / np.linalg.norm(vector)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    by_match = {True: [], False: []}
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("enrollment", "utterance", "same_speaker", "cosine_similarity"))
        for enrolled in ("a", "b"):
            for utterance in ("a", "b"):
                for index in range(1, 6):
                    a, b = f"{enrolled}-0", f"{utterance}-{index}"
                    same = enrolled == utterance
                    similarity = float(embeddings[a] @ embeddings[b])
                    by_match[same].append(similarity)
                    writer.writerow((a, b, same, round(similarity, 4)))
    for same, values in by_match.items():
        print(f"{'Same' if same else 'Different'} speaker median: {median(values):.4f}")


if __name__ == "__main__":
    main()
