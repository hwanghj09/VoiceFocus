"""Compare one- and two-reference matching on unison and imbalanced singing."""
import argparse
import csv
import importlib
from pathlib import Path

import numpy as np

from backend.app.audio.wav import playback_wav, write_analysis_wav
from backend.app.models.harmony import HarmonyExtractor

base = importlib.import_module("experiments.05_singing.evaluate")
# Tracks, mixture start, two enrollment starts, and A-to-B RMS difference in dB.
CASES = (
    ("locus-unison", ("LI_soprano_2", "LI_soprano_3"), 153.0, (173.0, 100.25), 0),
    ("rossinyol-unison", ("ER_tenor_2", "ER_tenor_4"), 125.75, (38.5, 15.25), 0),
    ("nino-unison", ("ND_alto_1", "ND_alto_3"), 33.75, (95.0, 95.0), 0),
    ("locus-unequal", ("LI_soprano_2", "LI_alto_1"), 150.25, (173.0, 173.25), 9),
    ("rossinyol-unequal", ("ER_tenor_2", "ER_bass_1"), 15.0, (126.0, 83.5), -9),
    ("nino-unequal", ("ND_soprano_2", "ND_alto_1"), 55.0, (30.75, 95.0), 9),
)


def prepare(corpus, tracks, start, enrollment_starts, imbalance_db):
    sources = np.stack([base.excerpt(corpus, track, start) for track in tracks])
    sources *= 0.075 / np.sqrt(np.mean(sources ** 2, axis=1, keepdims=True))
    sources[1] *= 10 ** (-imbalance_db / 20)
    sources /= max(1.0, float(np.max(np.abs(sources.sum(axis=0)))) / 0.9)
    enrollment = [base.excerpt(corpus, track, offset) for track, offset in zip(tracks, enrollment_starts, strict=True)]
    return sources, enrollment, sources.sum(axis=0)


def evaluate_case(model, destination, name, tracks, imbalance_db, sources, enrollment, mixture, stems, uncertain):
    output = destination / name
    output.mkdir(parents=True, exist_ok=True)
    write_analysis_wav(output / "mixed.wav", mixture)
    rows = []
    for index, track in enumerate(tracks):
        label = "ab"[index]
        reference, retained = sources[index], sources[1 - index]
        write_analysis_wav(output / f"enroll-{label}.wav", enrollment[index])
        (output / f"reference-{label}.wav").write_bytes(playback_wav(reference))
        before = base.si_sdr(retained, mixture)
        for scheme, other in (("one", None), ("two", enrollment[1 - index])):
            target, residual, ambiguous = model.identify(stems, enrollment[index], uncertain, other)
            np.testing.assert_allclose(target + residual, mixture, atol=3e-7, rtol=0)
            (output / f"remove-{label}-{scheme}.wav").write_bytes(playback_wav(residual))
            row = {
                "case": name, "target": track, "a_to_b_db": imbalance_db, "reference_count": 1 if other is None else 2,
                "assigned_target": base.si_sdr(reference, target) > base.si_sdr(retained, target),
                "retained_si_sdr_db": round(base.si_sdr(retained, residual), 2),
                "retained_si_sdri_db": round(base.si_sdr(retained, residual) - before, 2),
                "uncertain_match": bool(ambiguous),
            }
            print(row, flush=True)
            rows.append(row)
    return rows


def write_metrics(destination, rows):
    with (destination / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory containing the CSD individual tracks.")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model = HarmonyExtractor()
    rows = []
    for name, tracks, start, cue, imbalance in CASES:
        sources, enrollment, mixture = prepare(args.corpus, tracks, start, cue, imbalance)
        stems, uncertain = model.split(mixture)
        rows.extend(evaluate_case(model, args.output, name, tracks, imbalance, sources, enrollment, mixture, stems, uncertain))
    write_metrics(args.output, rows)


if __name__ == "__main__":
    main()
