"""Evaluate REAL-TSE against time-aligned ground truth and save listening files."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import soundfile as sf

from backend.app.audio.wav import SAMPLE_RATE, Signal, playback_wav, read_audio
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor

EPS = 1e-12


def si_sdr(reference: Signal, estimate: Signal) -> float:
    target = reference.astype(np.float64) - np.mean(reference, dtype=np.float64)
    output = estimate.astype(np.float64) - np.mean(estimate, dtype=np.float64)
    projected = np.dot(output, target) / (np.dot(target, target) + EPS) * target
    return float(10 * np.log10(
        (np.dot(projected, projected) + EPS) / (np.sum((output - projected) ** 2) + EPS)
    ))


def energy_ratio(estimate: Signal, reference: Signal) -> float:
    return float(10 * np.log10((np.sum(estimate ** 2) + EPS) / (np.sum(reference ** 2) + EPS)))


def evaluate_case(
    extractor: Extractor, directory: Path, output: Path, cue: str,
) -> dict:
    meta = json.loads((directory / "case.json").read_text(encoding="utf-8"))
    mixture = read_audio(directory / "mixed.wav")
    enrollment = read_audio(directory / ("enroll.wav" if cue == "A" else "enroll_b.wav"), min_seconds=1)
    target, target_rate = sf.read(directory / ("target.wav" if cue == "A" else "interferer.wav"), dtype="float32")
    other, other_rate = sf.read(directory / ("interferer.wav" if cue == "A" else "target.wav"), dtype="float32")
    if any(rate != SAMPLE_RATE for rate in (target_rate, other_rate)) or not mixture.shape == target.shape == other.shape:
        raise ValueError(f"References must match the 16 kHz mono mixture in {directory}.")

    start = perf_counter()
    estimate = extractor.extract(mixture, enrollment)
    duration = perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    (output / "01-mixed.wav").write_bytes(playback_wav(mixture))
    (output / f"02-extracted-{cue.lower()}.wav").write_bytes(playback_wav(estimate))
    (output / f"03-reference-{cue.lower()}.wav").write_bytes(playback_wav(target))
    target_present = cue == "B" or meta["target_present"]
    input_score = si_sdr(target, mixture) if target_present else None
    output_score = si_sdr(target, estimate) if target_present else None
    return {
        "case": directory.name,
        "cue": cue,
        "window_overlap": round(meta["window_overlap"], 3),
        "duration_seconds": round(len(mixture) / SAMPLE_RATE, 3),
        "input_si_sdr_db": round(input_score, 2) if input_score is not None else None,
        "output_si_sdr_db": round(output_score, 2) if output_score is not None else None,
        "si_sdri_db": round(output_score - input_score, 2) if target_present else None,
        "other_si_sdr_db": round(si_sdr(other, estimate), 2) if np.any(other) else None,
        "absent_leakage_db": round(energy_ratio(estimate, mixture), 2) if not target_present else None,
        "inference_ms": round(duration * 1000, 1),
        "rtf": round(duration / (len(mixture) / SAMPLE_RATE), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cases = sorted(path.parent for path in args.cases.glob("*/case.json"))
    if not cases:
        parser.error("No case.json files found. Run the mixture generator first.")
    model = Extractor(args.model, args.device)
    model.extract(np.zeros(4096, dtype=np.float32), read_audio(cases[0] / "enroll.wav"))
    rows = []
    for case in cases:
        for cue in ("A", "B") if json.loads((case / "case.json").read_text())["condition"] == "two_speakers" else ("A",):
            row = evaluate_case(model, case, args.output / case.name, cue)
            rows.append(row)
            print(row, flush=True)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Median SI-SDRi: {median(r['si_sdri_db'] for r in rows if r['si_sdri_db'] is not None):.2f} dB")


if __name__ == "__main__":
    main()
