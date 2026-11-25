"""Measure the harmony mode on the published Choral Singing Dataset."""
import argparse
import csv
import importlib
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from backend.app.audio.wav import SAMPLE_RATE, playback_wav, write_analysis_wav
from backend.app.models.harmony import HarmonyExtractor

si_sdr = importlib.import_module("experiments.03_tse_baseline.evaluate").si_sdr
EXCERPT_SECONDS = 6
# Mix and enrollment offsets are in seconds from the public individual WAVs.
CASES = (
    ("locus-iste", ("LI_soprano_1", "LI_alto_1"), (21.25, 173.25), (150.25, 69.75)),
    ("el-rossinyol", ("ER_tenor_1", "ER_bass_1"), (117.5, 20.0), (126.25, 68.75)),
    ("nino-dios", ("ND_soprano_2", "ND_tenor_2"), (30.75, 95.0), (55.0, 18.0)),
)


def excerpt(corpus: Path, track: str, start: float) -> np.ndarray:
    with sf.SoundFile(corpus / f"CSD_{track}.wav") as audio:
        rate = audio.samplerate
        audio.seek(round(start * rate))
        data = audio.read(rate * EXCERPT_SECONDS, dtype="float32", always_2d=True).mean(axis=1)
    if len(data) != rate * EXCERPT_SECONDS:
        raise ValueError(f"Insufficient audio in {track} at {start} seconds.")
    divisor = gcd(rate, SAMPLE_RATE)
    return resample_poly(data - data.mean(), SAMPLE_RATE // divisor, rate // divisor)


def evaluate(model: HarmonyExtractor, corpus: Path, output: Path, segments: int) -> list[dict]:
    rows = []
    for song, tracks, enroll_starts, mix_starts in CASES:
        cues = [excerpt(corpus, track, start) for track, start in zip(tracks, enroll_starts, strict=True)]
        for segment_index, mix_start in enumerate(mix_starts[:segments]):
            references = [excerpt(corpus, track, mix_start) for track in tracks]
            references = np.stack([clip * (0.075 / np.sqrt(np.mean(clip ** 2))) for clip in references])
            references *= 0.95 / max(0.95, float(np.max(np.abs(references.sum(axis=0)))))
            mixture = references.sum(axis=0)
            stems, uncertain = model.split(mixture)
            case = output / f"{song}-{segment_index + 1}"
            case.mkdir(parents=True, exist_ok=True)
            write_analysis_wav(case / "mixed.wav", mixture)
            source_dimensions = references.astype(np.float64).T

            for index, (track, enrollment) in enumerate(zip(tracks, cues, strict=True)):
                target, residual, uncertain_match = model.identify(stems, enrollment, uncertain)
                np.testing.assert_allclose(target + residual, mixture, atol=3e-7, rtol=0)
                reference, retained = references[index], references[1 - index]
                target_gain, other_gain = np.linalg.lstsq(
                    source_dimensions, residual - residual.mean(), rcond=None,
                )[0][[index, 1 - index]]
                label = "ab"[index]
                write_analysis_wav(case / ("enroll.wav" if index == 0 else "enroll_b.wav"), enrollment)
                (case / f"reference-{label}.wav").write_bytes(playback_wav(reference))
                (case / f"keep-{label}.wav").write_bytes(playback_wav(target))
                (case / f"remove-{label}.wav").write_bytes(playback_wav(residual))
                row = {
                    "song": song, "segment": segment_index + 1, "track": track,
                    "mixture_start_s": mix_start, "enrollment_start_s": enroll_starts[index],
                    "target_si_sdri_db": round(si_sdr(reference, target) - si_sdr(reference, mixture), 2),
                    "retained_si_sdri_db": round(si_sdr(retained, residual) - si_sdr(retained, mixture), 2),
                    "removed_target_gain_db": round(20 * np.log10(max(1e-6, abs(target_gain))), 2),
                    "retained_gain_db": round(20 * np.log10(max(1e-6, abs(other_gain))), 2),
                    "uncertain_match": bool(uncertain_match),
                }
                rows.append(row)
                print(row, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Directory containing the source CSD_*.wav tracks.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--segments", type=int, choices=(1, 2), default=2)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    rows = evaluate(HarmonyExtractor(args.device), args.corpus, args.output, args.segments)
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
