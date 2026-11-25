"""Build five reproducible two-speaker mixtures with separate enrollment WAVs."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from backend.app.audio.wav import SAMPLE_RATE, Signal, read_audio, write_analysis_wav

CASES = (
    ("e01-partial", 0.55, 0),
    ("e02-full", 1.0, 0),
    ("e03-loud-interferer", 0.85, -8),
    ("e01-partial-late-target", -0.65, 3),
    ("e02-full-louder-target", 1.0, 6),
)


def rms(signal: Signal) -> float:
    return float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))


def make_case(
    destination: Path, a: Signal, b: Signal, enroll_a: Signal, enroll_b: Signal,
    overlap: float, sir_db: float, rng: np.random.Generator, condition: str = "",
) -> None:
    length = min(len(a), len(b))
    a = a[:length].copy()
    b = b[:length].copy()
    a *= 0.08 / rms(a)
    b *= (0.08 / rms(b)) * 10 ** (-sir_db / 20)
    offset = round(length * (1 - abs(overlap)))
    start_a, start_b = (offset, 0) if overlap < 0 else (0, offset)
    target = np.zeros(length + offset, dtype=np.float32)
    interferer = np.zeros_like(target)
    target[start_a:start_a + length] = a
    interferer[start_b:start_b + length] = b
    if condition == "absent":
        target.fill(0)
    if condition == "telephone":
        bandpass = butter(4, (300, 3400), btype="bandpass", fs=SAMPLE_RATE, output="sos")
        target = sosfilt(bandpass, target).astype(np.float32)
        interferer = sosfilt(bandpass, interferer).astype(np.float32)
    noise = np.zeros_like(target)
    if condition == "noise":
        noise = rng.normal(0, 0.03, size=len(target)).astype(np.float32)
    mixture = target + interferer + noise
    gain = 0.95 / max(0.95, *(float(np.max(np.abs(s))) for s in (mixture, target, interferer)))
    destination.mkdir(parents=True, exist_ok=True)
    for name, signal in {
        "target": target * gain, "interferer": interferer * gain,
        "mixed": mixture * gain, "enroll": enroll_a, "enroll_b": enroll_b,
    }.items():
        write_analysis_wav(destination / f"{name}.wav", signal)
    if condition == "noise":
        write_analysis_wav(destination / "noise.wav", noise * gain)
    (destination / "case.json").write_text(json.dumps({
        "duration_seconds": len(mixture) / SAMPLE_RATE,
        "window_overlap": 0.0 if condition == "absent" else (length - offset) / length,
        "nominal_sir_db": None if condition == "absent" else sir_db,
        "target_present": condition != "absent",
        "condition": condition or "two_speakers",
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="a-0.wav…a-5.wav and b-0.wav…b-5.wav. Index 0 is enrollment.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    enroll_a, enroll_b = (
        read_audio(args.source / f"{speaker}-0.wav", min_seconds=1) for speaker in ("a", "b")
    )
    for index, (name, overlap, sir) in enumerate(CASES, 1):
        a, b = (read_audio(args.source / f"{speaker}-{index}.wav") for speaker in ("a", "b"))
        if abs(overlap) < 1:
            overlap *= float(rng.uniform(1.0, 1.06))
            sir += float(rng.uniform(-0.5, 0.5))
        make_case(args.output / name, a, b, enroll_a, enroll_b, overlap, sir, rng)
    if args.extended:
        for name, condition in (("e05-absent", "absent"), ("e06-noise", "noise"), ("e08-band-limited", "telephone")):
            make_case(args.output / name, a, b, enroll_a, enroll_b, 1.0, 0, rng, condition)


if __name__ == "__main__":
    main()
