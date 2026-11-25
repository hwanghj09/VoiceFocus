"""Measure target-present quality and target-absent leakage in simulated calls."""

import argparse
import csv
import importlib
import io
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt

from backend.app.audio.wav import SAMPLE_RATE, playback_wav, read_audio, write_analysis_wav
from backend.app.models.call import telephone_band, verify_speech
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor

metrics = importlib.import_module("experiments.03_tse_baseline.evaluate")
CASES = (
    ("overlap", 2, 3, -6, False, True),
    ("noise", 4, 2, 0, False, False),
    ("pauses", 3, 5, 0, True, True),
    ("absent", 2, 5, 0, False, True),
)


def rms_scale(audio, level):
    return audio * (level / max(float(np.sqrt(np.mean(audio ** 2))), 1e-6))


def mulaw_round_trip(audio):
    narrow = resample_poly(telephone_band(audio), 1, 2)
    encoded = io.BytesIO()
    sf.write(encoded, narrow, 8000, format="WAV", subtype="ULAW")
    encoded.seek(0)
    decoded, _ = sf.read(encoded, dtype="float32")
    return resample_poly(decoded, 2, 1)[:len(audio)]


def make_case(source, speakers, condition):
    name, first, second, balance, pauses, narrow = condition
    voices = [[read_audio(source / f"{speaker}-{index}.wav") for index in range(6)] for speaker in speakers]
    a, b = (voices[0][first][:6 * SAMPLE_RATE], voices[1][second][:6 * SAMPLE_RATE])
    pause = 2 * SAMPLE_RATE if pauses else 0
    frames = max(len(a), len(b)) + 2 * pause
    target = np.zeros(frames, dtype=np.float32)
    other = np.zeros_like(target)
    if name != "absent":
        target[pause:pause + len(a)] = rms_scale(a, 0.09 * 10 ** (balance / 20))
    other[pause:pause + len(b)] = rms_scale(b, 0.09)
    if pauses:
        before, after = [rms_scale(voices[1][i][:pause], 0.09) for i in (1, 2)]
        other[:len(before)] = before
        other[-len(after):] = after
    random = np.random.default_rng(514).standard_normal(frames)
    low_pass = butter(2, 900, fs=SAMPLE_RATE, output="sos")
    noise = rms_scale(sosfiltfilt(low_pass, random).astype(np.float32), 0.028 if name in ("absent", "noise") else 0.015)
    gain = max(1.0, float(np.max(np.abs(target + other + noise))) / 0.85)
    target, other, noise = target / gain, other / gain, noise / gain
    mixture = target + other + noise
    if narrow:
        mixture = mulaw_round_trip(mixture)
        target, other = telephone_band(target), telephone_band(other)
    cues = [samples[0][:5 * SAMPLE_RATE] for samples in voices]
    return mixture, (target, other), cues


def measure(model, directory, name, label, mixture, target, other, cue):
    torch.manual_seed(98)
    raw = model.extract(mixture, cue)
    start = perf_counter()
    verified = verify_speech(mixture, raw, cue, model.embedding)
    verify_ms = 1000 * (perf_counter() - start)
    blocks = len(target) // 400
    # The separate reference supplies the silent regions independently of the estimates.
    end = blocks * 400
    silent = np.repeat(np.max(np.abs(target[:end].reshape(blocks, 400)), axis=1) < 1e-5, 400)
    presence = bool(np.max(np.abs(target)) > 1e-4)
    leakage_measurable = bool(np.linalg.norm(mixture[:end][silent]) > 0.04)
    muted = not bool(np.any(verified))
    directory.mkdir(parents=True, exist_ok=True)
    write_analysis_wav(directory / "mixed.wav", mixture)
    write_analysis_wav(directory / ("enroll.wav" if label == "A" else "enroll_b.wav"), cue)
    for role, audio in (("reference", target), ("raw", raw), ("verified", verified)):
        (directory / f"{role}-{label.lower()}.wav").write_bytes(playback_wav(audio))
    return {
        "case": name, "target": label, "target_present": presence,
        "raw_si_sdr_db": round(metrics.si_sdr(target, raw), 2) if presence else None,
        "verified_si_sdr_db": round(metrics.si_sdr(target, verified), 2) if presence and not muted else None,
        "raw_other_si_sdr_db": round(metrics.si_sdr(other, raw), 2) if np.any(raw) else None,
        "verified_other_si_sdr_db": round(metrics.si_sdr(other, verified), 2) if not muted else None,
        "raw_pause_leakage_db": round(metrics.energy_ratio(raw[:end][silent], mixture[:end][silent]), 2) if leakage_measurable else None,
        "verified_pause_leakage_db": round(metrics.energy_ratio(verified[:end][silent], mixture[:end][silent]), 2) if leakage_measurable and np.any(verified[:end][silent]) else None,
        "pause_samples_zero": not bool(np.any(verified[:end][silent])) if leakage_measurable else None,
        "output_muted": muted,
        "verification_ms": round(verify_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="A directory containing six independent WAVs per test speaker.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), help="File prefixes of the two speakers.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    model = Extractor(args.model)
    rows = []
    for speakers in args.pair or [("a", "b")]:
        for case in CASES:
            name = f"{'-'.join(speakers)}-{case[0]}"
            mixture, references, cues = make_case(args.source, speakers, case)
            for target in (0,) if case[0] in ("pauses", "absent") else (0, 1):
                row = measure(model, args.output / name, name, "AB"[target], mixture,
                              references[target], references[1 - target], cues[target])
                rows.append(row)
                print(row, flush=True)
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
