"""Measure the verified stream with the WebSocket's PCM chunk size and paced input."""

import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch

from backend.app.audio.wav import SAMPLE_RATE, read_audio
from backend.app.live import CHUNK
from backend.app.models.live import VerifiedStream
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor

metrics = importlib.import_module("experiments.03_tse_baseline.evaluate")


def measure(extractor, mixture_path, cue_path, reference_path):
    mixture = read_audio(mixture_path)
    reference, rate = sf.read(reference_path, dtype="float32")
    if rate != SAMPLE_RATE or reference.shape != mixture.shape:
        raise ValueError("Reference must match the mixture's 16 kHz mono samples.")
    cue = read_audio(cue_path, min_seconds=1)
    torch.manual_seed(42)
    mark = perf_counter()
    stream = VerifiedStream(extractor, cue)
    enrollment_ms = 1000 * (perf_counter() - mark)
    output = []
    input_at = available_at = compute = 0.0
    emitted = 0
    delays = []
    chunk_times = []
    first_nonzero = None
    for start in range(0, len(mixture), CHUNK):
        chunk = mixture[start:start + CHUNK]
        input_at += len(chunk) / SAMPLE_RATE
        mark = perf_counter()
        block = stream.push(chunk)
        seconds = perf_counter() - mark
        compute += seconds
        chunk_times.append(seconds * 1000)
        available_at = max(available_at, input_at) + seconds
        if len(block):
            delays.append((available_at - emitted / SAMPLE_RATE) * 1000)
            if first_nonzero is None and np.any(block):
                first_nonzero = (available_at + np.flatnonzero(block)[0] / SAMPLE_RATE) * 1000
            emitted += len(block)
            output.append(block)
    mark = perf_counter()
    tail = stream.finish()
    seconds = perf_counter() - mark
    output.append(tail)
    compute += seconds
    if first_nonzero is None and np.any(tail):
        first_nonzero = (max(available_at, input_at) + seconds + np.flatnonzero(tail)[0] / SAMPLE_RATE) * 1000
    estimate = np.concatenate(output)
    if estimate.shape != mixture.shape or not np.isfinite(estimate).all():
        raise ValueError("Unexpected output samples.")
    present = bool(np.any(reference))
    muted = not bool(np.any(estimate))
    block_samples = SAMPLE_RATE // 40
    stop = len(reference) // block_samples * block_samples
    silent = np.repeat(np.max(np.abs(reference[:stop].reshape(-1, block_samples)), axis=1) < 1e-5, block_samples)
    pause_output = estimate[:stop][silent]
    pause_input = mixture[:stop][silent]
    has_measurable_pause = np.linalg.norm(pause_input) > 0.04
    return {
        "case": mixture_path.parent.name,
        "enrollment": cue_path.name,
        "seconds": round(input_at, 3),
        "rtf": round(compute / input_at, 3),
        "enrollment_ms": round(enrollment_ms, 1),
        "p95_push_ms": round(float(np.percentile(chunk_times, 95)), 1),
        "p95_block_latency_ms": round(float(np.percentile(delays, 95)), 1) if delays else None,
        "first_nonzero_ms": round(first_nonzero, 1) if first_nonzero is not None else None,
        "target_present": present,
        "all_samples_muted": muted,
        "pause_samples_zero": not bool(np.any(pause_output)) if has_measurable_pause else None,
        "pause_leakage_db": round(metrics.energy_ratio(pause_output, pause_input), 2)
        if has_measurable_pause and np.any(pause_output) else None,
        "si_sdri_db": round(metrics.si_sdr(reference, estimate) - metrics.si_sdr(reference, mixture), 2)
        if present and not muted else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case", nargs=3, action="append", required=True, type=Path, metavar=("MIX", "ENROLL", "REF"))
    args = parser.parse_args()
    extractor = Extractor(DEFAULT_MODEL_DIR.with_name("spk_emb_causal_100"))
    rows = []
    for entry in args.case:
        row = measure(extractor, *entry)
        rows.append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"chunk_samples": CHUNK, "runs": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
