"""Compare stateful REAL-TSE PCM streams with the same checkpoint run offline."""
import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from backend.app.audio.wav import SAMPLE_RATE, read_audio
from backend.app.models.causal import CausalStream
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor

si_sdr = importlib.import_module("experiments.03_tse_baseline.evaluate").si_sdr
SEED = 42


def measure(extractor, mixture, enrollment, reference, offline, chunk_ms):
    chunk_size = round(chunk_ms * SAMPLE_RATE / 1000)
    torch.manual_seed(SEED)
    setup_start = perf_counter()
    stream = CausalStream(extractor, enrollment)
    enrollment_ms = (perf_counter() - setup_start) * 1000
    elapsed = []
    latency = []
    output = []
    produced = 0
    available_at = 0.0

    def process(infer, captured_samples):
        nonlocal available_at, produced
        start = perf_counter()
        block = infer()
        seconds = perf_counter() - start
        elapsed.append(seconds)
        available_at = max(available_at, captured_samples / SAMPLE_RATE) + seconds
        if len(block):
            latency.append((available_at - produced / SAMPLE_RATE) * 1000)
            output.append(block)
            produced += len(block)

    for begin in range(0, len(mixture), chunk_size):
        pcm = mixture[begin:begin + chunk_size]
        process(lambda: stream.push(pcm), begin + len(pcm))
    process(stream.finish, len(mixture))

    estimate = np.concatenate(output)
    if estimate.shape != offline.shape:
        raise RuntimeError(f"Expected {len(offline)} samples, received {len(estimate)}.")
    peak_error = float(np.max(np.abs(estimate - offline)))
    if not np.isfinite(peak_error) or peak_error > 5e-4:
        raise RuntimeError(f"Stream differs from offline reference: peak error {peak_error:.8f}.")

    return {
        "chunk_ms": chunk_ms,
        "samples": len(estimate),
        "enrollment_ms": round(enrollment_ms, 2),
        "rtf": round(sum(elapsed) * SAMPLE_RATE / len(mixture), 3),
        "p95_pipeline_ms": round(float(np.percentile(latency, 95)), 2),
        "si_sdri_db": round(si_sdr(reference, estimate) - si_sdr(reference, mixture), 2),
        "offline_peak_error": round(peak_error, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR.with_name("spk_emb_causal_100"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cues", nargs="+", choices=("A", "B"), default=["A", "B"])
    parser.add_argument("--chunks", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()
    mixture = read_audio(args.case / "mixed.wav")
    if any(ms < 8 or ms * SAMPLE_RATE // 1000 > len(mixture) for ms in args.chunks):
        parser.error("Use chunk durations from 8 ms to the length of the mixture.")

    extractor = Extractor(args.model, args.device)
    if any(layer.band_rnn.rnn.bidirectional for layer in extractor.extractor.model.sep_model.separator.separation):
        parser.error("Use the causal REAL-TSE speaker-embedding checkpoint.")

    rows = []
    for cue in args.cues:
        enrollment_path, reference_path = ("enroll.wav", "target.wav") if cue == "A" else (
            "enroll_b.wav", "interferer.wav",
        )
        enrollment = read_audio(args.case / enrollment_path, min_seconds=1)
        reference = read_audio(args.case / reference_path)
        extractor.extract(mixture[:2048], enrollment)

        torch.manual_seed(SEED)
        offline = extractor.extract(mixture, enrollment)
        for size in args.chunks:
            row = {"cue": cue, **measure(extractor, mixture, enrollment, reference, offline, size)}
            rows.append(row)
            print(row, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": args.model.name,
        "case": args.case.name,
        "device": args.device,
        "torch_threads": torch.get_num_threads(),
        "chunks": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
