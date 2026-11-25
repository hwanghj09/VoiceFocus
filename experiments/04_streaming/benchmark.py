"""Measure a causal checkpoint in a simulated, sequential PCM chunk pipeline."""
import argparse
import importlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from backend.app.audio.wav import SAMPLE_RATE, read_audio
from backend.app.models.realtse import DEFAULT_MODEL_DIR
from backend.app.models.worker import Extractor

si_sdr = importlib.import_module("experiments.03_tse_baseline.evaluate").si_sdr


def simulate(extractor, mixture, enrollment, target, chunk_ms):
    chunk_samples = round(chunk_ms / 1000 * SAMPLE_RATE)
    available_at = 0.0
    latency, compute, chunks = [], [], []
    for begin in range(0, len(mixture) - chunk_samples + 1, chunk_samples):
        end = begin + chunk_samples
        chunk = mixture[begin:end]
        start = perf_counter()
        chunks.append(extractor.extract(chunk, enrollment))
        elapsed = perf_counter() - start
        # Clock starts when the first input sample of each chunk was captured.
        available_at = max(available_at, end / SAMPLE_RATE) + elapsed
        latency.append((available_at - begin / SAMPLE_RATE) * 1000)
        compute.append(elapsed)
    estimate = np.concatenate(chunks)
    return {
        "chunk_ms": chunk_ms,
        "chunks": len(chunks),
        "rtf": round(sum(compute) * SAMPLE_RATE / len(estimate), 3),
        "p50_inference_ms": round(float(np.median(compute)) * 1000, 2),
        "p95_pipeline_ms": round(float(np.percentile(latency, 95)), 2),
        "si_sdri_db": round(
            si_sdr(target[:len(estimate)], estimate) - si_sdr(target[:len(estimate)], mixture[:len(estimate)]), 2,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR.with_name("spk_emb_causal_100"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--chunks", nargs="+", type=int, default=(64, 128, 256, 512))
    args = parser.parse_args()
    mixture = read_audio(args.case / "mixed.wav")
    enrollment = read_audio(args.case / "enroll.wav", min_seconds=1)
    target = read_audio(args.case / "target.wav")
    if any(ms < 32 or ms > len(mixture) / SAMPLE_RATE * 1000 for ms in args.chunks):
        parser.error("Chunk durations must be 32 ms or longer and fit inside the input.")
    extractor = Extractor(args.model, args.device)
    if extractor.extractor.model.sep_model.separator.separation[0].band_rnn.rnn.bidirectional:
        parser.error("Use the causal REAL-TSE checkpoint.")
    extractor.extract(mixture[:4096], enrollment)
    start = perf_counter()
    whole = extractor.extract(mixture, enrollment)
    whole_rtf = (perf_counter() - start) * SAMPLE_RATE / len(mixture)
    report = {
        "checkpoint": args.model.name,
        "whole_rtf": round(whole_rtf, 3),
        "whole_si_sdri_db": round(si_sdr(target, whole) - si_sdr(target, mixture), 2),
        "chunks": [],
    }
    for size in args.chunks:
        row = simulate(extractor, mixture, enrollment, target, size)
        report["chunks"].append(row)
        print(row, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
