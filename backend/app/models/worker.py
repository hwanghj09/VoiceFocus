import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
import torch
from backend.app.audio.wav import SAMPLE_RATE, Signal, write_analysis_wav
from backend.app.models.realtse import VENDOR_DIR


class Extractor:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        sys.path[:0] = [str(VENDOR_DIR / "wesep-real-tse"), str(VENDOR_DIR / "wespeaker")]
        from wesep import load_model_local

        torch.set_num_threads(int(os.environ.get("VOICEFOCUS_TORCH_THREADS", "2")))
        self.extractor = load_model_local(str(model_dir))
        self.extractor.set_device(device)
        self.extractor.set_output_norm(False)
        if self.extractor.resample_rate != SAMPLE_RATE:
            raise ValueError("VoiceFocus expects a 16 kHz model.")

    def extract(self, mixture: Signal, enrollment: Signal) -> Signal:
        # The upstream iSTFT rounds down to a whole hop if no length is set.
        hop = self.extractor.model.sep_model.stft.hop_length
        padded = np.pad(mixture, (0, -len(mixture) % hop), mode="reflect")
        with torch.inference_mode():
            result = self.extractor.extract_speech_from_pcm(
                torch.from_numpy(padded[None, :]), SAMPLE_RATE,
                torch.from_numpy(enrollment[None, :]), SAMPLE_RATE,
            )
        if result is None or result.numel() != len(padded):
            raise ValueError("Unexpected TSE output length.")
        signal = result.numpy().reshape(-1)[:len(mixture)]
        if not np.isfinite(signal).all():
            raise ValueError("TSE returned invalid samples.")
        return signal

    def embedding(self, signal: Signal) -> Signal:
        frontend = self.extractor.model.spk_ft
        waveform = torch.from_numpy(signal[None, :]).to(self.extractor.device)
        with torch.inference_mode():
            result = frontend.encoder(frontend.fbank(waveform))
            if isinstance(result, (list, tuple)):
                result = result[-1]
        return result.cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the enrolled speaker using REAL-TSE.")
    model_choice = parser.add_mutually_exclusive_group(required=True)
    model_choice.add_argument("--model", type=Path)
    model_choice.add_argument("--residual", type=Path, help="Separate singing voices and write the remaining audio here.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mixture", required=True, type=Path)
    parser.add_argument("--enroll", required=True, type=Path)
    parser.add_argument("--keep-enroll", type=Path, help="Optional recording of the singer to keep in harmony mode.")
    parser.add_argument("--verify-target", action="store_true", help="Verify target speech before releasing the call result.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args()
    if args.keep_enroll is not None and args.residual is None:
        parser.error("--keep-enroll requires harmony separation.")
    if args.verify_target and args.model is None:
        parser.error("--verify-target requires a speech model.")

    mixture, mix_rate = sf.read(args.mixture, dtype="float32")
    enrollment, enroll_rate = sf.read(args.enroll, dtype="float32")
    if any(data.ndim != 1 or rate != SAMPLE_RATE for data, rate in (
        (mixture, mix_rate), (enrollment, enroll_rate),
    )):
        parser.error("Expected mono, 16 kHz WAVs.")
    metadata = {}
    if args.residual is not None:
        from backend.app.models.harmony import HarmonyExtractor

        keep_enrollment = None
        if args.keep_enroll is not None:
            keep_enrollment, keep_rate = sf.read(args.keep_enroll, dtype="float32")
            if keep_enrollment.ndim != 1 or keep_rate != SAMPLE_RATE:
                parser.error("Expected a mono, 16 kHz reference for the singer to keep.")
        splitter = HarmonyExtractor(args.device)
        start = perf_counter()
        result, remaining, uncertain = splitter.separate(mixture, enrollment, keep_enrollment)
        metadata["inference_ms"] = (perf_counter() - start) * 1000
        metadata["uncertain_match"] = uncertain
        write_analysis_wav(args.residual, remaining)
    else:
        extractor = Extractor(args.model, args.device)
        start = perf_counter()
        result = extractor.extract(mixture, enrollment)
        if args.verify_target:
            from backend.app.models.call import verify_speech

            result = verify_speech(mixture, result, enrollment, extractor.embedding)
            metadata["target_muted"] = not bool(np.any(result))
        metadata["inference_ms"] = (perf_counter() - start) * 1000
    write_analysis_wav(args.output, result)
    args.metrics.write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    main()
