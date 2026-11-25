"""Two-singer separation with enrollment-based assignment at 16 kHz."""

import importlib.util
import os
import sys
from hashlib import file_digest
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.signal import find_peaks, resample_poly

from backend.app.audio.wav import SAMPLE_RATE, Signal
from backend.app.models.realtse import ROOT, VENDOR_DIR

MODEL_ID = "harmony-vocal-split"
MODEL_NAME = "UNMIXX · 노래 음색 대조"
MODEL_VERSION = "unmixx-8e75052"
SOURCE = VENDOR_DIR / "unmixx"
WEIGHTS = ROOT / ".voicefocus" / "models" / "singer-identity.ts"
SINGER_SHA256 = "97f17b9de1848877ed16801cb873b9c615fcf65cc23fa1c43dfa0d0fe948f40a"
SPLITTER_SHA256 = "edd8bd0782f85ebb8a6b26bec47808b3130d15c2ccd49c741d2ff5983cb6c359"
CONFIG_SHA256 = "93411b004da365abf26a694be310a79f655339c2a472e73872560943e1150412"
WINDOW = 6 * SAMPLE_RATE
OVERLAP = SAMPLE_RATE
PITCH_WEIGHT = 0.025
MATCH_MARGIN = 0.04


def available() -> bool:
    return (
        (SOURCE / "look2hear" / "models" / "unmixx_model.py").is_file()
        and (SOURCE / "ckpt" / "best.ckpt").is_file()
        and (SOURCE / "ckpt" / "conf.yml").is_file()
        and WEIGHTS.is_file()
        and all(importlib.util.find_spec(package) is not None for package in (
            "torch", "yaml", "librosa", "asteroid", "torch_complex", "typeguard", "huggingface_hub",
        ))
    )


def verify_file(path: Path, expected: str) -> None:
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {path.name}")


def median_pitch(signal: Signal) -> float | None:
    """Median MIDI pitch across energetic frames with a periodicity minimum."""
    frame_size = 800
    if len(signal) < frame_size:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::320].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    energy = (frames ** 2).sum(axis=1)
    frames = frames[energy > max(1e-12, energy.max() * 0.008)]
    if not len(frames):
        return None

    # YIN's cumulative mean-normalized difference computed using FFT correlation.
    lags = np.arange(267)
    spectrum = np.fft.rfft(frames, n=2048)
    correlation = np.fft.irfft(spectrum * spectrum.conj(), axis=1)[:, :len(lags)]
    squares = np.pad(np.cumsum(frames ** 2, axis=1), ((0, 0), (1, 0)))
    difference = np.maximum(1e-12, squares[:, frame_size - lags] + squares[:, -1:] - squares[:, lags] - 2 * correlation)
    normalized = difference[:, 1:] * lags[1:] / np.cumsum(difference[:, 1:], axis=1)
    periods = []
    for frame in normalized:
        minima, _ = find_peaks(-frame)
        voiced = minima[(minima + 1 >= 20) & (frame[minima] < 0.30)]
        if len(voiced):
            periods.append(voiced[0] + 1)
    if len(periods) < 5:
        return None
    return float(np.median(69 + 12 * np.log2(SAMPLE_RATE / (440 * np.asarray(periods)))))


def match_scores(
    reference: Signal, candidates: tuple[Signal, Signal],
    similarities: tuple[float, float],
) -> tuple[float, float]:
    reference_pitch = median_pitch(reference)
    candidates_pitch = tuple(map(median_pitch, candidates))
    first, second = candidates_pitch
    if reference_pitch is None or first is None or second is None:
        return similarities
    return (
        similarities[0] - PITCH_WEIGHT * min(12, abs(reference_pitch - first)),
        similarities[1] - PITCH_WEIGHT * min(12, abs(reference_pitch - second)),
    )


def overlap_order(previous: np.ndarray, current: np.ndarray) -> tuple[bool, bool]:
    """Use the common audio to follow singer order across independent windows."""
    norms = np.linalg.norm(previous, axis=1)[:, None] * np.linalg.norm(current, axis=1)[None, :]
    similarities = (previous @ current.T) / np.maximum(norms, 1e-8)
    straight = float(similarities[0, 0] + similarities[1, 1])
    swapped = float(similarities[0, 1] + similarities[1, 0])
    return swapped > straight, abs(swapped - straight) < 0.2


class HarmonyExtractor:
    def __init__(self, device: str = "cpu") -> None:
        import torch
        import yaml

        checkpoint = SOURCE / "ckpt" / "best.ckpt"
        config = SOURCE / "ckpt" / "conf.yml"
        verify_file(checkpoint, SPLITTER_SHA256)
        verify_file(config, CONFIG_SHA256)
        # TorchScript can execute code when loaded. Accept only the pinned release.
        verify_file(WEIGHTS, SINGER_SHA256)
        sys.path.insert(0, str(SOURCE))
        # The upstream initializer eagerly loads training tools. The separator
        # imports inference utilities directly from submodules of this package.
        helpers = ModuleType("look2hear.utils")
        helpers.__path__ = [str(SOURCE / "look2hear" / "utils")]
        sys.modules.setdefault("look2hear.utils", helpers)
        from look2hear.models import UNMIXX

        torch.set_num_threads(int(os.environ.get("VOICEFOCUS_TORCH_THREADS", "2")))
        self.torch = torch
        self.device = torch.device(device)
        cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
        self.separator = UNMIXX(sample_rate=24000, **cfg["audionet"]["audionet_config"])
        parameters = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
        self.separator.load_state_dict({key.removeprefix("audio_model."): value for key, value in parameters.items()})
        self.separator.to(self.device).eval()
        self.encoder = torch.jit.load(str(WEIGHTS), map_location=self.device).eval()

    def _embedding(self, signal: Signal) -> Signal:
        # The encoder expects 44.1 kHz and can also compare short vocal clips.
        audio = resample_poly(signal, 441, 160)
        audio /= max(1e-8, float(np.max(np.abs(audio))))
        with self.torch.inference_mode():
            vector = self.encoder(self.torch.from_numpy(audio[None, :]).to(self.device)).cpu().numpy().reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid singing voice embedding.")
        return vector / norm

    def _split(self, mixture: Signal) -> np.ndarray:
        audio = resample_poly(mixture, 3, 2)
        with self.torch.inference_mode():
            stems, *_ = self.separator(self.torch.from_numpy(audio[None, None, :]).to(self.device), istest=True)
        stems = resample_poly(stems[0].cpu().numpy(), 2, 3, axis=-1)[..., :len(mixture)]
        if stems.shape != (2, len(mixture)) or not np.isfinite(stems).all():
            raise ValueError("Unexpected singing separation output.")
        return stems

    def split(self, mixture: Signal) -> tuple[np.ndarray, bool]:
        if mixture.ndim != 1 or len(mixture) < SAMPLE_RATE:
            raise ValueError("Harmony separation requires at least one second of mono audio.")
        stems = self._split(mixture[:WINDOW])
        uncertain = False
        while stems.shape[1] < len(mixture):
            start = stems.shape[1] - OVERLAP
            new = self._split(mixture[start:start + WINDOW])
            swap, ambiguous = overlap_order(stems[:, -OVERLAP:], new[:, :OVERLAP])
            if swap:
                new = new[::-1]
            weights = np.linspace(0, 1, OVERLAP, dtype=np.float32)
            stems[:, -OVERLAP:] += weights * (new[:, :OVERLAP] - stems[:, -OVERLAP:])
            stems = np.concatenate((stems, new[:, OVERLAP:]), axis=1)
            uncertain |= ambiguous

        # Project the two estimates onto the original mix at their native gains.
        stems += (mixture - stems.sum(axis=0)) * 0.5
        # Mark an ambiguous assignment if the separator did not find two audible candidates.
        level = np.sqrt(np.mean(stems ** 2, axis=1))
        uncertain |= bool(min(level) < 0.05 * max(level))
        return stems, uncertain

    def identify(self, stems: np.ndarray, enrollment: Signal, uncertain: bool) -> tuple[Signal, Signal, bool]:
        reference = self._embedding(enrollment)
        candidates = (stems[0], stems[1])
        first, second = match_scores(enrollment, candidates, (
            float(reference @ self._embedding(candidates[0])),
            float(reference @ self._embedding(candidates[1])),
        ))
        target = int(second > first)
        uncertain |= abs(second - first) < MATCH_MARGIN
        return stems[target], stems[1 - target], uncertain

    def separate(self, mixture: Signal, enrollment: Signal) -> tuple[Signal, Signal, bool]:
        stems, uncertain = self.split(mixture)
        return self.identify(stems, enrollment, uncertain)
