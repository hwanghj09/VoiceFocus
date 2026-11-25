"""Verify the enrolled speaker before releasing an offline call estimate."""

from collections.abc import Callable

import numpy as np
from scipy.signal import butter, sosfiltfilt

from backend.app.audio.wav import SAMPLE_RATE, Signal

CONTEXT = 3 * SAMPLE_RATE // 2
HOP = SAMPLE_RATE // 4
FADE = SAMPLE_RATE // 50
MIN_MATCH = 0.33
RESIDUAL_MARGIN = 0.08
MIN_EXPLAINED_ENERGY = 0.25


def telephone_band(audio: Signal) -> Signal:
    band = butter(4, (300, 3400), btype="bandpass", fs=SAMPLE_RATE, output="sos")
    return sosfiltfilt(band, audio).astype(np.float32)


def unit(vector: Signal) -> Signal:
    return vector / max(1e-8, float(np.linalg.norm(vector)))


class SpeakerVerifier:
    def __init__(self, enrollment: Signal, embed: Callable[[Signal], Signal]) -> None:
        self.embed = embed
        self.references = np.stack((unit(embed(enrollment)), unit(embed(telephone_band(enrollment)))))

    def similarity(self, signal: Signal) -> float:
        if np.max(np.abs(signal)) < 1e-4:
            return -1.0
        return float(np.max(self.references @ unit(self.embed(signal))))

    def select(self, mixture: Signal, estimate: Signal) -> tuple[Signal, float]:
        # The TSE objective is scale invariant, so match the component's gain before subtraction.
        estimate_energy = float(estimate @ estimate)
        gain = np.clip(float(mixture @ estimate) / max(1e-8, estimate_energy), -2, 2)
        explained = gain ** 2 * estimate_energy / max(1e-8, float(mixture @ mixture))
        score = self.similarity(estimate)
        if explained >= MIN_EXPLAINED_ENERGY:
            residual = mixture - np.float32(gain) * estimate
            residual_score = self.similarity(residual)
            if residual_score >= max(MIN_MATCH, score + RESIDUAL_MARGIN):
                return residual, residual_score
        return (-estimate if gain < 0 else estimate), score


def verify_speech(mixture: Signal, estimate: Signal, enrollment: Signal, embed: Callable[[Signal], Signal]) -> Signal:
    """Choose a verified candidate and mute blocks that lack local evidence."""
    verifier = SpeakerVerifier(enrollment, embed)
    candidate, score = verifier.select(mixture, estimate)
    if score < MIN_MATCH:
        return np.zeros_like(mixture)
    if len(candidate) <= CONTEXT:
        return candidate

    # Short windows have less identity evidence than the full recording.
    local_threshold = np.clip(score - MIN_MATCH, 0.15, 0.21)
    result = np.zeros_like(candidate)
    decisions = {}
    kept = []
    for start in range(0, len(candidate), HOP):
        context_start = min(max(0, start + (HOP - CONTEXT) // 2), len(candidate) - CONTEXT)
        if context_start not in decisions:
            decisions[context_start] = verifier.similarity(candidate[context_start:context_start + CONTEXT]) >= local_threshold
        allowed = decisions[context_start]
        kept.append(allowed)
        if allowed:
            result[start:start + HOP] = candidate[start:start + HOP]

    # Fade only inside kept blocks. Samples within rejected blocks stay at zero.
    ramp = np.linspace(0, 1, FADE, dtype=np.float32)
    for boundary in (np.flatnonzero(np.diff(kept)) + 1) * HOP:
        if kept[boundary // HOP]:
            fade = result[boundary:boundary + FADE]
            fade *= ramp[:len(fade)]
        else:
            result[boundary - FADE:boundary] *= ramp[::-1]
    return result
