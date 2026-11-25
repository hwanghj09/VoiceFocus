"""Apply target verification to a continuous, sample-aligned causal stream."""

import numpy as np

from backend.app.audio.wav import SAMPLE_RATE, Signal
from backend.app.models.call import FADE, MIN_MATCH, SpeakerVerifier
from backend.app.models.causal import CausalStream
from backend.app.models.worker import Extractor

CONTEXT = 5 * SAMPLE_RATE // 2
UNPLAYED_CONTEXT = 5 * SAMPLE_RATE // 4
STEP = 5 * SAMPLE_RATE // 8
STANDARD_STEP = 3 * SAMPLE_RATE // 4
ENROLLMENT_SEGMENT = 2 * SAMPLE_RATE
SHORT_CONTEXT_MATCH = 0.30
MIN_EARLY_ENERGY_RATIO = 0.04


class VerifiedStream:
    def __init__(self, extractor: Extractor, enrollment: Signal) -> None:
        self.separator = CausalStream(extractor, enrollment)
        last_start = len(enrollment) - ENROLLMENT_SEGMENT
        segments = tuple(
            enrollment[start:start + ENROLLMENT_SEGMENT]
            for start in sorted({0, last_start // 2, last_start})
        ) if last_start > 0 else ()
        self.verifier = SpeakerVerifier(enrollment, extractor.embedding, segments=segments)
        self.mixture = np.empty(0, dtype=np.float32)
        self.estimate = np.empty(0, dtype=np.float32)
        self.history = np.empty((2, 0), dtype=np.float32)
        self.previous = np.empty(0, dtype=np.float32)
        self.ramp = np.linspace(0, 1, FADE, dtype=np.float32)
        self.short_context: bool | None = None

    @property
    def unplayed_context(self) -> int:
        return CONTEXT if self.short_context is False else UNPLAYED_CONTEXT

    def _use_standard_context(self) -> None:
        self.short_context = False
        # The first two references cover the full enrollment in both bands.
        self.verifier.references = self.verifier.references[:2]

    def _release(self, size: int, *, finishing: bool = False) -> Signal:
        length = min(self.unplayed_context, len(self.estimate))
        current = np.stack((self.mixture[:length], self.estimate[:length]))
        offset = min(CONTEXT - length, self.history.shape[1]) if self.short_context or finishing else 0
        signals = np.concatenate((self.history[:, self.history.shape[1] - offset:], current), axis=1)
        if signals.shape[1] >= SAMPLE_RATE // 2 and np.isfinite(signals[1]).all():
            candidate, score = self.verifier.select(*signals)
        else:
            candidate, score = signals[1], -1.0
        candidate = candidate[offset:]
        short_window = self.short_context is not False and UNPLAYED_CONTEXT <= signals.shape[1] < CONTEXT
        threshold = SHORT_CONTEXT_MATCH if short_window else MIN_MATCH
        accepted = score >= threshold
        if self.short_context is None:
            block, mixture = candidate[:size], self.mixture[:size]
            if not accepted or float(block @ block) < MIN_EARLY_ENERGY_RATIO * max(1e-8, float(mixture @ mixture)):
                self._use_standard_context()
                return np.empty(0, dtype=np.float32)
            self.short_context = True
        output = np.zeros(size, dtype=np.float32)
        if accepted:
            output[:] = candidate[:size]
            fade = min(FADE, size)
            if fade:
                previous = self.previous[:fade] if len(self.previous) >= fade else 0
                output[:fade] = previous * (1 - self.ramp[:fade]) + output[:fade] * self.ramp[:fade]
        self.previous = candidate[size:size + FADE].copy() if accepted else np.empty(0, dtype=np.float32)
        self.history = np.concatenate((self.history, current[:, :size]), axis=1)[:, -CONTEXT:]
        self.estimate = self.estimate[size:].copy()
        self.mixture = self.mixture[size:].copy()
        return output

    def push(self, samples: Signal) -> Signal:
        estimate = self.separator.push(samples)
        self.mixture = np.concatenate((self.mixture, samples))
        self.estimate = np.concatenate((self.estimate, estimate))
        return self._drain()

    def _drain(self, *, finishing: bool = False) -> Signal:
        blocks = []
        # At end of input, verify the last window once and drain its entire tail.
        while len(self.estimate) >= self.unplayed_context + int(finishing):
            size = STANDARD_STEP if self.short_context is False else STEP
            block = self._release(size)
            if len(block):
                blocks.append(block)
        if finishing and len(self.estimate):
            if self.short_context is None:
                self._use_standard_context()
            blocks.append(self._release(len(self.estimate), finishing=True))
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=np.float32)

    def finish(self) -> Signal:
        if not self.separator.started:
            return np.zeros_like(self.mixture)
        self.estimate = np.concatenate((self.estimate, self.separator.finish()))
        return self._drain(finishing=True)
