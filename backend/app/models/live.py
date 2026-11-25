"""Apply target verification to a continuous, sample-aligned causal stream."""

import numpy as np

from backend.app.audio.wav import SAMPLE_RATE, Signal
from backend.app.models.call import FADE, MIN_MATCH, SpeakerVerifier
from backend.app.models.causal import CausalStream
from backend.app.models.worker import Extractor

CONTEXT = 5 * SAMPLE_RATE // 2
STEP = 3 * SAMPLE_RATE // 4


class VerifiedStream:
    def __init__(self, extractor: Extractor, enrollment: Signal) -> None:
        self.separator = CausalStream(extractor, enrollment)
        self.verifier = SpeakerVerifier(enrollment, extractor.embedding)
        self.mixture = np.empty(0, dtype=np.float32)
        self.estimate = np.empty(0, dtype=np.float32)
        self.history = np.empty((2, 0), dtype=np.float32)
        self.previous = np.empty(0, dtype=np.float32)
        self.ramp = np.linspace(0, 1, FADE, dtype=np.float32)

    def _release(self, size: int, *, finishing: bool = False) -> Signal:
        length = min(CONTEXT, len(self.estimate))
        signals = np.stack((self.mixture[:length], self.estimate[:length]))
        offset = 0
        if finishing and length < CONTEXT:
            offset = min(CONTEXT - length, self.history.shape[1])
            signals = np.concatenate((self.history[:, self.history.shape[1] - offset:], signals), axis=1)
        if signals.shape[1] >= SAMPLE_RATE // 2 and np.isfinite(signals[1]).all():
            candidate, score = self.verifier.select(*signals)
        else:
            candidate, score = signals[1], -1.0
        candidate = candidate[offset:]
        output = np.zeros(size, dtype=np.float32)
        if score >= MIN_MATCH:
            output[:] = candidate[:size]
            fade = min(FADE, size)
            if fade:
                previous = self.previous[:fade] if len(self.previous) >= fade else 0
                output[:fade] = previous * (1 - self.ramp[:fade]) + output[:fade] * self.ramp[:fade]
        self.previous = candidate[size:size + FADE].copy() if score >= MIN_MATCH else np.empty(0, dtype=np.float32)
        if not finishing:
            self.history = np.concatenate((self.history, signals[:, :size]), axis=1)[:, -CONTEXT:]
        self.estimate = self.estimate[size:].copy()
        self.mixture = self.mixture[size:].copy()
        return output

    def push(self, samples: Signal) -> Signal:
        self.mixture = np.concatenate((self.mixture, samples))
        self.estimate = np.concatenate((self.estimate, self.separator.push(samples)))
        if len(self.estimate) >= CONTEXT:
            return self._release(STEP)
        return np.empty(0, dtype=np.float32)

    def finish(self) -> Signal:
        if not self.separator.started:
            return np.zeros_like(self.mixture)
        self.estimate = np.concatenate((self.estimate, self.separator.finish()))
        if len(self.estimate) > CONTEXT:
            return np.concatenate((self._release(STEP), self._release(len(self.estimate), finishing=True)))
        return self._release(len(self.estimate), finishing=True)
