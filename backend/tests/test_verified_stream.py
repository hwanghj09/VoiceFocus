from types import SimpleNamespace

import numpy as np
import pytest


def test_verified_stream_releases_earlier_with_history_and_mutes_rejected_samples(monkeypatch):
    pytest.importorskip("torch")
    from backend.app.models import live

    class AlignedSeparator:
        started = False

        def __init__(self, _extractor, _enrollment):
            pass

        def push(self, audio):
            self.started = True
            return audio

        def finish(self):
            return np.empty(0, dtype=np.float32)

    evidence = []
    decisions = iter((0.80, 0.15, 0.80))

    class Verifier:
        def __init__(self, _enrollment, _embed, *, segments):
            assert len(segments) == 3
            self.references = np.ones((8, 1))

        def select(self, mixture, estimate):
            np.testing.assert_array_equal(mixture, estimate)
            evidence.append(mixture.copy())
            return estimate, next(decisions)

    monkeypatch.setattr(live, "CausalStream", AlignedSeparator)
    monkeypatch.setattr(live, "SpeakerVerifier", Verifier)
    pcm = np.linspace(0.1, 0.5, live.UNPLAYED_CONTEXT + live.STEP, dtype=np.float32)
    stream = live.VerifiedStream(SimpleNamespace(embedding=None), np.tile(pcm, 2))
    assert stream.push(pcm[:live.UNPLAYED_CONTEXT - 1]).size == 0
    first = stream.push(pcm[live.UNPLAYED_CONTEXT - 1:live.UNPLAYED_CONTEXT])
    middle = stream.push(pcm[live.UNPLAYED_CONTEXT:])
    last = stream.finish()

    assert len(first) == len(middle) == live.STEP
    assert len(first) + len(middle) + len(last) == len(pcm)
    np.testing.assert_array_equal(first[live.FADE:], pcm[live.FADE:live.STEP])
    assert not np.any(middle)
    assert last[0] == 0
    np.testing.assert_array_equal(last[live.FADE:], pcm[2 * live.STEP + live.FADE:])
    np.testing.assert_array_equal(evidence[0], pcm[:live.UNPLAYED_CONTEXT])
    np.testing.assert_array_equal(evidence[1], pcm)

    class QuietEarlyVerifier(Verifier):
        def select(self, _mixture, estimate):
            return estimate * 0.05 if len(estimate) < live.CONTEXT else estimate, 0.80

    monkeypatch.setattr(live, "SpeakerVerifier", QuietEarlyVerifier)
    longer_pcm = np.linspace(0.1, 0.5, live.CONTEXT, dtype=np.float32)
    stream = live.VerifiedStream(SimpleNamespace(embedding=None), np.tile(pcm, 2))
    assert stream.push(longer_pcm[:live.UNPLAYED_CONTEXT]).size == 0
    first = stream.push(longer_pcm[live.UNPLAYED_CONTEXT:])
    assert len(first) == live.STANDARD_STEP
    np.testing.assert_array_equal(first[live.FADE:], longer_pcm[live.FADE:live.STANDARD_STEP])
    assert len(stream.finish()) + len(first) == len(longer_pcm)


def test_live_stream_reuses_weights_but_creates_fresh_session(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    from backend.app import live
    from backend.app.models import live as stream_model, worker

    loaded = []

    def load_weights(path, device):
        loaded.append((path, device))
        return object()

    monkeypatch.setattr(worker, "Extractor", load_weights)
    monkeypatch.setattr(stream_model, "VerifiedStream", lambda weights, cue: SimpleNamespace(weights=weights, cue=cue))
    live.live_extractor.cache_clear()
    try:
        first = live.load_stream(tmp_path, "cpu", np.full(16_000, 0.1, dtype=np.float32))
        second = live.load_stream(tmp_path, "cpu", np.full(16_000, 0.2, dtype=np.float32))
    finally:
        live.live_extractor.cache_clear()

    assert loaded == [(tmp_path, "cpu")]
    assert first is not second and first.weights is second.weights
    assert not np.array_equal(first.cue, second.cue)
