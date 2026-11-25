import numpy as np
import pytest

from backend.app.audio.wav import SAMPLE_RATE
from backend.app.models.harmony import HarmonyExtractor, WINDOW, match_scores, median_pitch, overlap_order


def sine(hz, frames=SAMPLE_RATE):
    times = np.arange(frames) / SAMPLE_RATE
    return (0.12 * np.sin(2 * np.pi * hz * times)).astype(np.float32)


def test_pitch_prior_distinguishes_close_timbre_matches():
    candidates = (sine(262), sine(330))
    reference = sine(277)
    assert median_pitch(reference) == pytest.approx(61.0, abs=0.2)
    first, second = match_scores(reference, candidates, (0.72, 0.75))
    assert first > second
    assert match_scores(np.zeros(SAMPLE_RATE, dtype=np.float32), candidates, (0.72, 0.75)) == (0.72, 0.75)


def test_overlap_alignment_detects_swapped_and_silent_sources():
    sources = np.stack((sine(210), sine(381)))
    assert overlap_order(sources, sources[::-1]) == (True, False)
    assert overlap_order(sources, np.zeros_like(sources)) == (False, True)


def test_chunk_alignment_keeps_sources_and_exact_mixture(monkeypatch):
    frames = WINDOW + 3200
    first = sine(244, frames)
    second = sine(330, frames)
    model = HarmonyExtractor.__new__(HarmonyExtractor)
    chunks = iter([np.stack((first[:WINDOW], second[:WINDOW])),
                   np.stack((second[WINDOW-SAMPLE_RATE:], first[WINDOW-SAMPLE_RATE:]))])
    monkeypatch.setattr(model, "_split", lambda _part: next(chunks))
    mixture = first + second
    stems, ambiguous = model.split(mixture)
    assert not ambiguous
    np.testing.assert_allclose(stems[0], first, atol=1e-7, rtol=0)
    np.testing.assert_allclose(stems[1], second, atol=1e-7, rtol=0)
    np.testing.assert_allclose(stems.sum(axis=0), mixture, atol=1e-7, rtol=0)
