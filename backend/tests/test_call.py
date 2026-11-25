import numpy as np
import pytest

from backend.app.audio.wav import SAMPLE_RATE
from backend.app.models.call import verify_speech


def voice(frequency, seconds=1):
    time = np.arange(seconds * SAMPLE_RATE) / SAMPLE_RATE
    return (0.15 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def embedding(samples):
    time = np.arange(len(samples)) / SAMPLE_RATE
    return np.array([abs(samples @ np.exp(-2j * np.pi * frequency * time)) for frequency in (480, 720)])


@pytest.mark.parametrize("gain", [1.25, -1.25])
def test_call_switches_away_from_the_other_speaker(gain):
    target, other = voice(480), voice(720)
    result = verify_speech(target + other, gain * other, target, embedding)
    np.testing.assert_allclose(result, target, atol=1e-7, rtol=0)


def test_call_mutes_if_neither_candidate_matches():
    target, other = voice(480), voice(720)
    result = verify_speech(other, other, target, embedding)
    np.testing.assert_array_equal(result, np.zeros_like(other))


def test_call_does_not_fall_back_to_nearly_the_whole_mixture():
    other, residual = voice(480) * 0.4, voice(720)
    result = verify_speech(other + residual, other, residual, embedding)
    np.testing.assert_array_equal(result, np.zeros_like(other))


def test_call_mutes_unverified_region_within_a_recording():
    target = voice(480)
    estimate = np.concatenate((voice(720, seconds=2), voice(480, seconds=3)))
    result = verify_speech(estimate, estimate, target, embedding)
    np.testing.assert_array_equal(result[:SAMPLE_RATE], np.zeros_like(target))
    np.testing.assert_allclose(result[-SAMPLE_RATE:], target, atol=1e-7, rtol=0)
