"""Real media decoding regressions (no AI checkpoint required)."""
from io import BytesIO
import subprocess

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.audio import wav
from backend.app import main


def encoded(tmp_path, fmt, codec, seconds=1.5, silence=False):
    rate = 48000
    t = np.arange(round(rate * seconds)) / rate
    signal = np.zeros_like(t) if silence else 0.2 * np.sin(2 * np.pi * 440 * t)
    source = tmp_path / "source.wav"
    sf.write(source, np.column_stack([signal, signal]), rate, subtype="PCM_16")
    output = tmp_path / "encoded.bin"
    subprocess.run(
        [wav.ffmpeg_executable(), "-nostdin", "-hide_banner", "-loglevel", "error",
         "-y", "-i", str(source), "-c:a", codec, "-f", fmt, str(output)],
        check=True, timeout=30, capture_output=True,
    )
    return output.read_bytes()


@pytest.mark.parametrize("fmt,codec", [
    ("wav", "pcm_s16le"), ("mp3", "libmp3lame"), ("ipod", "aac"),
    ("adts", "aac"), ("mp4", "aac"), ("webm", "libopus"), ("flac", "flac"),
])
def test_decode_content_not_extension_and_normalize(tmp_path, fmt, codec):
    payload = encoded(tmp_path, fmt, codec)
    misleading = tmp_path / "recording.wav"
    misleading.write_bytes(payload)
    for source in (misleading, BytesIO(payload)):
        signal = wav.read_audio(source, wav_only=False)
        assert signal.dtype == np.float32 and signal.ndim == 1
        # Lossy codecs may add encoder padding.
        assert abs(len(signal) - 24000) < 2000
        assert np.isfinite(signal).all() and 0.01 < np.max(np.abs(signal)) <= 1
        decoded, rate = sf.read(BytesIO(wav.playback_wav(signal)))
        assert rate == 16000 and decoded.ndim == 1
        assert sf.info(BytesIO(wav.playback_wav(signal))).subtype == "PCM_16"


def test_api_mixes_disguised_aac_and_flac(tmp_path):
    a = encoded(tmp_path, "mp4", "aac")
    b = encoded(tmp_path, "flac", "flac")
    with TestClient(main.create_app()) as client:
        response = client.post("/api/v1/mix", files={
            "a": ("recording.wav", a, "audio/wav"),
            "b": ("recording.flac", b, "audio/flac"),
        })
    assert response.status_code == 200, response.text
    assert sf.info(BytesIO(response.content)).samplerate == 16000
    assert sf.info(BytesIO(response.content)).channels == 1


def test_decode_rejects_silence_duration_and_invalid_data(tmp_path):
    for payload, limit in (
        (encoded(tmp_path, "mp4", "aac", silence=True), 30),
        (encoded(tmp_path, "mp4", "aac", seconds=3), 1),
        (b"not audio", 30),
    ):
        with pytest.raises(wav.AudioError):
            wav.read_audio(BytesIO(payload), max_seconds=limit, wav_only=False)


def test_decode_rejects_oversized_encoded_payload(monkeypatch):
    monkeypatch.setattr(wav, "MAX_FILE_BYTES", 8)
    with pytest.raises(wav.AudioError, match="16 MiB"):
        wav.decode_media(BytesIO(b"x" * 9), 0.1, 30)


def test_decode_timeout_is_reported(tmp_path, monkeypatch):
    payload = encoded(tmp_path, "mp4", "aac")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 20)
    monkeypatch.setattr(wav.subprocess, "run", timeout)
    with pytest.raises(wav.AudioError, match="초과"):
        wav.read_audio(BytesIO(payload), wav_only=False)
