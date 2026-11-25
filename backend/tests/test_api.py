import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app import main


def tone(rate=16_000, seconds=1.1):
    time_axis = np.arange(round(rate * seconds)) / rate
    return (0.2 * np.sin(2 * np.pi * 220 * time_axis)).astype(np.float32)


def upload(samples, rate=16_000, format="WAV"):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format=format)
    return ("sample.wav", stream.getvalue(), "audio/wav")


def wait_for_job(client, path):
    for _ in range(100):
        response = client.get(path)
        if response.status_code != 200 or response.json()["status"] in {"completed", "failed"}:
            return response
        time.sleep(0.01)
    pytest.fail("Test worker did not finish.")


def test_mix_resamples_aligns_and_pads():
    a = tone(rate=48_000, seconds=0.3)
    b = tone(seconds=0.5)
    stereo = np.column_stack([a, a])
    with TestClient(main.create_app()) as client:
        response = client.post("/api/v1/mix", files={
            "a": upload(stereo, 48_000),
            "b": upload(b),
        })
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    result, rate = sf.read(io.BytesIO(response.content))
    assert rate == 16_000
    assert result.shape == b.shape
    assert np.max(np.abs(result[:4800])) == pytest.approx(0.4, abs=0.002)
    assert np.allclose(result[4800:], b[4800:], atol=1e-4)


@pytest.mark.parametrize("files", [
    {"a": ("fake.wav", b"not a WAV", "audio/wav"), "b": upload(tone())},
    {"a": upload(tone(), format="FLAC"), "b": upload(tone())},
    {"a": upload(np.zeros(16000)), "b": upload(tone())},
    {"a": upload(tone(seconds=31)), "b": upload(tone())},
    {"a": upload(tone())},
])
def test_mix_rejects_invalid_wav_or_missing_file(files):
    with TestClient(main.create_app()) as client:
        assert client.post("/api/v1/mix", files=files).status_code == 422


def test_limits_multipart_even_without_content_length(monkeypatch):
    monkeypatch.setattr(main, "MAX_REQUEST_BYTES", 1024)
    with TestClient(main.create_app()) as client:
        headers = {"Content-Type": "multipart/form-data; boundary=voicefocus"}
        chunks = iter([
            b'--voicefocus\r\nContent-Disposition: form-data; name="a"; filename="a.wav"\r\n\r\n',
            b"x" * 1024,
            b"\r\n--voicefocus--\r\n",
        ])
        assert client.post("/api/v1/mix", headers=headers, content=chunks).status_code == 413


def test_extraction_job_returns_both_audio_files_and_expires(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", True)
    monotonic = time.monotonic()
    monkeypatch.setattr(main, "monotonic", lambda: monotonic)

    async def fake_infer(_runtime, mixture, enrollment):
        assert len(mixture) == 17603
        assert len(enrollment) == 17600
        return main.playback_wav(mixture), main.playback_wav(mixture * 0.5), 12.5

    monkeypatch.setattr(main, "infer", fake_infer)
    with TestClient(main.create_app()) as client:
        response = client.post("/api/v1/extract", files={
            "mixture": upload(tone(seconds=17603 / 16000)),
            "enrollment": upload(tone()),
        })
        assert response.status_code == 202
        assert response.json()["status"] == "queued"
        path = response.headers["Location"]
        done = wait_for_job(client, path).json()
        assert done["status"] == "completed"
        assert done["inference_ms"] == 12.5
        for name, peak in (("original", 0.2), ("result", 0.1)):
            audio = client.get(f"{path}/{name}")
            assert audio.status_code == 200
            signal, rate = sf.read(io.BytesIO(audio.content))
            assert rate == main.SAMPLE_RATE
            assert len(signal) == 17603
            assert np.max(np.abs(signal)) == pytest.approx(peak, abs=1e-4)
        monotonic += main.RESULT_TTL + 1
        assert client.get(path).status_code == 404
        assert client.get(path + "/result").status_code == 404


def test_failed_inference_has_no_download(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", True)

    async def timeout(*_args):
        raise TimeoutError

    monkeypatch.setattr(main, "infer", timeout)
    with TestClient(main.create_app()) as client:
        response = client.post("/api/v1/extract", files={
            "mixture": upload(tone()),
            "enrollment": upload(tone()),
        })
        path = response.headers["Location"]
        done = wait_for_job(client, path).json()
        assert done["status"] == "failed"
        assert "초과" in done["error"]
        assert client.get(path + "/result").status_code == 409


def test_model_unavailable_is_reported(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", False)
    with TestClient(main.create_app()) as client:
        assert not client.get("/api/v1/models").json()["models"][0]["available"]
        assert client.post("/api/v1/extract").status_code == 503
