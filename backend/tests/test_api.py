import asyncio
import io
import time
from threading import Event

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
    {"a": upload(np.zeros(16000)), "b": upload(tone())},
    {"a": upload(tone(seconds=31)), "b": upload(tone())},
    {"a": upload(tone())},
])
def test_mix_rejects_invalid_audio_or_missing_file(files):
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

    async def fake_infer(_runtime, job, mixture, enrollment):
        assert len(mixture) == 17603
        assert len(enrollment) == 17600
        job.original = main.playback_wav(mixture)
        job.result = main.playback_wav(mixture * 0.5)
        job.inference_ms = 12.5

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
        assert done["mode"] == "speech"
        assert client.get(path + "/residual").status_code == 404
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


def test_delete_job_preserves_active_work_and_removes_completed_audio(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", True)
    finish = Event()

    async def fake_infer(_runtime, job, mixture, _enrollment):
        assert await asyncio.to_thread(finish.wait, 5)
        job.original = job.result = main.playback_wav(mixture)
        job.inference_ms = 1.0

    monkeypatch.setattr(main, "infer", fake_infer)
    with TestClient(main.create_app()) as client:
        cue = upload(tone())
        response = client.post("/api/v1/extract?mode=speech", files={"mixture": cue, "enrollment": cue})
        assert response.status_code == 202
        path = response.headers["Location"]
        try:
            assert client.delete(path).status_code == 409
        finally:
            finish.set()
        assert wait_for_job(client, path).json()["status"] == "completed"
        assert client.get(path + "/result").status_code == 200
        for _ in range(2):
            deletion = client.delete(path)
            assert deletion.status_code == 204
            assert deletion.content == b""
            assert deletion.headers["Cache-Control"] == "no-store"
        for suffix in ("", "/original", "/result", "/residual"):
            assert client.get(path + suffix).status_code == 404


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
        assert client.post("/api/v1/extract?mode=call").status_code == 503


def test_call_jobs_keep_muted_waveform_downloadable(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", True)

    async def fake_infer(_runtime, job, mixture, _enrollment):
        assert job.mode == "call"
        job.target_muted = True
        job.result = main.playback_wav(np.zeros_like(mixture))

    monkeypatch.setattr(main, "infer", fake_infer)
    route = "/api/v1/extract?mode=call"
    with TestClient(main.create_app()) as client:
        cue = upload(tone())
        assert client.post(route, files={"mixture": upload(tone(seconds=0.5)), "enrollment": cue}).status_code == 422
        accepted = client.post(route, files={"mixture": cue, "enrollment": cue})
        assert accepted.status_code == 202
        path = accepted.headers["Location"]
        result = wait_for_job(client, path).json()
        assert result["status"] == "completed" and result["target_muted"] is True
        audio, rate = sf.read(io.BytesIO(client.get(path + "/result").content))
        assert rate == 16_000 and not np.any(audio)


def test_auto_alias_uses_verified_speech_without_classifier(monkeypatch):
    monkeypatch.setattr(main.Runtime, "model_ready", True)

    async def fake_infer(_runtime, job, mixture, _enrollment):
        assert job.mode == "call"
        job.original = job.result = main.playback_wav(mixture)

    monkeypatch.setattr(main, "infer", fake_infer)
    with TestClient(main.create_app()) as client:
        status = client.get("/api/v1/models").json()
        assert status["automatic_available"] is True
        assert len(status["models"]) == 1
        wav = upload(tone())
        accepted = client.post("/api/v1/extract?mode=auto", files={"mixture": wav, "enrollment": wav})
        assert accepted.status_code == 202
        assert wait_for_job(client, accepted.headers["Location"]).json()["mode"] == "call"
        assert client.post("/api/v1/extract?mode=unsupported").status_code == 422


def test_rejects_incorrect_multipart_media_type():
    with TestClient(main.create_app()) as client:
        response = client.post("/api/v1/mix", headers={"Content-Type": "multipart/form-data-extra"})
        assert response.status_code == 415


def test_example_validates_audio_and_does_not_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICEFOCUS_EXAMPLE_DIR", str(tmp_path))
    (tmp_path / "mixed.wav").write_bytes(upload(tone(seconds=0.5))[1])
    (tmp_path / "enroll.wav").write_bytes(upload(tone(seconds=0.5))[1])
    (tmp_path / "enroll_b.wav").write_bytes(b"unreadable")
    with TestClient(main.create_app()) as client:
        audio = client.get("/api/v1/examples/mixed.wav")
        assert audio.status_code == 200
        assert audio.headers["Cache-Control"] == "no-store"
        assert client.get("/api/v1/examples/enroll.wav").status_code == 422
        assert client.get("/api/v1/examples/enroll_b.wav").status_code == 422
        assert client.get("/api/v1/examples/unknown.wav").status_code == 404


def test_queue_recovers_after_a_busy_response(monkeypatch):
    import asyncio
    from threading import Event

    release = Event()
    monkeypatch.setattr(main.Runtime, "model_ready", True)
    monkeypatch.setattr(main, "MAX_ACTIVE_JOBS", 1)

    async def fake_infer(_runtime, job, mixture, _enrollment):
        await asyncio.to_thread(release.wait, 5)
        audio = main.playback_wav(mixture)
        job.original = job.result = audio
        job.inference_ms = 1.0

    monkeypatch.setattr(main, "infer", fake_infer)
    files = {"mixture": upload(tone()), "enrollment": upload(tone())}
    with TestClient(main.create_app()) as client:
        first = client.post("/api/v1/extract", files=files)
        assert first.status_code == 202
        path = first.headers["Location"]
        try:
            assert client.get(path + "/result").status_code == 409
            assert client.post("/api/v1/extract", files=files).status_code == 503
        finally:
            release.set()
        assert wait_for_job(client, path).json()["status"] == "completed"
        second = client.post("/api/v1/extract", files=files)
        assert second.status_code == 202
        assert second.json()["id"] != first.json()["id"]
        assert wait_for_job(client, second.headers["Location"]).json()["status"] == "completed"


