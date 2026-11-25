"""Browser checks against the built frontend and the real local FastAPI worker."""
import importlib
import io
import os
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
si_sdr = importlib.import_module("experiments.03_tse_baseline.evaluate").si_sdr
TRACK_CAPTURE_SCRIPT = """
    const capture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    window.testStreams = [];
    navigator.mediaDevices.getUserMedia = async (...args) => {
        const stream = await capture(...args);
        window.testStreams.push(stream);
        return stream;
    };
"""


@pytest.fixture(scope="session")
def samples():
    path = Path(os.environ.get("VOICEFOCUS_E2E_AUDIO_DIR", ROOT / "artifacts" / "synthetic" / "e02-full")).resolve()
    for name in ("mixed.wav", "target.wav", "interferer.wav", "enroll.wav", "enroll_b.wav"):
        assert (path / name).is_file(), f"Generate the demo fixtures before running browser tests: {path / name}"
    return path


@pytest.fixture(scope="session")
def server(samples, tmp_path_factory):
    assert (ROOT / "frontend" / "dist" / "index.html").is_file(), "Build the frontend before running browser tests."
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    log_path = tmp_path_factory.mktemp("uvicorn") / "output.log"
    env = {**os.environ, "VOICEFOCUS_EXAMPLE_DIR": str(samples), "OPENBLAS_NUM_THREADS": "1"}
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", str(port)],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(80):
                if process.poll() is not None:
                    pytest.fail(log_path.read_text())
                try:
                    response = httpx.get(url + "/api/v1/models", timeout=0.5)
                except httpx.TransportError:
                    time.sleep(0.25)
                    continue
                assert response.json()["models"][0]["available"], "Install the official REAL-TSE checkpoint first."
                assert response.json()["automatic_available"], "Install the automatic audio routing model first."
                break
            else:
                pytest.fail(f"FastAPI did not start.\n{log_path.read_text()}")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture(scope="session")
def browser(samples, tmp_path_factory):
    # Chromium's virtual microphone accepts PCM WAV. Use the same public or
    # synthetic voice as the fixture, keeping real microphones out of the run.
    cue, rate = sf.read(samples / "enroll.wav", dtype="float32")
    path = tmp_path_factory.mktemp("microphone") / "cue.wav"
    sf.write(path, cue, rate, subtype="PCM_16")
    with sync_playwright() as playwright:
        instance = playwright.chromium.launch(
            executable_path=os.environ.get("VOICEFOCUS_CHROMIUM"),
            args=[
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                f"--use-file-for-fake-audio-capture={path}",
            ],
        )
        try:
            yield instance
        finally:
            instance.close()


@pytest.fixture
def page(browser, server):
    context = browser.new_context(viewport={"width": 1320, "height": 1000}, accept_downloads=True)
    tab = context.new_page()
    javascript_errors = []
    tab.on("pageerror", lambda error: javascript_errors.append(str(error)))
    try:
        yield tab
        assert not javascript_errors
    finally:
        context.close()


def submit_and_wait(page, expected_mode="call"):
    with page.expect_response(lambda response: response.url.endswith("/api/v1/extract?mode=auto")) as accepted:
        page.get_by_role("button", name="음성 추출", exact=True).click()
    response = accepted.value
    assert response.status == 202, response.text()
    assert response.json()["mode"] == expected_mode
    expect(page.get_by_role("status")).to_have_text("추출 완료", timeout=250_000 if expected_mode == "harmony" else 130_000)
    job = page.request.get(response.url.split("?", 1)[0].replace("/extract", f"/jobs/{response.json()['id']}")).json()
    assert job["status"] == "completed"
    assert job["processing_ms"] >= job["inference_ms"] > 0
    return job


def test_upload_extract_seek_switch_and_download(page, server, samples, tmp_path):
    page.goto(server)
    expect(page.get_by_label("분리 대상", exact=True)).to_have_count(0)
    expect(page.get_by_text("모델 사용 가능")).to_be_visible()
    page.get_by_label("혼합 음성", exact=True).set_input_files(samples / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files(samples / "enroll.wav")
    job = submit_and_wait(page)

    slider = page.get_by_role("slider", name="공통 재생 위치")
    expect(slider).to_be_enabled()
    width = slider.bounding_box()["width"]
    slider.click(position={"x": width * 0.4, "y": 5})
    original, estimate = (page.locator(".compare-player audio").nth(i) for i in range(2))
    start = original.evaluate("(audio) => audio.currentTime")
    assert 1.0 < start < job["duration_seconds"]
    assert page.locator(".plot-playhead").count() == 4
    expect(page.get_by_role("button", name="원본 혼합 음성", exact=True)).to_have_attribute("aria-pressed", "true")

    page.get_by_role("button", name="비교 재생", exact=True).click()
    expect(page.get_by_role("button", name="비교 재생 일시정지")).to_be_visible()
    before = original.evaluate("(audio) => audio.currentTime")
    page.get_by_role("button", name="추정 음성", exact=True).click()
    expect(page.get_by_role("button", name="추정 음성", exact=True)).to_have_attribute("aria-pressed", "true")
    assert original.evaluate("(audio) => audio.paused")
    assert not estimate.evaluate("(audio) => audio.paused")
    after = estimate.evaluate("(audio) => audio.currentTime")
    assert before - 0.05 <= after < before + 0.6
    page.get_by_role("button", name="비교 재생 일시정지").click()

    for label, filename in (("원본 혼합 음성", "mixed.wav"), ("추정 음성", "voicefocus-extracted.wav")):
        expect(page.get_by_role("img", name=f"{label} 파형")).to_be_visible()
        page.wait_for_function(
            """name => [...document.querySelectorAll('canvas')].some(canvas =>
                canvas.getAttribute('aria-label') === name
                && canvas.getContext('2d').getImageData(0, 0, 1, 1).data[3] === 255)""",
            arg=f"{label} 스펙트로그램",
        )
        with page.expect_download() as download:
            page.get_by_role("link", name=f"{label} WAV 다운로드").click()
        assert download.value.suggested_filename == filename
        download.value.save_as(tmp_path / filename)

    assert (tmp_path / "mixed.wav").read_bytes() == (samples / "mixed.wav").read_bytes()
    extracted, rate = sf.read(tmp_path / "voicefocus-extracted.wav", dtype="float32")
    assert rate == 16000 and extracted.ndim == 1 and len(extracted) / rate == job["duration_seconds"]
    target, mixture = (sf.read(samples / name, dtype="float32")[0] for name in ("target.wav", "mixed.wav"))
    assert si_sdr(target, extracted) - si_sdr(target, mixture) > 3

    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") == 390
    page.get_by_role("button", name="비교 재생", exact=True).click()
    page.get_by_role("navigation").get_by_role("button", name="WAV 혼합").click()
    assert estimate.evaluate("(audio) => audio.paused")


def test_microphone_recording_generates_wav_and_extracts(page, server, samples, tmp_path):
    page.add_init_script(TRACK_CAPTURE_SCRIPT)
    page.goto(server)
    page.get_by_label("혼합 음성", exact=True).set_input_files(samples / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files({
        "name": "wrong-format.mp3", "mimeType": "audio/mpeg", "buffer": b"invalid audio",
    })
    expect(page.get_by_role("alert")).to_contain_text("WAV 파일을 선택해 주세요")
    page.get_by_role("button", name="등록 음성 녹음", exact=True).click()
    expect(page.get_by_role("button", name="녹음 종료")).to_be_visible()
    expect(page.get_by_role("button", name="음성 추출", exact=True)).to_be_disabled()
    expect(page.get_by_text("녹음 중 · 2 / 20초")).to_be_visible()
    page.get_by_role("button", name="녹음 종료").click()
    expect(page.get_by_text("voicefocus-enrollment.wav", exact=True)).to_be_visible()
    expect(page.get_by_role("alert")).to_have_count(0)
    assert page.evaluate("window.testStreams.length === 1 && !window.testStreams[0].active")

    submit_and_wait(page)
    with page.expect_download() as download:
        page.get_by_role("link", name="추정 음성 WAV 다운로드").click()
    result = tmp_path / "result.wav"
    download.value.save_as(result)
    with wave.open(str(result)) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1
        assert wav.getnframes() > 0


def test_microphone_permission_denied_keeps_file_upload_available(page, server):
    page.add_init_script("""
        navigator.mediaDevices.getUserMedia = async () => {
            throw new DOMException('User denied', 'NotAllowedError');
        };
    """)
    page.goto(server)
    page.get_by_role("button", name="등록 음성 녹음", exact=True).click()
    expect(page.get_by_role("alert")).to_have_text("브라우저에서 마이크 권한을 허용해 주세요.")
    expect(page.get_by_role("button", name="등록 음성 파일 선택")).to_be_enabled()


def test_cancelled_permission_releases_late_microphone(page, server):
    page.add_init_script("""
        const capture = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
        navigator.mediaDevices.getUserMedia = async (...args) => {
            await new Promise(resolve => { window.releaseMic = resolve; });
            const stream = await capture(...args);
            window.lateMic = stream;
            return stream;
        };
    """)
    page.goto(server)
    page.get_by_role("button", name="등록 음성 녹음").click()
    expect(page.get_by_text("마이크 권한 확인 중")).to_be_visible()
    page.get_by_role("button", name="녹음 취소").click()
    page.evaluate("window.releaseMic()")
    page.wait_for_function("window.lateMic && !window.lateMic.active")
    expect(page.get_by_role("button", name="등록 음성 파일 선택")).to_be_enabled()
    expect(page.get_by_text("voicefocus-enrollment.wav")).to_have_count(0)


def test_moving_away_discards_active_recording_and_releases_microphone(page, server):
    page.add_init_script(TRACK_CAPTURE_SCRIPT)
    page.goto(server)
    page.get_by_role("button", name="등록 음성 녹음").click()
    expect(page.get_by_role("button", name="녹음 종료")).to_be_visible()
    page.get_by_role("navigation").get_by_role("button", name="WAV 혼합").click()
    page.wait_for_function("window.testStreams.length === 1 && !window.testStreams[0].active")
    page.get_by_role("navigation").get_by_role("button", name="화자 추출").click()
    expect(page.get_by_role("button", name="등록 음성 녹음")).to_be_visible()
    expect(page.get_by_text("voicefocus-enrollment.wav")).to_have_count(0)


def test_short_recording_is_rejected_and_releases_microphone(page, server):
    page.add_init_script(TRACK_CAPTURE_SCRIPT)
    page.goto(server)
    page.get_by_role("button", name="등록 음성 녹음").click()
    expect(page.get_by_role("button", name="녹음 종료")).to_be_visible()
    # Give Chromium's recorder enough media to produce a decodable container.
    page.wait_for_timeout(300)
    page.get_by_role("button", name="녹음 종료").click()
    expect(page.get_by_role("alert")).to_contain_text("1초 이상 녹음해 주세요")
    assert page.evaluate("!window.testStreams[0].active")
    expect(page.get_by_role("button", name="등록 음성 녹음")).to_be_enabled()


def test_mix_invalid_audio_recovers_and_transfers(page, server, samples):
    page.goto(server)
    page.get_by_role("navigation").get_by_role("button", name="WAV 혼합").click()
    page.get_by_label("화자 A 음성", exact=True).set_input_files({
        "name": "broken.wav", "mimeType": "audio/wav", "buffer": b"invalid WAV",
    })
    page.get_by_label("화자 B 음성", exact=True).set_input_files(samples / "enroll_b.wav")
    mix = page.get_by_role("main").get_by_role("button", name="WAV 혼합", exact=True)
    mix.click()
    expect(page.get_by_role("alert")).to_contain_text("오디오 파일을 읽을 수 없습니다")
    page.get_by_label("화자 A 음성", exact=True).set_input_files(samples / "enroll.wav")
    mix.click()
    page.get_by_role("button", name="추출 입력에 사용").click()
    expect(page.get_by_text("voicefocus-mixed.wav", exact=True)).to_be_visible()
    page.get_by_label("예제 목표 화자").select_option("B")
    page.get_by_role("button", name="예제 불러오기").click()
    expect(page.get_by_role("region", name="입력 설정").get_by_text("enroll_b.wav", exact=True)).to_be_visible()

    job = submit_and_wait(page)
    response = page.request.get(f"{server}/api/v1/jobs/{job['id']}/result")
    assert response.status == 200
    extracted, _ = sf.read(io.BytesIO(response.body()), dtype="float32")
    target, mixture = (sf.read(samples / name, dtype="float32")[0] for name in ("interferer.wav", "mixed.wav"))
    assert si_sdr(target, extracted) - si_sdr(target, mixture) > 3


def test_harmony_removes_enrolled_tenor_and_can_swap_outputs(page, server, tmp_path):
    case_dir = os.environ.get("VOICEFOCUS_E2E_HARMONY_DIR")
    if not case_dir:
        pytest.skip("Set VOICEFOCUS_E2E_HARMONY_DIR to a measured Niño Dios fixture.")
    case = Path(case_dir)
    page.goto(server)
    expect(page.get_by_text("모델 사용 가능")).to_be_visible()
    page.get_by_label("혼합 음성", exact=True).set_input_files(case / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files(case / "enroll_b.wav")
    job = submit_and_wait(page, expected_mode="harmony")
    assert job["mode"] == "harmony"

    removed = page.get_by_role("button", name="목표 제거 후", exact=True)
    removed.click()
    expect(removed).to_have_attribute("aria-pressed", "true")
    link = page.get_by_role("link", name="목표 제거 후 WAV 다운로드")
    with page.expect_download() as download:
        link.click()
    assert download.value.suggested_filename == "voicefocus-target-removed.wav"
    result_path = tmp_path / "removed.wav"
    download.value.save_as(result_path)
    retained, rate = sf.read(result_path, dtype="float32")
    reference = sf.read(case / "reference-a.wav", dtype="float32")[0]
    mixture = sf.read(case / "mixed.wav", dtype="float32")[0]
    assert rate == 16000 and retained.shape == mixture.shape
    assert si_sdr(reference, retained) - si_sdr(reference, mixture) > 10

    path = server + f"/api/v1/jobs/{job['id']}"
    target = page.request.get(path + "/result").body()
    original = page.request.get(path + "/original").body()
    estimate = sf.read(io.BytesIO(target))[0]
    original_signal = sf.read(io.BytesIO(original))[0]
    np.testing.assert_allclose(estimate + retained, original_signal, atol=2 / 32768, rtol=0)

    page.get_by_role("button", name="제거 대상 바꾸기").click()
    expect(page.get_by_role("button", name="목표 목소리만", exact=True)).to_have_attribute("aria-pressed", "true")
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") == 390
    with page.expect_download() as swapped:
        link.click()
    second = tmp_path / "swapped.wav"
    swapped.value.save_as(second)
    assert second.read_bytes() == target


def test_harmony_two_references_disambiguate_singers_in_unison(page, server, samples, tmp_path):
    case_dir = os.environ.get("VOICEFOCUS_E2E_ACCURACY_DIR")
    if not case_dir:
        pytest.skip("Set VOICEFOCUS_E2E_ACCURACY_DIR to the Rossinyol unison fixture.")
    case = Path(case_dir)
    page.goto(server)
    page.get_by_label("혼합 음성", exact=True).set_input_files(case / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files(case / "enroll-a.wav")
    page.get_by_text("남길 가수도 등록").click()
    page.get_by_label("남길 가수 음성", exact=True).set_input_files(case / "enroll-b.wav")
    expect(page.get_by_text("enroll-b.wav", exact=True)).to_be_visible()
    page.set_viewport_size({"width": 390, "height": 844})
    assert page.evaluate("document.documentElement.scrollWidth") == 390
    submit_and_wait(page, expected_mode="harmony")

    with page.expect_download() as download:
        page.get_by_role("link", name="목표 제거 후 WAV 다운로드").click()
    result_path = tmp_path / "two-references.wav"
    download.value.save_as(result_path)
    result, rate = sf.read(result_path, dtype="float32")
    retained, mixture, one_reference = (
        sf.read(case / name, dtype="float32")[0]
        for name in ("reference-b.wav", "mixed.wav", "remove-a-one.wav")
    )
    assert rate == 16000 and result.shape == mixture.shape
    assert si_sdr(retained, result) - si_sdr(retained, mixture) > 5
    assert si_sdr(retained, result) - si_sdr(retained, one_reference) > 10

    page.get_by_label("혼합 음성", exact=True).set_input_files(samples / "mixed.wav")
    expect(page.get_by_role("button", name="남길 가수 음성 파일 선택")).to_be_visible()


def test_call_with_the_enrolled_speaker_absent_downloads_silence(page, server, tmp_path):
    case_dir = os.environ.get("VOICEFOCUS_E2E_CALL_DIR")
    if not case_dir:
        pytest.skip("Set VOICEFOCUS_E2E_CALL_DIR to the target-absent call fixture.")
    case = Path(case_dir)
    page.goto(server)
    page.get_by_label("혼합 음성", exact=True).set_input_files(case / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files(case / "enroll.wav")
    job = submit_and_wait(page)
    assert job["mode"] == "call" and job["target_muted"] is True
    expect(page.get_by_text("등록한 화자를 확인하지 못해 추정 음성을 음소거했습니다.")).to_be_visible()
    with page.expect_download() as download:
        page.get_by_role("link", name="추정 음성 WAV 다운로드").click()
    result = tmp_path / "muted.wav"
    download.value.save_as(result)
    audio, rate = sf.read(result)
    assert rate == 16000 and len(audio) / rate == job["duration_seconds"]
    assert not np.any(audio)


def test_call_recovers_enrolled_speaker_in_narrowband_overlap(page, server):
    case_dir = os.environ.get("VOICEFOCUS_E2E_CALL_OVERLAP_DIR")
    if not case_dir:
        pytest.skip("Set VOICEFOCUS_E2E_CALL_OVERLAP_DIR to the narrowband overlap fixture.")
    case = Path(case_dir)
    page.goto(server)
    page.get_by_label("혼합 음성", exact=True).set_input_files(case / "mixed.wav")
    page.get_by_label("등록 음성", exact=True).set_input_files(case / "enroll_b.wav")
    job = submit_and_wait(page)
    assert job["target_muted"] is False
    response = page.request.get(f"{server}/api/v1/jobs/{job['id']}/result")
    assert response.status == 200
    output, rate = sf.read(io.BytesIO(response.body()), dtype="float32")
    reference = sf.read(case / "reference-b.wav", dtype="float32")[0]
    raw = sf.read(case / "raw-b.wav", dtype="float32")[0]
    assert rate == 16000 and output.shape == reference.shape
    assert si_sdr(reference, output) - si_sdr(reference, raw) > 15
