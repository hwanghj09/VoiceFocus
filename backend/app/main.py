import asyncio
import importlib.util
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from backend.app.audio.wav import (
    MAX_ENROLL_SECONDS,
    MAX_FILE_BYTES,
    MAX_MIX_SECONDS,
    SAMPLE_RATE,
    AudioError,
    Signal,
    playback_wav,
    read_audio,
    write_analysis_wav,
)
from backend.app.models.realtse import (
    DEFAULT_MODEL_DIR,
    MODEL_ID,
    MODEL_NAME,
    MODEL_VERSION,
    ROOT,
    VENDOR_DIR,
    checkpoint_exists,
)
from backend.app.live import serve_live
from backend.app.calls import Room, serve_call

log = logging.getLogger(__name__)
MAX_REQUEST_BYTES = 2 * MAX_FILE_BYTES + 64 * 1024
MAX_ACTIVE_JOBS = 4
MAX_JOBS = 32
RESULT_TTL = 15 * 60
EXTRACTION_TIMEOUT = 120
NO_STORE = {"Cache-Control": "no-store"}
EXAMPLE_FILES = {"mixed.wav", "enroll.wav", "enroll_b.wav"}
Mode = Literal["speech", "call"]
RequestedMode = Literal["auto", "speech", "call"]


@dataclass
class Job:
    mode: Mode
    model_id: str
    model_version: str
    status: Literal["queued", "processing", "completed", "failed"] = "queued"
    error: str | None = None
    processing_ms: int | None = None
    inference_ms: float | None = None
    duration_seconds: float = 0
    original: bytes | None = None
    result: bytes | None = None
    target_muted: bool | None = None
    expires_at: float = float("inf")

    def public(self, job_id: UUID) -> dict:
        return {
            "id": str(job_id),
            "status": self.status,
            "error": self.error,
            "processing_ms": self.processing_ms,
            "inference_ms": self.inference_ms,
            "duration_seconds": self.duration_seconds,
            "mode": self.mode,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "target_muted": self.target_muted,
            "sample_rate": SAMPLE_RATE,
        }


@dataclass
class Runtime:
    model_dir: Path
    device: str
    example_dir: Path
    jobs: dict[UUID, Job] = field(default_factory=dict)
    tasks: set[asyncio.Task] = field(default_factory=set)
    worker: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def model_ready(self) -> bool:
        return (
            checkpoint_exists(self.model_dir)
            and (VENDOR_DIR / "wesep-real-tse" / "wesep" / "__init__.py").is_file()
            and (VENDOR_DIR / "wespeaker" / "wespeaker" / "__init__.py").is_file()
            and importlib.util.find_spec("torch") is not None
        )

    @property
    def model_version(self) -> str:
        return self.model_dir.name if self.model_dir.name.startswith("spk_emb_") else MODEL_VERSION

    def expire(self) -> None:
        now = monotonic()
        for job_id in [job_id for job_id, job in self.jobs.items() if job.expires_at <= now]:
            del self.jobs[job_id]


async def infer(
    runtime: Runtime, job: Job, mixture: Signal, enrollment: Signal,
) -> None:
    with TemporaryDirectory(prefix="voicefocus-") as temporary:
        work_dir = Path(temporary)
        mix, enroll, output, metrics = (
            work_dir / name for name in ("mixed.wav", "enroll.wav", "result.wav", "metrics.json")
        )
        write_analysis_wav(mix, mixture)
        write_analysis_wav(enroll, enrollment)
        mode_args = ["--model", str(runtime.model_dir)]
        if job.mode == "call":
            mode_args.append("--verify-target")
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "backend.app.models.worker", "--device", runtime.device,
            "--mixture", str(mix), "--enroll", str(enroll),
            "--output", str(output), "--metrics", str(metrics), *mode_args,
            cwd=ROOT, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=EXTRACTION_TIMEOUT)
        finally:
            if process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()

        if process.returncode != 0:
            log.error("VoiceFocus %s worker exited with %s: %s", job.mode, process.returncode, stderr.decode(errors="replace")[-4000:])
            raise RuntimeError("VoiceFocus inference failed.")
        signal, rate = sf.read(output, dtype="float32")
        if rate != SAMPLE_RATE or signal.shape != mixture.shape or not np.isfinite(signal).all():
            raise ValueError("Invalid result waveform.")
        metadata = json.loads(metrics.read_text(encoding="utf-8"))
        job.original, job.result = playback_wav(mixture), playback_wav(signal)
        job.inference_ms = round(metadata["inference_ms"], 1)
        if job.mode == "call":
            job.target_muted = bool(metadata["target_muted"])


async def run_job(
    runtime: Runtime, job: Job, mixture: Signal, enrollment: Signal,
) -> None:
    async with runtime.worker:
        job.status = "processing"
        start = monotonic()
        try:
            await infer(runtime, job, mixture, enrollment)
            job.status = "completed"
        except TimeoutError:
            job.error = "처리 시간이 초과되었습니다. 더 짧은 혼합 음성을 사용해 주세요."
            job.status = "failed"
        except Exception:
            log.exception("VoiceFocus extraction failed")
            job.error = "화자 추출에 실패했습니다. 서버 모델 설정과 로그를 확인해 주세요."
            job.status = "failed"
        finally:
            job.processing_ms = round((monotonic() - start) * 1000)
            job.expires_at = monotonic() + RESULT_TTL


async def read_uploads(
    request: Request, limits: dict[str, tuple[float, int]],
) -> dict[str, Signal]:
    request_limit = MAX_REQUEST_BYTES

    async def limited_body():
        seen = 0
        async for chunk in request.stream():
            seen += len(chunk)
            if seen > request_limit:
                raise MultiPartException("Request is too large.")
            yield chunk

    length = request.headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > request_limit):
        raise HTTPException(413, "업로드 용량을 초과했습니다.")
    media_type = request.headers.get("content-type", "").partition(";")[0].strip().lower()
    if media_type != "multipart/form-data":
        raise HTTPException(415, "오디오 파일을 multipart/form-data로 올려 주세요.")
    parser = MultiPartParser(request.headers, limited_body(), max_files=len(limits), max_fields=0)
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        status = 413 if exc.message == "Request is too large." else 400
        raise HTTPException(status, "오디오 파일과 업로드 용량을 확인해 주세요.") from exc

    try:
        names = [name for name, _ in form.multi_items()]
        if set(names) - limits.keys() or len(set(names)) != len(names):
            raise HTTPException(422, "올린 오디오 파일의 입력 항목을 확인해 주세요.")
        signals = {}
        for name, (minimum, maximum) in limits.items():
            upload = form.get(name)
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, f"{name} 오디오 파일이 필요합니다.")
            if upload.size is not None and upload.size > MAX_FILE_BYTES:
                raise HTTPException(413, "파일당 최대 16 MiB까지 올릴 수 있습니다.")
            try:
                signal = await asyncio.to_thread(
                    read_audio, upload.file, min_seconds=minimum, max_seconds=maximum, wav_only=False,
                )
            except AudioError as exc:
                raise HTTPException(422, f"{name}: {exc}") from exc
            signals[name] = signal
        return signals
    finally:
        await form.close()


async def cleanup(runtime: Runtime) -> None:
    while True:
        await asyncio.sleep(10)
        runtime.expire()


def create_app() -> FastAPI:
    runtime = Runtime(
        model_dir=Path(os.environ.get("VOICEFOCUS_MODEL_DIR", DEFAULT_MODEL_DIR)).resolve(),
        device=os.environ.get("VOICEFOCUS_DEVICE", "cpu"),
        example_dir=Path(os.environ.get("VOICEFOCUS_EXAMPLE_DIR", ROOT / "artifacts" / "synthetic" / "e02-full")).resolve(),
    )
    uploads = asyncio.Semaphore(2)
    live_slot = asyncio.Lock()
    rooms: dict[str, Room] = {}
    live_model = DEFAULT_MODEL_DIR.with_name("spk_emb_causal_100").resolve()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        ticker = asyncio.create_task(cleanup(runtime))
        try:
            yield
        finally:
            tasks = [ticker, *runtime.tasks]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="VoiceFocus", version="0.1.0", lifespan=lifespan)

    @app.get("/healthz", include_in_schema=False)
    async def health() -> Response:
        return Response("ok\n", media_type="text/plain", headers=NO_STORE)

    @app.websocket("/ws/v1/stream")
    async def live(socket: WebSocket) -> None:
        await serve_live(socket, live_model, runtime.device, checkpoint_exists(live_model), live_slot)

    @app.websocket("/ws/v1/calls")
    async def calls(socket: WebSocket) -> None:
        await serve_call(socket, rooms)

    @app.get("/api/v1/models")
    async def models(response: Response) -> dict:
        response.headers.update(NO_STORE)
        return {
            "models": [
                {"id": MODEL_ID, "name": MODEL_NAME, "version": runtime.model_version,
                 "sample_rate": SAMPLE_RATE, "available": runtime.model_ready},
            ],
            "example_available": all((runtime.example_dir / name).is_file() for name in EXAMPLE_FILES),
            "automatic_available": runtime.model_ready,
            "live_available": checkpoint_exists(live_model),
        }

    @app.get("/api/v1/examples/{name}")
    async def example(name: str) -> Response:
        if name not in EXAMPLE_FILES or not (runtime.example_dir / name).is_file():
            raise HTTPException(404, "예제 음성을 찾을 수 없습니다.")
        path = runtime.example_dir / name
        minimum, maximum = (0.1, MAX_MIX_SECONDS) if name == "mixed.wav" else (1.0, MAX_ENROLL_SECONDS)
        try:
            signal = await asyncio.to_thread(read_audio, path, min_seconds=minimum, max_seconds=maximum)
        except AudioError as exc:
            log.warning("Invalid VoiceFocus example %s: %s", name, exc)
            raise HTTPException(422, "예제 WAV를 읽을 수 없습니다.") from exc
        return Response(playback_wav(signal), media_type="audio/wav", headers=NO_STORE)

    @app.post("/api/v1/mix")
    async def mix(request: Request) -> Response:
        if uploads.locked():
            raise HTTPException(503, "다른 파일을 읽고 있습니다. 잠시 후 다시 시도해 주세요.")
        async with uploads:
            audio = await read_uploads(request, {
                "a": (0.1, MAX_MIX_SECONDS), "b": (0.1, MAX_MIX_SECONDS),
            })
            a, b = audio["a"], audio["b"]
            combined = np.zeros(max(len(a), len(b)), dtype=np.float32)
            combined[:len(a)] += a
            combined[:len(b)] += b
            return wav_response(playback_wav(combined), "voicefocus-mixed.wav")

    @app.post("/api/v1/extract", status_code=202)
    async def extract(request: Request, response: Response, mode: RequestedMode = "speech") -> dict:
        # Legacy auto calls use verified speech extraction directly.
        if mode == "auto":
            mode = "call"
        if not runtime.model_ready:
            raise HTTPException(503, "REAL-TSE 모델을 설치한 뒤 다시 시도해 주세요.")
        if len(runtime.tasks) >= MAX_ACTIVE_JOBS or uploads.locked():
            raise HTTPException(503, "처리 대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
        async with uploads:
            limits = {
                "mixture": (0.1 if mode == "speech" else 1.0, MAX_MIX_SECONDS),
                "enrollment": (1.0, MAX_ENROLL_SECONDS),
            }
            audio = await read_uploads(request, limits)
            mixture, enrollment = audio["mixture"], audio["enrollment"]
            runtime.expire()
            if len(runtime.tasks) >= MAX_ACTIVE_JOBS:
                raise HTTPException(503, "처리 대기열이 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
            if len(runtime.jobs) >= MAX_JOBS:
                raise HTTPException(503, "임시 결과 저장소가 가득 찼습니다. 잠시 후 다시 시도해 주세요.")
            model = (MODEL_ID, runtime.model_version)
            if mode == "call":
                model = (MODEL_ID, f"{runtime.model_version}-verified")
            job_id = uuid4()
            job = Job(mode, *model, duration_seconds=len(mixture) / SAMPLE_RATE)
            runtime.jobs[job_id] = job
            task = asyncio.create_task(run_job(runtime, job, mixture, enrollment))
            runtime.tasks.add(task)
            task.add_done_callback(runtime.tasks.discard)
        response.headers.update(NO_STORE)
        response.headers["Location"] = f"/api/v1/jobs/{job_id}"
        return job.public(job_id)

    def find_job(job_id: UUID) -> Job:
        runtime.expire()
        job = runtime.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "작업이 없거나 15분 보관 시간이 지났습니다.")
        return job

    @app.get("/api/v1/jobs/{job_id}")
    async def status(job_id: UUID, response: Response) -> dict:
        response.headers.update(NO_STORE)
        return find_job(job_id).public(job_id)

    @app.delete("/api/v1/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: UUID) -> Response:
        runtime.expire()
        job = runtime.jobs.get(job_id)
        if job is not None:
            if job.status in ("queued", "processing"):
                raise HTTPException(409, "처리가 끝난 뒤 결과를 삭제할 수 있습니다.", headers=NO_STORE)
            del runtime.jobs[job_id]
        return Response(status_code=204, headers=NO_STORE)

    @app.get("/api/v1/jobs/{job_id}/result")
    async def result(job_id: UUID) -> Response:
        job = find_job(job_id)
        if job.status != "completed" or job.result is None:
            raise HTTPException(409, "다운로드할 추출 결과가 없습니다.")
        return wav_response(job.result, "voicefocus-extracted.wav")

    @app.get("/api/v1/jobs/{job_id}/original")
    async def original(job_id: UUID) -> Response:
        job = find_job(job_id)
        if job.status != "completed" or job.original is None:
            raise HTTPException(409, "비교 재생용 혼합 음성이 없습니다.")
        return wav_response(job.original, "voicefocus-input-16khz.wav")

    dist = ROOT / "frontend" / "dist"
    if (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

    return app


def wav_response(blob: bytes, filename: str) -> Response:
    return Response(blob, media_type="audio/wav", headers={
        **NO_STORE, "Content-Disposition": f'attachment; filename="{filename}"',
    })


app = create_app()
