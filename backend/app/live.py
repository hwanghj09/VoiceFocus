import asyncio
import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from backend.app.audio.wav import AudioError, MAX_ENROLL_SECONDS, MAX_FILE_BYTES, SAMPLE_RATE, read_audio

CHUNK = 2048
log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def live_extractor(model_dir: Path, device: str):
    from backend.app.models.worker import Extractor

    return Extractor(model_dir, device)


def load_stream(model_dir: Path, device: str, enrollment):
    from backend.app.models.live import VerifiedStream

    return VerifiedStream(live_extractor(model_dir, device), enrollment)


def is_same_origin(socket: WebSocket) -> bool:
    origin = socket.headers.get("origin")
    try:
        return not origin or urlsplit(origin).netloc == socket.headers.get("host")
    except ValueError:
        return False


async def serve_live(socket: WebSocket, model_dir: Path, device: str, available: bool, slot: asyncio.Lock) -> None:
    if not is_same_origin(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    if not available:
        await socket.send_json({"error": "실시간 모델을 설치해 주세요."})
        await socket.close(code=1013)
        return
    try:
        async with asyncio.timeout(1):
            await slot.acquire()
    except TimeoutError:
        try:
            await socket.send_json({"error": "다른 실시간 음성 처리가 실행 중입니다."})
            await socket.close(code=1013)
        except WebSocketDisconnect:
            pass
        return

    try:
        first = await asyncio.wait_for(socket.receive(), 10)
        if first["type"] == "websocket.disconnect":
            return
        wav = first.get("bytes")
        if wav is None or len(wav) > MAX_FILE_BYTES:
            raise AudioError("최대 16 MiB의 등록 WAV가 필요합니다.")
        enrollment = await asyncio.to_thread(
            read_audio, BytesIO(wav), min_seconds=1, max_seconds=MAX_ENROLL_SECONDS,
        )
        stream = await asyncio.to_thread(load_stream, model_dir, device, enrollment)
        await socket.send_json({"ready": True, "sample_rate": SAMPLE_RATE, "chunk": CHUNK})
        receive_timeout = 30
        while True:
            packet = await asyncio.wait_for(socket.receive(), receive_timeout)
            receive_timeout = 10
            if packet["type"] == "websocket.disconnect":
                return
            start = perf_counter()
            if packet.get("text") == "end":
                tail = await asyncio.to_thread(stream.finish)
                if len(tail):
                    await socket.send_bytes(np.asarray(tail, dtype="<f4").tobytes())
                await socket.close()
                return
            binary = packet.get("bytes")
            if binary is None or len(binary) != CHUNK * 4:
                raise AudioError("마이크 오디오 형식이 올바르지 않습니다.")
            pcm = np.frombuffer(binary, dtype="<f4").copy()
            if not np.isfinite(pcm).all() or np.max(np.abs(pcm)) > 1.0:
                raise AudioError("마이크 오디오 샘플이 올바르지 않습니다.")
            output = await asyncio.to_thread(stream.push, pcm)
            elapsed = (perf_counter() - start) * 1000
            if len(output):
                await socket.send_bytes(np.asarray(output, dtype="<f4").tobytes())
            await socket.send_json({"processing_ms": round(elapsed, 1)})
    except WebSocketDisconnect:
        return
    except (AudioError, TimeoutError) as error:
        await socket.send_json({"error": str(error) or "마이크 입력이 중단되었습니다."})
        await socket.close(code=1008)
    except Exception:
        log.exception("VoiceFocus live inference failed")
        await socket.send_json({"error": "실시간 음성을 처리할 수 없습니다."})
        await socket.close(code=1011)
    finally:
        slot.release()
