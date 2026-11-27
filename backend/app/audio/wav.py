from io import BytesIO
from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import imageio_ffmpeg
from typing import BinaryIO

import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000
MAX_MIX_SECONDS = 30
MAX_ENROLL_SECONDS = 20
MAX_FILE_BYTES = 16 * 1024 * 1024
WAV_FORMATS = {"WAV", "WAVEX", "RF64"}
Signal = NDArray[np.float32]


class AudioError(ValueError):
    pass


# FFmpeg probes the bytes, not the supplied extension or MIME type.
# Restrict demuxers and protocols; never pass the user's filename to FFmpeg.
MEDIA_FORMATS = "wav,mp3,mov,mp4,m4a,3gp,3g2,mj2,aac,matroska,webm,flac,ogg"
DECODE_TIMEOUT = 20


def ffmpeg_executable() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_media(source: Path | BinaryIO, min_seconds: float, max_seconds: int) -> Signal:
    # A seekable local file is needed for MP4 files with metadata at the end.
    # Limit encoded bytes before decoding and decoded duration even for streams
    # that do not advertise a duration.
    try:
        if isinstance(source, Path):
            with source.open("rb") as stream:
                payload = stream.read(MAX_FILE_BYTES + 1)
        else:
            source.seek(0)
            payload = source.read(MAX_FILE_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise AudioError("오디오 파일을 읽을 수 없습니다.") from exc
    if not payload or len(payload) > MAX_FILE_BYTES:
        raise AudioError("파일당 최대 16 MiB까지 올릴 수 있습니다.")
    try:
        executable = ffmpeg_executable()
    except (RuntimeError, OSError) as exc:
        raise AudioError("FFmpeg 변환기를 사용할 수 없습니다. 서버 설치를 확인해 주세요.") from exc
    try:
        with TemporaryDirectory(prefix="voicefocus-decode-") as temporary:
            input_path = Path(temporary) / "input.bin"
            input_path.write_bytes(payload)
            result = subprocess.run(
                [executable, "-nostdin", "-hide_banner", "-loglevel", "error",
                 "-xerror", "-protocol_whitelist", "file,pipe",
                 "-format_whitelist", MEDIA_FORMATS, "-i", str(input_path),
                 "-map", "0:a:0", "-vn", "-sn", "-dn",
                 "-t", str(max_seconds + 1), "-ac", "1", "-ar", str(SAMPLE_RATE),
                 "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=DECODE_TIMEOUT, check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise AudioError("오디오 변환 시간이 초과되었습니다.") from exc
    except OSError as exc:
        raise AudioError("FFmpeg 변환기를 실행할 수 없습니다.") from exc
    if result.returncode or not result.stdout or len(result.stdout) % 4:
        raise AudioError("오디오 파일을 읽을 수 없습니다. 지원 형식과 파일 손상 여부를 확인해 주세요.")
    mono = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if not min_seconds * SAMPLE_RATE <= len(mono) <= max_seconds * SAMPLE_RATE:
        raise AudioError(f"{min_seconds:g}~{max_seconds}초 길이의 오디오를 올려 주세요.")
    if not np.isfinite(mono).all():
        raise AudioError("유효한 진폭 범위의 오디오를 올려 주세요.")
    # Lossy codecs/resampling can overshoot; normalize without hard clipping.
    return finish_mono(mono)


def finish_mono(mono: Signal) -> Signal:
    peak = float(np.max(np.abs(mono)))
    if peak < 1e-4:
        raise AudioError("무음 파일입니다. 말소리가 포함된 파일을 올려 주세요.")
    mono /= max(1.0, peak)
    return mono.astype(np.float32, copy=False)


class _NeedsDecode(Exception):
    pass


def read_audio(
    source: Path | BinaryIO,
    *,
    min_seconds: float = 0.1,
    max_seconds: int = MAX_MIX_SECONDS,
    wav_only: bool = True,
) -> Signal:
    try:
        with sf.SoundFile(source) as audio:
            if audio.format not in WAV_FORMATS:
                if wav_only:
                    raise AudioError("WAV 형식의 파일을 올려 주세요.")
                # Close the SoundFile handle before rewinding its input.
                raise _NeedsDecode()
            if not 8_000 <= audio.samplerate <= 192_000 or not 1 <= audio.channels <= 8:
                raise AudioError("8–192 kHz, 최대 8채널 오디오를 사용해 주세요.")
            rate = audio.samplerate
            if not min_seconds * rate <= audio.frames <= max_seconds * rate:
                raise AudioError(f"{min_seconds:g}~{max_seconds}초 길이의 오디오를 올려 주세요.")
            frames = audio.read(dtype="float32", always_2d=True)
            if len(frames) != audio.frames:
                raise AudioError("오디오 데이터가 잘렸습니다.")
    except _NeedsDecode:
        return decode_media(source, min_seconds, max_seconds)
    except (sf.SoundFileError, OSError) as exc:
        if not wav_only:
            return decode_media(source, min_seconds, max_seconds)
        raise AudioError("오디오 파일을 읽을 수 없습니다.") from exc

    if not np.isfinite(frames).all() or np.max(np.abs(frames)) > 1.0:
        raise AudioError("유효한 진폭 범위의 오디오를 올려 주세요.")
    mono = frames.mean(axis=1)
    if rate != SAMPLE_RATE:
        factor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, rate // factor)
    return finish_mono(mono)


def write_analysis_wav(path: Path, signal: Signal) -> None:
    sf.write(path, signal, SAMPLE_RATE, subtype="FLOAT")


def playback_wav(signal: Signal) -> bytes:
    peak = float(np.max(np.abs(signal)))
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    buffer = BytesIO()
    sf.write(buffer, signal * gain, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
