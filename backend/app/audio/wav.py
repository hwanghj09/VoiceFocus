from io import BytesIO
from math import gcd
from pathlib import Path
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


def read_audio(
    source: Path | BinaryIO,
    *,
    min_seconds: float = 0.1,
    max_seconds: int = MAX_MIX_SECONDS,
    wav_only: bool = True,
) -> Signal:
    try:
        with sf.SoundFile(source) as audio:
            if wav_only and audio.format not in WAV_FORMATS:
                raise AudioError("WAV 형식의 파일을 올려 주세요.")
            if not 8_000 <= audio.samplerate <= 192_000 or not 1 <= audio.channels <= 8:
                raise AudioError("8–192 kHz, 최대 8채널 오디오를 사용해 주세요.")
            rate = audio.samplerate
            if not min_seconds * rate <= audio.frames <= max_seconds * rate:
                raise AudioError(f"{min_seconds:g}~{max_seconds}초 길이의 오디오를 올려 주세요.")
            frames = audio.read(dtype="float32", always_2d=True)
            if len(frames) != audio.frames:
                raise AudioError("오디오 데이터가 잘렸습니다.")
    except (sf.SoundFileError, OSError) as exc:
        raise AudioError("오디오 파일을 읽을 수 없습니다.") from exc

    if not np.isfinite(frames).all() or np.max(np.abs(frames)) > 1.0:
        raise AudioError("유효한 진폭 범위의 오디오를 올려 주세요.")
    mono = frames.mean(axis=1)
    if rate != SAMPLE_RATE:
        factor = gcd(rate, SAMPLE_RATE)
        mono = resample_poly(mono, SAMPLE_RATE // factor, rate // factor)
    if np.max(np.abs(mono)) < 1e-4:
        raise AudioError("무음 파일입니다. 말소리가 포함된 파일을 올려 주세요.")
    # Resampling can overshoot at transients. Preserve the entire waveform's gain.
    mono /= max(1.0, float(np.max(np.abs(mono))))
    return mono


def write_analysis_wav(path: Path, signal: Signal) -> None:
    sf.write(path, signal, SAMPLE_RATE, subtype="FLOAT")


def playback_wav(signal: Signal) -> bytes:
    peak = float(np.max(np.abs(signal)))
    gain = min(1.0, 0.98 / peak) if peak else 1.0
    buffer = BytesIO()
    sf.write(buffer, signal * gain, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
