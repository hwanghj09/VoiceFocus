"""Choose the two-singer separator only with vocal music in both inputs."""

import importlib.util
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from backend.app.audio.wav import SAMPLE_RATE, Signal
from backend.app.models.realtse import ROOT

WEIGHTS = ROOT / ".voicefocus" / "models" / "yamnet.tflite"
WEIGHTS_SHA256 = "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
WINDOW = 15_600
HOP = 7_680
SINGING_LABELS = (
    "Singing", "Choir", "Yodeling", "Chant", "Mantra", "Child singing", "Synthetic singing",
    "A capella", "Humming", "Vocal music",
)
SPEECH_LABELS = ("Speech", "Conversation", "Narration, monologue", "Child speech, kid speaking", "Speech synthesizer")


def available() -> bool:
    return WEIGHTS.is_file() and importlib.util.find_spec("ai_edge_litert") is not None


def verified_weights(path: Path) -> bytes:
    weights = path.read_bytes()
    if sha256(weights).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("YAMNet checkpoint checksum mismatch.")
    return weights


class AudioTagger:
    def __init__(self) -> None:
        from ai_edge_litert.interpreter import Interpreter

        weights = verified_weights(WEIGHTS)
        with ZipFile(BytesIO(weights)) as archive:
            labels = archive.read("yamnet_label_list.txt").decode("utf-8").splitlines()
        self.singing = [labels.index(label) for label in SINGING_LABELS]
        self.speech = [labels.index(label) for label in SPEECH_LABELS]
        self.model = Interpreter(model_content=weights, num_threads=1)
        self.model.allocate_tensors()
        self.input = self.model.get_input_details()[0]["index"]
        self.output = self.model.get_output_details()[0]["index"]

    def singing_clip(self, signal: Signal) -> bool:
        if len(signal) < WINDOW:
            signal = np.pad(signal, (0, WINDOW - len(signal)))
        last = len(signal) - WINDOW
        starts = [*range(0, last, HOP), last]
        singing_frames = speech_frames = 0
        for start in starts:
            self.model.set_tensor(self.input, signal[start:start + WINDOW])
            self.model.invoke()
            scores = self.model.get_tensor(self.output)[0]
            singing_frames += int(np.max(scores[self.singing]) >= 0.13)
            speech_frames += int(np.max(scores[self.speech]) >= 0.35)
        return singing_frames * 4 >= len(starts) and speech_frames * 4 < len(starts)


def is_harmony(mixture: Signal, enrollment: Signal) -> bool:
    if len(mixture) < SAMPLE_RATE:
        return False
    tagger = AudioTagger()
    # Check the target's solo recording too. Loud background vocals in a
    # conversation must not route a spoken target to the two-singer model.
    return tagger.singing_clip(enrollment) and tagger.singing_clip(mixture)
