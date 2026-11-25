"""Download short LibriSpeech test-clean excerpts or synthesize matching text."""
import argparse
import json
import subprocess
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

from backend.app.audio.wav import read_audio, write_analysis_wav

DATASET = "openslr/librispeech_asr"
SPEAKERS = {
    "a": (0, "6930-75918", 0),
    "b": (200, "260-123286", 0),
}
CALL_SPEAKERS = {
    **SPEAKERS,
    "1320": (100, "1320-122617", 22),
    "3575": (400, "3575-170457", 31),
    "8455": (1000, "8455-210777", 28),
}
TEXTS = (
    "The window was open when the first bus reached the station.",
    "On Saturday morning we packed a small bag and walked to the market.",
    "The rain cleared before noon, and we could see the hills from the road.",
    "My sister left the blue notebook near the lamp beside the kitchen door.",
    "A bright moon rose above the trees while the last visitors left the park.",
    "Next week I will bring the new map so we can find a quieter path together.",
)


def fetch_librispeech(directory: Path, speakers=SPEAKERS) -> None:
    for speaker, (offset, chapter, first) in speakers.items():
        query = urlencode({
            "dataset": DATASET, "config": "clean", "split": "test",
            "offset": offset, "length": 6,
        })
        with urlopen(f"https://datasets-server.huggingface.co/rows?{query}", timeout=60) as response:
            rows = json.load(response)["rows"]
        for index, entry in enumerate(rows):
            row = entry["row"]
            expected = f"{chapter}-{first + index:04d}"
            if row["id"] != expected:
                raise ValueError(f"Expected LibriSpeech utterance {expected}.")
            with urlopen(row["audio"][0]["src"], timeout=60) as response:
                signal = read_audio(BytesIO(response.read()), min_seconds=1, wav_only=False)
            write_analysis_wav(directory / f"{speaker}-{index}.wav", signal)
        if len(rows) != 6:
            raise ValueError("Expected six LibriSpeech utterances per speaker.")


def synthesize(directory: Path) -> None:
    with TemporaryDirectory() as temporary:
        wav = Path(temporary) / "sample.wav"
        for speaker, voice, speed in (("a", "en-us+m3", "155"), ("b", "en-us+f3", "170")):
            for index, sentence in enumerate(TEXTS):
                subprocess.run(
                    ["espeak-ng", "-v", voice, "-s", speed, "-w", str(wav), sentence],
                    check=True,
                )
                signal = read_audio(wav, min_seconds=1)
                write_analysis_wav(directory / f"{speaker}-{index}.wav", signal)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("librispeech", "librispeech-calls", "synthetic"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.source == "synthetic":
        synthesize(args.directory)
    else:
        fetch_librispeech(args.directory, CALL_SPEAKERS if args.source == "librispeech-calls" else SPEAKERS)


if __name__ == "__main__":
    main()
