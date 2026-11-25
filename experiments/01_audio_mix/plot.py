"""Save waveform and STFT comparisons for WAVs in a common amplitude scale."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from scipy.signal import stft

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from backend.app.audio.wav import SAMPLE_RATE, read_audio


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("wavs", nargs="+", type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(len(args.wavs), 2, squeeze=False, figsize=(12, 2.7 * len(args.wavs)))
    for (wave, spectrum), path in zip(axes, args.wavs, strict=True):
        audio = read_audio(path)
        time = np.arange(len(audio)) / SAMPLE_RATE
        frequencies, frames, transform = stft(audio, SAMPLE_RATE, nperseg=512, noverlap=384)
        wave.plot(time, audio, color="#126b68", linewidth=0.7)
        wave.set(ylim=(-1, 1), xlim=(0, time[-1]), xlabel="Time (s)", ylabel="Amplitude", title=path.stem)
        image = spectrum.pcolormesh(
            frames, frequencies / 1000, 20 * np.log10(np.abs(transform) + 1e-8) + 6.02,
            vmin=-85, vmax=-10, cmap="magma", shading="auto",
        )
        spectrum.set(xlabel="Time (s)", ylabel="Frequency (kHz)", title=f"{path.stem} · STFT")
    fig.colorbar(image, ax=axes[:, 1], label="dBFS")
    fig.subplots_adjust(hspace=0.65, bottom=0.12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=144, bbox_inches="tight")


if __name__ == "__main__":
    main()
