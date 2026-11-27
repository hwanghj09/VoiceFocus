"""Reproduce the introductory 440/880 Hz FFT and chirp STFT experiments."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import chirp, stft


def main():
    output = Path("artifacts/dsp")
    output.mkdir(parents=True, exist_ok=True)
    fs = 16000
    t = np.arange(fs) / fs
    a = .4 * np.sin(2 * np.pi * 440 * t)
    b = .4 * np.sin(2 * np.pi * 880 * t)
    y = a + b
    f = np.fft.rfftfreq(len(t), 1 / fs)
    z = np.fft.rfft(y)
    results = {}
    recovered = []
    for freq, original in [(440, a), (880, b)]:
        index = np.argmin(np.abs(f - freq))
        isolated = np.zeros_like(z)
        isolated[index] = z[index]
        r = np.fft.irfft(isolated, n=len(y))
        recovered.append(r)
        results[str(freq)] = {
            "amplitude": float(2 * abs(z[index]) / len(y)),
            "rmse": float(np.sqrt(np.mean((r - original) ** 2))),
            "energy": float(np.sum(original ** 2)),
        }
    results["mixture_energy"] = float(np.sum(y ** 2))
    results["cross_term"] = float(2 * np.dot(a, b))
    view = t <= .02
    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    for ax, signal, label in zip(axes, [a, b, y], ["A: 440 Hz", "B: 880 Hz", "Mixture"]):
        ax.plot(t[view] * 1000, signal[view])
        ax.set_ylabel("Amplitude")
        ax.set_title(label)
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output / "01_superposition.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(f, 2 * abs(z) / len(y))
    ax.set(xlim=(0, 1400), xlabel="Frequency (Hz)", ylabel="Amplitude", title="FFT of mixture")
    fig.tight_layout()
    fig.savefig(output / "02_fft.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ax, original, reconstructed, freq in zip(axes, [a, b], recovered, [440, 880]):
        ax.plot(t[view] * 1000, original[view], label="Original")
        ax.plot(t[view] * 1000, reconstructed[view], "--", label="Reconstructed")
        ax.set(title=f"{freq} Hz", ylabel="Amplitude")
        ax.legend()
    axes[-1].set_xlabel("Time (ms)")
    fig.tight_layout()
    fig.savefig(output / "03_reconstruction.png", dpi=180)
    plt.close(fig)
    t2 = np.arange(2 * fs) / fs
    x = chirp(t2, 200, 2, 1800)
    freq, times, spec = stft(x, fs, nperseg=512, noverlap=384, window="hann")
    valid = (times >= .016) & (times <= 1.984)
    error = np.abs(freq[np.argmax(abs(spec), axis=0)][valid] - (200 + 800 * times[valid]))
    results["chirp"] = {
        "median_absolute_error_hz": float(np.median(error)),
        "max_absolute_error_hz": float(np.max(error)),
        "bin_hz": float(freq[1] - freq[0]),
        "hop_seconds": float(times[1] - times[0]),
    }
    fig, ax = plt.subplots(figsize=(9, 4))
    mesh = ax.pcolormesh(times, freq, 20 * np.log10(np.maximum(abs(spec), 1e-8)), shading="auto", vmin=-80, vmax=0)
    ax.set(ylim=(0, 2200), xlabel="Time (s)", ylabel="Frequency (Hz)", title="STFT of 200–1800 Hz chirp")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    fig.tight_layout()
    fig.savefig(output / "04_stft.png", dpi=180)
    plt.close(fig)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
