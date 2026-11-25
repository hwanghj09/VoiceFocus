"""Stateful PCM inference for the pinned REAL-TSE speaker-embedding checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.nn import functional as F

from backend.app.audio.wav import SAMPLE_RATE, Signal

if TYPE_CHECKING:
    from backend.app.models.worker import Extractor


class CausalStream:
    """Keep one enrollment, STFT overlap, and temporal LSTM state per stream."""

    @torch.inference_mode()
    def __init__(self, extractor: Extractor, enrollment: Signal) -> None:
        model = extractor.extractor.model
        self.separator = model.sep_model
        self.layers = self.separator.separator.separation
        active_features = {
            name for name, config in model.spk_configs["features"].items() if config["enabled"]
        }
        if (
            active_features != {"spkemb"}
            or any(layer.band_rnn.rnn.bidirectional for layer in self.layers)
            or self.separator.nspk != 1
        ):
            raise ValueError("Use the causal REAL-TSE speaker-embedding checkpoint.")

        self.device = extractor.extractor.device
        self.size = self.separator.stft.n_fft
        self.hop = self.separator.stft.hop_length
        self.radius = self.size // 2
        self.window = torch.hann_window(self.size, device=self.device)
        self.window_power = self.window.square()
        self.speaker = model.spk_ft.spkemb
        cue = torch.as_tensor(enrollment[None, :], dtype=torch.float32, device=self.device)
        self.embedding = self.speaker.compute(cue).unsqueeze(1).unsqueeze(3)

        self.states: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * len(self.layers)
        self.input = self.window.new_empty(0)
        self.input_tail = self.window.new_empty(0)
        self.output_sum = self.window.new_zeros(self.size - self.hop)
        self.output_weight = torch.zeros_like(self.output_sum)
        self.received = 0
        self.emitted = 0
        self.leading_padding = self.radius
        self.started = False
        self.closed = False

    @torch.inference_mode()
    def push(self, signal: Signal) -> Signal:
        if self.closed:
            raise RuntimeError("The stream has ended.")
        if signal.ndim != 1 or not np.isfinite(signal).all():
            raise ValueError(f"Expected finite mono PCM at {SAMPLE_RATE} Hz.")

        pcm = torch.as_tensor(signal, dtype=torch.float32, device=self.device)
        self.received += len(pcm)
        self.input = torch.cat((self.input, pcm))
        keep = self.radius + self.hop
        self.input_tail = torch.cat((self.input_tail, pcm))[-keep:].clone()
        if not self.started and self.received > self.radius:
            left_pad = self.input[1:self.radius + 1].flip(0)
            self.input = torch.cat((left_pad, self.input))
            self.started = True

        output = self._process()
        self.emitted += len(output)
        return output.cpu().numpy()

    @torch.inference_mode()
    def finish(self) -> Signal:
        if self.closed:
            raise RuntimeError("The stream has ended.")
        if not self.started:
            raise ValueError("The input is shorter than the STFT reflection window.")

        self.closed = True
        # Match the offline worker, which first pads to a whole hop and then
        # runs a centered STFT with reflection padding at both ends.
        extra = -self.received % self.hop
        tail = F.pad(self.input_tail[None, None], (0, extra), mode="reflect")
        tail = F.pad(tail, (0, self.radius), mode="reflect").reshape(-1)
        self.input = torch.cat((self.input, tail[-(extra + self.radius):]))
        output = self._process()
        overlap = self.output_sum / self.output_weight.clamp_min(1e-8)
        remaining = self.received - self.emitted
        return torch.cat((output, overlap))[:remaining].cpu().numpy()

    def _process(self) -> torch.Tensor:
        if not self.started or len(self.input) < self.size:
            return self.window.new_empty(0)

        frames = self.input.unfold(0, self.size, self.hop)
        ready = len(frames) * self.hop
        length = ready + self.size - self.hop
        spec = torch.fft.rfft(frames * self.window).transpose(0, 1).unsqueeze(0)
        estimated = self._separate(spec)
        restored = torch.fft.irfft(estimated, n=self.size) * self.window

        geometry = dict(output_size=(1, length), kernel_size=(1, self.size), stride=(1, self.hop))
        total = F.fold(restored.T.unsqueeze(0), **geometry).reshape(-1)
        power = self.window_power[:, None].expand(-1, len(frames)).unsqueeze(0)
        weight = F.fold(power, **geometry).reshape(-1)
        overlap = self.size - self.hop
        total[:overlap] += self.output_sum
        weight[:overlap] += self.output_weight

        self.input = self.input[ready:].clone()
        self.output_sum = total[ready:].clone()
        self.output_weight = weight[ready:].clone()
        trim = min(ready, self.leading_padding)
        self.leading_padding -= trim
        return total[trim:ready] / weight[trim:ready].clamp_min(1e-8)

    def _separate(self, spec: torch.Tensor) -> torch.Tensor:
        separator = self.separator
        bands = separator.band_split(torch.stack((spec.real, spec.imag), dim=1))
        features = separator.subband_norm(bands)
        features = self.speaker.post(features, self.embedding)
        _, band_count, feature_count, frame_count = features.shape

        for index, layer in enumerate(self.layers):
            # Temporal LSTMs carry their hidden and cell states forward.
            temporal = features.reshape(band_count, feature_count, frame_count)
            band_rnn = layer.band_rnn
            sequence = band_rnn.norm(temporal).transpose(1, 2).contiguous()
            output, self.states[index] = band_rnn.rnn(sequence, self.states[index])
            temporal = temporal + band_rnn.proj(output).transpose(1, 2)

            # The bidirectional frequency LSTM completes within each frame.
            frequency = temporal.permute(2, 1, 0).contiguous()
            features = layer.band_comm(frequency).permute(2, 1, 0).unsqueeze(0).contiguous()

        complex_masked = separator.band_masker(features, separator.band_split(spec))
        return torch.complex(complex_masked[0, 0, 0], complex_masked[0, 1, 0]).transpose(0, 1)
