"""RMVPE pitch estimator (robust to accompaniment; standard in RVC / DiffSinger / OpenUtau).

Model code adapted from Applio ``rvc/lib/predictors/RMVPE.py`` (MIT License,
Copyright (c) 2023 liujing04, 源文雨, Ftps; 2026 AI Hispano), itself from
RVC-Project. Weights: ``lj1995/VoiceConversionWebUI/rmvpe.pt`` (MIT).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 128
N_CLASS = 360
SAMPLE_RATE = 16000
HOP_LENGTH = 160  # 10 ms frames
WEIGHTS = ("lj1995/VoiceConversionWebUI", "rmvpe.pt")


class _ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.is_shortcut = in_channels != out_channels
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + (self.shortcut(x) if self.is_shortcut else x)


class _ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([_ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(_ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for block in self.conv:
            x = block(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, in_channels, in_size, n_encoders, kernel_size, n_blocks, out_channels=16, momentum=0.01):
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        for _ in range(n_encoders):
            self.layers.append(_ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum=momentum))
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors: list[torch.Tensor] = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class _Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList([_ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)])
        for _ in range(n_inters - 1):
            self.layers.append(_ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride, (1, 1), out_padding, bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([_ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(_ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = torch.cat((self.conv1(x), concat_tensor), dim=1)
        for block in self.conv2:
            x = block(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(_ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum))
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for index, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - index])
        return x


class _DeepUnet(nn.Module):
    def __init__(self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.encoder = _Encoder(in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels)
        self.intermediate = _Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = _Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        return self.decoder(self.intermediate(x), concat_tensors)


class _BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_features, num_layers=num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        return self.gru(x)[0]


class _E2E(nn.Module):
    def __init__(self, n_blocks, n_gru, kernel_size, en_de_layers=5, inter_layers=4, in_channels=1, en_out_channels=16):
        super().__init__()
        self.unet = _DeepUnet(kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels)
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            _BiGRU(3 * 128, 256, n_gru), nn.Linear(512, N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class _MelSpectrogram(nn.Module):
    def __init__(self, n_mels, sample_rate, win_length, hop_length, mel_fmin=30, mel_fmax=8000, clamp=1e-5):
        super().__init__()
        from librosa.filters import mel

        basis = mel(sr=sample_rate, n_fft=win_length, n_mels=n_mels, fmin=mel_fmin, fmax=mel_fmax, htk=True)
        self.register_buffer("mel_basis", torch.from_numpy(basis).float())
        self.register_buffer("window", torch.hann_window(win_length))
        self.n_fft = win_length
        self.hop_length = hop_length
        self.clamp = clamp

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        ).abs()
        return torch.log(torch.clamp(self.mel_basis @ spec, min=self.clamp))


class RMVPE:
    """Frame-level F0 (Hz, 0 = unvoiced) at 16 kHz / 10 ms hop."""

    def __init__(self, device: str | torch.device | None = None, weights_path: str | None = None) -> None:
        from humanslice.common.models import hf_file

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        path = weights_path or hf_file(*WEIGHTS)
        model = _E2E(4, 1, (2, 2))
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.model = model.eval().to(self.device)
        self.mel = _MelSpectrogram(N_MELS, SAMPLE_RATE, 1024, HOP_LENGTH).to(self.device)
        cents = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self._cents = np.pad(cents, (4, 4))

    @torch.no_grad()
    def salience(self, audio_16k: np.ndarray, chunk_frames: int = 32000) -> np.ndarray:
        audio = torch.from_numpy(np.ascontiguousarray(audio_16k, dtype=np.float32)).to(self.device).unsqueeze(0)
        mel = self.mel(audio)
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
        chunks = [self.model(mel[..., start:start + chunk_frames]) for start in range(0, mel.shape[-1], chunk_frames)]
        return torch.cat(chunks, dim=1)[0, :n_frames].float().cpu().numpy()

    def infer(self, audio_16k: np.ndarray, threshold: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
        """Return (f0_hz, confidence) per 10 ms frame."""
        salience = self.salience(audio_16k)
        center = np.argmax(salience, axis=1)
        padded = np.pad(salience, ((0, 0), (4, 4)))
        idx = center[:, None] + 4 + np.arange(-4, 5)[None, :]
        local = padded[np.arange(padded.shape[0])[:, None], idx]
        cents = np.sum(local * self._cents[idx], axis=1) / np.maximum(np.sum(local, axis=1), 1e-9)
        confidence = np.max(salience, axis=1)
        f0 = 10.0 * 2.0 ** (cents / 1200.0)
        f0[confidence <= threshold] = 0.0
        return f0.astype(np.float32), confidence.astype(np.float32)
