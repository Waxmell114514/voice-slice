"""PC-NSF-HiFiGAN backend: parameter-domain concatenation in log-mel space, neural vocoding.

Same placement / time-warp / crossfade logic as the WORLD backend, but units are
represented by DiffSinger-style log-mel frames (44.1 kHz, hop 512, 128 bins) and the
waveform comes from OpenVPI's pitch-controllable NSF-HiFiGAN, which was trained with
pitch augmentation and so tolerates mel / F0 mismatch from pitch-shifting.

Weights: openvpi/vocoders ``pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`` (CC BY-NC-SA 4.0,
downloaded on first use).
"""

from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

from humanslice.common.audio import load_wav
from humanslice.common.paths import data_root
from humanslice.corpus.units import Bank, Unit
from humanslice.pitch.f0 import interpolate_unvoiced
from humanslice.render.plan import Placement, phrases
from humanslice.render.world import WorldOptions

SR = 44100
HOP = 512
N_FFT = 2048
N_MELS = 128
FMIN, FMAX = 40.0, 16000.0
FRAME = HOP / SR
LOG_FLOOR = float(np.log(1e-5))
RELEASE = "https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
PACKAGE = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"


def vocoder_path() -> str:
    folder = data_root() / "vocoders"
    onnx = folder / "pc_nsf_onnx" / f"{PACKAGE}.onnx"
    if not onnx.exists():
        folder.mkdir(parents=True, exist_ok=True)
        archive = folder / f"{PACKAGE}.oudep"
        if not archive.exists():
            with urllib.request.urlopen(RELEASE + f"{PACKAGE}.oudep", timeout=1200) as response:
                archive.write_bytes(response.read())
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(folder / "pc_nsf_onnx")
    return str(onnx)


class MelExtractor:
    """DiffSinger mel: reflect-pad (n_fft - hop) / 2, |STFT|, slaney mel, natural log."""

    def __init__(self) -> None:
        from librosa.filters import mel

        self.basis = mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float32)
        self.window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        pad = (N_FFT - HOP) // 2
        x = np.pad(audio.astype(np.float32), (pad, pad), mode="reflect")
        n_frames = 1 + (x.size - N_FFT) // HOP
        if n_frames <= 0:
            return np.full((0, N_MELS), LOG_FLOOR, dtype=np.float32)
        frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP][:n_frames] * self.window
        magnitude = np.sqrt(np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-9).astype(np.float32)
        return np.log(np.maximum(magnitude @ self.basis.T, 1e-5))


@dataclass(slots=True)
class UnitMel:
    t0: float  # source time of frame 0's centre
    mel: np.ndarray  # (frames, 128) log magnitude
    voiced: np.ndarray  # (frames,) bool

    def index(self, src_times: np.ndarray) -> np.ndarray:
        return np.clip(np.round((src_times - self.t0) / FRAME).astype(int), 0, self.mel.shape[0] - 1)


class MelAnalyzer:
    def __init__(self, bank: Bank, margin: float = 0.12) -> None:
        self.bank = bank
        self.margin = margin
        self.extract = MelExtractor()
        self._sources: dict[str, np.ndarray] = {}
        self._cache: dict[str, UnitMel] = {}

    def _source(self, source_id: str) -> np.ndarray:
        if source_id not in self._sources:
            if len(self._sources) > 16:
                self._sources.clear()
            self._sources[source_id], _ = load_wav(self.bank.master_path(source_id), SR)
        return self._sources[source_id]

    def features(self, unit: Unit) -> UnitMel:
        cached = self._cache.get(unit.unit_id)
        if cached is not None:
            return cached
        import pyworld as pw

        audio = self._source(unit.source_id)
        a = max(0, int((unit.start - self.margin) * SR))
        b = min(audio.size, int((unit.end + self.margin) * SR))
        clip = audio[a:b]
        mel = self.extract(clip)
        f0, t = pw.dio(clip.astype(np.float64), SR, f0_floor=55.0, f0_ceil=1100.0, frame_period=FRAME * 1000)
        voiced = np.zeros(mel.shape[0], dtype=bool)
        voiced[: min(mel.shape[0], f0.size)] = f0[: mel.shape[0]] > 0
        features = UnitMel(t0=a / SR + 0.5 * FRAME, mel=mel, voiced=voiced)
        self._cache[unit.unit_id] = features
        return features


class Vocoder:
    def __init__(self) -> None:
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in ort.get_available_providers()]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(vocoder_path(), options, providers=providers + ["CPUExecutionProvider"])

    def __call__(self, mel: np.ndarray, f0: np.ndarray) -> np.ndarray:
        out = self.session.run(
            ["waveform"],
            {"mel": mel[None].astype(np.float32), "f0": f0[None].astype(np.float32)},
        )[0]
        return out[0]


def render_nsf(
    bank: Bank,
    placements: list[Placement],
    target_f0: np.ndarray,
    target_period: float,
    duration: float,
    transpose: int,
    loudness_db: np.ndarray | None = None,
    note_curve: np.ndarray | None = None,
    options: WorldOptions | None = None,
    progress=None,
) -> tuple[np.ndarray, int]:
    options = options or WorldOptions()
    analyzer = MelAnalyzer(bank)
    vocoder = Vocoder()
    out = np.zeros(int(np.ceil((duration + 1.0) * SR)), dtype=np.float64)

    filled = interpolate_unvoiced(target_f0) if np.any(target_f0 > 0) else np.full_like(target_f0, 220.0)
    target_midi = 69 + 12 * np.log2(np.maximum(filled, 1e-3) / 440.0)
    if note_curve is not None and options.pitch_follow < 1.0:
        n = min(note_curve.size, target_midi.size)
        target_midi = note_curve[:n] + options.pitch_follow * (target_midi[:n] - note_curve[:n])
    target_midi = target_midi + transpose

    loud = None
    if loudness_db is not None and loudness_db.size and np.any(target_f0 > 0):
        voiced_level = loudness_db[target_f0[: loudness_db.size] > 0]
        loud = loudness_db - float(np.median(voiced_level))
    energies = [p.unit.energy_db for p in placements if p.unit.energy_db is not None]
    reference = float(np.median(energies)) if energies else -20.0

    groups = phrases(placements)
    for number, group in enumerate(groups, start=1):
        t_start = min(p.out_start for p in group) - 0.03
        t_end = max(p.out_end for p in group) + 0.03
        n_frames = int(np.ceil((t_end - t_start) / FRAME)) + 1
        times = t_start + (np.arange(n_frames) + 0.5) * FRAME
        acc_mel = np.zeros((n_frames, N_MELS))
        acc_w = np.zeros(n_frames)
        acc_voiced = np.zeros(n_frames)
        for placement in group:
            mask = (times >= placement.out_start) & (times <= placement.out_end)
            if not mask.any():
                continue
            ts = times[mask]
            w = placement.weight(ts)
            feats = analyzer.features(placement.unit)
            idx = feats.index(placement.source_time(ts))
            mel = feats.mel[idx].astype(np.float64)
            if placement.pingpong is not None:
                out_a, out_b, src_a, src_b, _ = placement.pingpong
                loop = (ts >= out_a) & (ts < out_b)
                nucleus = feats.index(np.arange(src_a, src_b, FRAME))
                mel[loop] = (1 - options.loop_smoothing) * mel[loop] + options.loop_smoothing * feats.mel[nucleus].mean(axis=0)
            gain_db = np.clip(reference - (placement.unit.energy_db or reference), -options.max_gain_db, options.max_gain_db)
            gains = np.full(ts.size, gain_db)
            if loud is not None:
                li = np.clip(np.round(ts / target_period).astype(int), 0, loud.size - 1)
                gains = gains + np.clip(options.level_follow * loud[li], -15.0, 6.0)
            mel = mel + (gains * np.log(10.0) / 20.0)[:, None]  # log magnitude
            voiced = feats.voiced[idx] | (ts >= placement.onset)
            acc_mel[mask] += w[:, None] * mel
            acc_w[mask] += w
            acc_voiced[mask] += w * voiced

        present = acc_w > 1e-4
        norm = np.where(present, acc_w, 1.0)
        mel = acc_mel / norm[:, None] + np.log(np.clip(acc_w, 1e-5, 1.0))[:, None]
        mel[~present] = LOG_FLOOR
        mel = np.maximum(mel, LOG_FLOOR)
        fi = np.clip(np.round(times / target_period).astype(int), 0, target_midi.size - 1)
        f0 = 440.0 * 2.0 ** ((target_midi[fi] - 69.0) / 12.0)
        f0[(acc_voiced / norm < 0.5) | ~present] = 0.0

        y = vocoder(mel.astype(np.float32), f0.astype(np.float32))
        a = int(round(t_start * SR))
        if a < 0:
            y, a = y[-a:], 0
        b = min(out.size, a + y.size)
        out[a:b] += y[: b - a]
        if progress is not None and (number % 10 == 0 or number == len(groups)):
            progress(f"[render] phrase {number}/{len(groups)}")

    out = out[: int(np.ceil(duration * SR))]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 10 ** (-1.0 / 20) / peak
    return out.astype(np.float32), SR
