"""Audio decoding / resampling helpers shared by the pipelines."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".flv", ".webm", ".mov", ".avi", ".ts"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def decode(path: str | Path, sample_rate: int) -> np.ndarray:
    """Decode any audio / video file to mono float32 at ``sample_rate``.

    Uses ffmpeg when available (handles video containers and odd codecs),
    otherwise soundfile + librosa resampling.
    """
    path = Path(path)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=False)
        if result.returncode == 0 and result.stdout:
            return np.frombuffer(result.stdout, dtype=np.float32).copy()
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            raise RuntimeError(f"ffmpeg failed on {path}: {result.stderr.decode(errors='replace')[-500:]}")

    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(samples.mean(axis=1), sr, sample_rate)


def resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return np.ascontiguousarray(samples, dtype=np.float32)
    import soxr

    return soxr.resample(samples.astype(np.float32), orig_sr, target_sr, quality="HQ").astype(np.float32)


def load_wav(path: str | Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    samples, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate is not None and sr != sample_rate:
        return resample(mono, sr, sample_rate), sample_rate
    return mono, sr


def save_wav(path: str | Path, samples: np.ndarray, sample_rate: int, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(samples, -1.0, 1.0), sample_rate, subtype=subtype)


def frame_rms_db(samples: np.ndarray, sample_rate: int, period: float = 0.01, window: float = 0.025) -> np.ndarray:
    """RMS energy in dBFS on a fixed frame grid (frame i centred at i * period)."""
    hop = int(round(period * sample_rate))
    win = int(round(window * sample_rate))
    n_frames = int(np.ceil(samples.size / hop)) if samples.size else 0
    padded = np.pad(samples.astype(np.float64), (win // 2, win))
    squares = np.concatenate([[0.0], np.cumsum(padded**2)])
    starts = np.arange(n_frames) * hop
    energy = (squares[starts + win] - squares[starts]) / win
    return (10.0 * np.log10(np.maximum(energy, 1e-12))).astype(np.float32)


def split_on_silence(
    samples: np.ndarray,
    sample_rate: int,
    max_chunk: float = 20.0,
    min_silence: float = 0.25,
    min_speech: float = 0.3,
    pad: float = 0.08,
    threshold_db: float | None = None,
) -> list[tuple[float, float]]:
    """Energy-based chunking: speech regions separated by pauses, each at most ``max_chunk`` seconds.

    Long regions are split at their quietest internal frame. Returns (start, end) in seconds.
    """
    period = 0.01
    db = frame_rms_db(samples, sample_rate, period)
    if db.size == 0:
        return []
    if threshold_db is None:
        noise_floor = float(np.percentile(db, 10))
        speech_level = float(np.percentile(db, 90))
        threshold_db = max(noise_floor + 6.0, speech_level - 35.0)
    active = db > threshold_db

    # Close short gaps, then collect runs.
    gap_frames = int(min_silence / period)
    runs: list[list[int]] = []
    for start, end in _runs(active):
        if runs and start - runs[-1][1] < gap_frames:
            runs[-1][1] = end
        else:
            runs.append([start, end])

    chunks: list[tuple[int, int]] = []
    max_frames = int(max_chunk / period)
    for start, end in runs:
        if end - start < int(min_speech / period):
            continue
        stack = [(start, end)]
        while stack:
            a, b = stack.pop()
            if b - a <= max_frames:
                chunks.append((a, b))
                continue
            inner = db[a + int(1.0 / period):b - int(1.0 / period)]
            cut = a + int(1.0 / period) + int(np.argmin(inner)) if inner.size else (a + b) // 2
            stack.extend([(cut, b), (a, cut)])
    chunks.sort()
    duration = samples.size / sample_rate
    return [
        (max(0.0, a * period - pad), min(duration, b * period + pad))
        for a, b in chunks
    ]


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))
