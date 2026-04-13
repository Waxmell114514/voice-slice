from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from models.project import AudioTrack


class AudioLoadError(RuntimeError):
    """Raised when an input file cannot be decoded as audio."""


def load_audio_file(path: str | Path, target_sr: int | None = None) -> AudioTrack:
    """Load an audio file as mono float32 data.

    The loader tries `soundfile` first and falls back to `librosa.load`
    to handle formats such as mp3 on systems where backend support varies.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise AudioLoadError(f"Audio file does not exist: {file_path}")

    try:
        samples, sample_rate = sf.read(file_path, always_2d=False, dtype="float32")
        if isinstance(samples, np.ndarray) and samples.ndim == 2:
            samples = samples.mean(axis=1)
    except Exception:
        try:
            import librosa

            samples, sample_rate = librosa.load(file_path.as_posix(), sr=target_sr, mono=True)
            samples = samples.astype(np.float32, copy=False)
        except Exception as exc:
            raise AudioLoadError(f"Failed to decode audio file: {file_path}") from exc
    else:
        samples = np.asarray(samples, dtype=np.float32)
        if target_sr is not None and int(sample_rate) != int(target_sr):
            import librosa

            samples = librosa.resample(samples, orig_sr=int(sample_rate), target_sr=int(target_sr))
            sample_rate = int(target_sr)

    if samples.size == 0 or int(sample_rate) <= 0:
        raise AudioLoadError(f"Decoded empty audio file: {file_path}")

    samples = np.ascontiguousarray(samples, dtype=np.float32)
    duration = float(samples.shape[0] / int(sample_rate))
    return AudioTrack(path=str(file_path.resolve()), samples=samples, sample_rate=int(sample_rate), duration=duration)


def extract_clip(samples: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Return a clipped sample window from absolute time bounds."""

    if sample_rate <= 0:
        return np.zeros(0, dtype=np.float32)

    start_index = max(0, int(round(start * sample_rate)))
    end_index = min(samples.shape[0], int(round(end * sample_rate)))
    if end_index <= start_index:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(samples[start_index:end_index], dtype=np.float32)
