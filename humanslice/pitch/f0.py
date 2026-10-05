"""F0 curve utilities: unit conversion, octave-error repair, smoothing, interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_PERIOD = 0.01  # seconds; RMVPE hop at 16 kHz


@dataclass(slots=True)
class F0Curve:
    """Frame-level pitch: ``hz[i]`` at time ``i * period``; 0 means unvoiced."""

    hz: np.ndarray
    period: float = FRAME_PERIOD

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.hz.size) * self.period

    @property
    def voiced(self) -> np.ndarray:
        return self.hz > 0

    def midi(self) -> np.ndarray:
        """MIDI note numbers (float); NaN where unvoiced."""
        return hz_to_midi(self.hz)

    def slice(self, start: float, end: float) -> np.ndarray:
        a = max(0, int(round(start / self.period)))
        b = min(self.hz.size, int(round(end / self.period)))
        return self.hz[a:max(a, b)]

    def at(self, times: np.ndarray) -> np.ndarray:
        """Sample Hz at arbitrary times (nearest frame; 0 outside)."""
        idx = np.round(np.asarray(times) / self.period).astype(int)
        valid = (idx >= 0) & (idx < self.hz.size)
        out = np.zeros(idx.shape, dtype=np.float32)
        out[valid] = self.hz[idx[valid]]
        return out


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(hz / 440.0)
    return np.where(hz > 0, midi, np.nan)


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float64)
    hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    return np.where(np.isfinite(midi), hz, 0.0)


def fix_octave_errors(
    hz: np.ndarray,
    window: int = 41,
    tolerance: float = 1.5,
    max_run: int = 12,
) -> np.ndarray:
    """Fold short runs of frames that sit about an octave off the local median back into range.

    Only runs shorter than ``max_run`` frames are touched, so a genuine sung octave leap
    (which lasts longer than ~100 ms) is preserved.
    """
    midi = hz_to_midi(hz)
    voiced = np.isfinite(midi)
    if voiced.sum() < 5:
        return hz
    idx = np.flatnonzero(voiced)
    values = midi[idx]
    half = window // 2
    medians = np.array([np.median(values[max(0, p - half):p + half + 1]) for p in range(values.size)])
    delta = values - medians
    octave_off = np.abs(np.abs(delta) - 12.0) < tolerance
    out = hz.astype(np.float64).copy()
    for start, end in voiced_runs(octave_off):
        if end - start < max_run:
            for position in range(start, end):
                out[idx[position]] *= 0.5 if delta[position] > 0 else 2.0
    return out.astype(np.float32)


def remove_short_voiced(hz: np.ndarray, min_frames: int = 3) -> np.ndarray:
    """Zero out voiced islands shorter than ``min_frames`` (spurious detections)."""
    out = hz.copy()
    for start, end in voiced_runs(hz > 0):
        if end - start < min_frames:
            out[start:end] = 0.0
    return out


def voiced_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True values."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def interpolate_unvoiced(hz: np.ndarray) -> np.ndarray:
    """Fill unvoiced gaps by linear interpolation in the log domain (edges held)."""
    voiced = hz > 0
    if not voiced.any():
        return hz.copy()
    idx = np.arange(hz.size)
    log_hz = np.interp(idx, idx[voiced], np.log(hz[voiced]))
    return np.exp(log_hz).astype(np.float32)


def median_smooth(values: np.ndarray, size: int = 5) -> np.ndarray:
    if size <= 1 or values.size < size:
        return values.copy()
    from scipy.signal import medfilt

    return medfilt(values.astype(np.float64), kernel_size=size | 1).astype(values.dtype)


def clean_f0(hz: np.ndarray) -> np.ndarray:
    """Standard cleanup applied after RMVPE: octave repair + removal of tiny voiced islands."""
    return remove_short_voiced(fix_octave_errors(hz), min_frames=3)
