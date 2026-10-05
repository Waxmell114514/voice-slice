"""Heuristic note segmentation of a sung syllable from its F0 curve.

Used when no neural note transcriber (GAME) is available. Within each syllable the
pitch is smoothed over ~one vibrato period, then split wherever it settles on a new
level for long enough; each level becomes a (slur) note.
"""

from __future__ import annotations

import numpy as np

from humanslice.pitch.f0 import FRAME_PERIOD, hz_to_midi, interpolate_unvoiced


def tuning_offset(f0_hz: np.ndarray, min_concentration: float = 0.35) -> float:
    """Global detuning of the singer relative to A440, in semitones (-0.5..0.5).

    Only sustained frames (pitch nearly flat over 50 ms) vote, so glides and vibrato
    don't smear the estimate; if the votes are too dispersed, assume no detuning.
    """
    midi = hz_to_midi(f0_hz)
    slope = np.abs(np.diff(midi, prepend=np.nan))
    steady = np.isfinite(midi) & (_moving_average(np.nan_to_num(slope, nan=1.0), 5) < 0.03)
    if steady.sum() < 50:
        return 0.0
    frac = midi[steady] - np.round(midi[steady])
    vector = np.mean(np.exp(2j * np.pi * frac))  # circular mean: +-0.5 don't cancel out
    if abs(vector) < min_concentration:
        return 0.0
    return float(np.angle(vector) / (2 * np.pi))


def segment_syllable(
    f0_hz: np.ndarray,
    start: float,
    end: float,
    tuning: float = 0.0,
    period: float = FRAME_PERIOD,
    min_note: float = 0.12,
    threshold: float = 0.8,
) -> list[tuple[float, float, int]]:
    """Split [start, end] into notes; returns (start, end, midi) tuples.

    ``f0_hz`` is the whole-song curve; unvoiced frames are bridged by interpolation.
    """
    a = max(0, int(round(start / period)))
    b = min(f0_hz.size, int(round(end / period)))
    if b - a < 2:
        return []
    segment = f0_hz[a:b]
    voiced = segment > 0
    if voiced.sum() < 3:
        return []
    midi = hz_to_midi(interpolate_unvoiced(segment)) - tuning
    smooth = _moving_average(midi, 15)  # ~150 ms: removes vibrato, keeps note changes
    light = _moving_average(midi, 5)  # for locating the transition precisely

    min_frames = max(2, int(round(min_note / period)))
    boundaries = [0]
    level = float(np.median(smooth[:min_frames]))
    run_start: int | None = None
    for index in range(1, smooth.size):
        if abs(smooth[index] - level) > threshold:
            if run_start is None:
                run_start = index
            elif index - run_start >= min_frames // 2:
                cut = _steepest(light, max(boundaries[-1] + 1, run_start - 10), index)
                if cut - boundaries[-1] >= min_frames and smooth.size - cut >= min_frames // 2:
                    boundaries.append(cut)
                    level = float(np.median(smooth[cut:min(smooth.size, cut + min_frames)]))
                run_start = None
        else:
            run_start = None
            # Track slow drift of the held level.
            level = 0.95 * level + 0.05 * float(smooth[index])
    boundaries.append(smooth.size)

    notes: list[tuple[float, float, int]] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        core = midi[left:right][voiced[left:right]] if voiced[left:right].any() else midi[left:right]
        # Ignore transition edges when estimating the held pitch.
        trim = (right - left) // 5
        if right - left > 10 and voiced[left + trim:right - trim].any():
            core = midi[left + trim:right - trim][voiced[left + trim:right - trim]]
        pitch = int(round(float(np.median(core))))
        note_start = start + left * period
        note_end = start + right * period if right < smooth.size else end
        if notes and notes[-1][2] == pitch:
            notes[-1] = (notes[-1][0], note_end, pitch)
        else:
            notes.append((note_start, note_end, pitch))
    return notes


def _moving_average(values: np.ndarray, size: int) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    size = min(size, values.size if values.size % 2 else values.size - 1)
    kernel = np.ones(size) / size
    padded = np.pad(values, (size // 2, size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _steepest(values: np.ndarray, left: int, right: int) -> int:
    """Index of the largest pitch change between left and right (the transition point)."""
    if right - left < 2:
        return right
    diffs = np.abs(np.diff(values[left:right]))
    return left + int(np.argmax(diffs)) + 1
