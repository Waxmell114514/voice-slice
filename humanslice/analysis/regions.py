from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanslice.audio.io import extract_clip
from humanslice.models.project import RegionRange, SegmentRegions


@dataclass(slots=True)
class RegionConfig:
    """Parameters for onset / nucleus / tail heuristic partitioning."""

    onset_max_sec: float = 0.12
    tail_max_sec: float = 0.16
    onset_ratio: float = 0.22
    tail_ratio: float = 0.20
    min_nucleus_sec: float = 0.04
    frame_length: int = 1024
    hop_length: int = 256


def analyze_segment_regions(
    samples: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    config: RegionConfig | None = None,
) -> SegmentRegions:
    """Split a segment into onset, nucleus, and tail with simple heuristics."""
    cfg = config or RegionConfig()
    duration = max(0.0, end - start)
    if duration <= 0:
        return _fallback_regions(start, end)

    clip = extract_clip(samples, sample_rate, start, end)
    if clip.size == 0 or duration <= cfg.min_nucleus_sec * 1.5:
        return _fallback_regions(start, end)

    import librosa

    frame_length = min(cfg.frame_length, max(256, clip.size))
    energy = librosa.feature.rms(y=clip, frame_length=frame_length, hop_length=cfg.hop_length)[0]
    if energy.size < 3:
        return _fallback_regions(start, end)

    smooth = _smooth(energy)
    max_energy = max(float(np.max(smooth)), 1e-6)
    times = librosa.frames_to_time(np.arange(smooth.shape[0]), sr=sample_rate, hop_length=cfg.hop_length)

    onset_threshold = max_energy * 0.72
    onset_candidates = np.flatnonzero(smooth >= onset_threshold)
    onset_end_rel = float(times[onset_candidates[0]]) if onset_candidates.size else duration * cfg.onset_ratio
    onset_end_rel = max(onset_end_rel + 0.02, duration * 0.12)
    onset_end_rel = min(onset_end_rel, cfg.onset_max_sec, duration * 0.38)

    sustain_threshold = max_energy * 0.45
    sustain_candidates = np.flatnonzero(smooth >= sustain_threshold)
    if sustain_candidates.size:
        last_sustain_rel = float(times[sustain_candidates[-1]])
        base_tail_rel = duration - min(cfg.tail_max_sec, max(duration * cfg.tail_ratio, 0.03))
        tail_start_rel = max(base_tail_rel, min(last_sustain_rel, duration - 0.01))
    else:
        tail_start_rel = duration - min(cfg.tail_max_sec, duration * cfg.tail_ratio)

    onset_end_rel = max(0.02, min(onset_end_rel, duration - cfg.min_nucleus_sec - 0.01))
    tail_start_rel = max(onset_end_rel + cfg.min_nucleus_sec, min(tail_start_rel, duration - 0.01))

    if tail_start_rel - onset_end_rel < cfg.min_nucleus_sec:
        return _fallback_regions(start, end)

    onset = RegionRange(start=start, end=start + onset_end_rel)
    nucleus = RegionRange(start=onset.end, end=start + tail_start_rel)
    tail = RegionRange(start=nucleus.end, end=end)
    return SegmentRegions(onset=onset, nucleus=nucleus, tail=tail)


def _fallback_regions(start: float, end: float) -> SegmentRegions:
    duration = max(0.0, end - start)
    onset_end = start + duration * 0.22
    tail_start = start + duration * 0.78
    return SegmentRegions(
        onset=RegionRange(start=start, end=onset_end),
        nucleus=RegionRange(start=onset_end, end=tail_start),
        tail=RegionRange(start=tail_start, end=end),
    )


def _smooth(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(5, dtype=np.float32) / 5.0
    return np.convolve(values, kernel, mode="same").astype(np.float32, copy=False)
