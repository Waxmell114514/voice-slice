from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import find_peaks

from models.project import Segment


@dataclass(slots=True)
class CutpointConfig:
    """Parameters for heuristic cutpoint detection and cleanup."""

    frame_length: int = 2048
    hop_length: int = 256
    min_gap_ms: int = 120
    silence_percentile: float = 18.0
    silence_min_ms: int = 150
    max_candidates: int = 96
    weak_boundary_threshold: float = 0.30
    short_segment_ms: int = 120
    breath_max_ms: int = 260
    breath_energy_ratio: float = 0.82
    breath_zcr_threshold: float = 0.105
    breath_flatness_threshold: float = 0.22
    chinese_merge_max_ms: int = 170


@dataclass(slots=True)
class SegmentFeatureSummary:
    """Summary of a provisional segment used for cutpoint refinement."""

    start: float
    end: float
    duration: float
    rms_mean: float
    rms_peak: float
    zcr_mean: float
    flatness_mean: float
    voiced_like: bool
    breath_like: bool
    silence_like: bool


def short_time_energy(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Return RMS energy over frames."""
    import librosa

    return librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]


def spectral_flux(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Return positive spectral flux over frames."""
    import librosa

    spectrum = np.abs(librosa.stft(samples, n_fft=frame_length, hop_length=hop_length))
    flux = np.maximum(0.0, np.diff(spectrum, axis=1)).sum(axis=0)
    return np.concatenate([np.zeros(1, dtype=np.float32), flux.astype(np.float32)])


def sanitize_cut_points(cut_points: list[float], duration: float, min_gap_sec: float) -> list[float]:
    """Sort, de-duplicate, clip, and merge cutpoints with a minimum gap."""
    if duration <= 0:
        return [0.0]

    cleaned = sorted({round(float(max(0.0, min(duration, point))), 6) for point in cut_points})
    result: list[float] = [0.0]
    for point in cleaned:
        if point <= 0.0 or point >= duration:
            continue
        if point - result[-1] >= min_gap_sec:
            result.append(point)
    if duration - result[-1] < min_gap_sec and len(result) > 1:
        result[-1] = round(duration, 6)
    else:
        result.append(round(duration, 6))
    if result[0] != 0.0:
        result.insert(0, 0.0)
    return result


def build_segments(cut_points: list[float], duration: float) -> list[Segment]:
    """Create sequential Segment objects from cutpoints."""
    cleaned = sanitize_cut_points(cut_points, duration, min_gap_sec=0.02)
    return [
        Segment(segment_id=f"seg_{index:03d}", start=float(start), end=float(end))
        for index, (start, end) in enumerate(zip(cleaned[:-1], cleaned[1:]), start=1)
    ]


def generate_candidate_cutpoints(
    samples: np.ndarray,
    sample_rate: int,
    config: CutpointConfig | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> list[float]:
    """Generate heuristic cutpoints from onset, flux, energy, and silence cues."""
    cfg = config or CutpointConfig()
    duration = float(samples.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if sample_rate <= 0 or samples.size == 0 or duration <= 0:
        return [0.0]

    _report_progress(progress_callback, 2, "Preparing audio for segmentation...")
    normalized = samples.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if peak > 0:
        normalized = normalized / peak

    import librosa

    _report_progress(progress_callback, 12, "Computing onset and energy features...")
    onset_env = librosa.onset.onset_strength(y=normalized, sr=sample_rate, hop_length=cfg.hop_length)
    rms = short_time_energy(normalized, frame_length=cfg.frame_length, hop_length=cfg.hop_length)
    flux = spectral_flux(normalized, frame_length=cfg.frame_length, hop_length=cfg.hop_length)
    zcr = librosa.feature.zero_crossing_rate(
        normalized,
        frame_length=cfg.frame_length,
        hop_length=cfg.hop_length,
    )[0].astype(np.float32)
    flatness = librosa.feature.spectral_flatness(
        y=normalized,
        n_fft=cfg.frame_length,
        hop_length=cfg.hop_length,
    )[0].astype(np.float32)

    _report_progress(progress_callback, 36, "Combining segmentation cues...")
    feature_length = min(onset_env.size, rms.size, flux.size, zcr.size, flatness.size)
    onset_env = onset_env[:feature_length]
    rms = rms[:feature_length]
    flux = flux[:feature_length]
    zcr = zcr[:feature_length]
    flatness = flatness[:feature_length]
    energy_diff = np.concatenate([np.zeros(1, dtype=np.float32), np.maximum(0.0, np.diff(rms))])

    combined = (
        0.42 * _normalize_feature(onset_env)
        + 0.30 * _normalize_feature(flux)
        + 0.18 * _normalize_feature(energy_diff)
        + 0.10 * _normalize_feature(np.maximum(0.0, rms - np.percentile(rms, 40)))
    )

    distance_frames = max(1, int(sample_rate * (cfg.min_gap_ms / 1000.0) / cfg.hop_length))
    prominence = max(0.08, float(np.percentile(combined, 84) * 0.55))
    _report_progress(progress_callback, 55, "Detecting onset and silence boundaries...")
    novelty_peaks, _ = find_peaks(combined, distance=distance_frames, prominence=prominence)
    onset_peaks = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=cfg.hop_length,
        backtrack=True,
        units="frames",
    )
    silence_boundaries = _detect_silence_boundaries(
        rms=rms,
        sample_rate=sample_rate,
        hop_length=cfg.hop_length,
        silence_percentile=cfg.silence_percentile,
        silence_min_ms=cfg.silence_min_ms,
    )

    candidate_frames = set(int(frame) for frame in novelty_peaks.tolist())
    candidate_frames.update(int(frame) for frame in onset_peaks.tolist())
    candidate_frames.update(silence_boundaries)

    if len(candidate_frames) > cfg.max_candidates:
        ranked = sorted(
            candidate_frames,
            key=lambda frame_index: float(combined[min(frame_index, combined.shape[0] - 1)]),
            reverse=True,
        )
        candidate_frames = set(ranked[: cfg.max_candidates])

    candidate_times = librosa.frames_to_time(
        sorted(candidate_frames),
        sr=sample_rate,
        hop_length=cfg.hop_length,
    ).tolist()
    _report_progress(progress_callback, 72, "Refining weak boundaries and breath fragments...")
    initial = sanitize_cut_points([0.0, *candidate_times, duration], duration, cfg.min_gap_ms / 1000.0)
    refined = refine_candidate_cutpoints(
        samples=normalized,
        sample_rate=sample_rate,
        cut_points=initial,
        config=cfg,
        combined_feature=combined,
        rms=rms,
        zcr=zcr,
        flatness=flatness,
    )
    _report_progress(progress_callback, 100, "Candidate cutpoints ready.")
    return refined


def refine_candidate_cutpoints(
    samples: np.ndarray,
    sample_rate: int,
    cut_points: list[float],
    config: CutpointConfig | None = None,
    *,
    combined_feature: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    zcr: np.ndarray | None = None,
    flatness: np.ndarray | None = None,
) -> list[float]:
    """Merge weak word-internal boundaries and breath over-segmentation."""
    cfg = config or CutpointConfig()
    duration = float(samples.shape[0] / sample_rate) if sample_rate > 0 else 0.0
    if sample_rate <= 0 or samples.size == 0 or duration <= 0:
        return [0.0]

    import librosa

    if rms is None:
        rms = short_time_energy(samples, frame_length=cfg.frame_length, hop_length=cfg.hop_length)
    if zcr is None:
        zcr = librosa.feature.zero_crossing_rate(
            samples,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )[0].astype(np.float32)
    if flatness is None:
        flatness = librosa.feature.spectral_flatness(
            y=samples,
            n_fft=cfg.frame_length,
            hop_length=cfg.hop_length,
        )[0].astype(np.float32)
    if combined_feature is None:
        combined_feature = _normalize_feature(rms)

    feature_length = min(rms.size, zcr.size, flatness.size, combined_feature.size)
    rms = rms[:feature_length]
    zcr = zcr[:feature_length]
    flatness = flatness[:feature_length]
    combined_feature = combined_feature[:feature_length]
    frame_times = librosa.frames_to_time(np.arange(feature_length), sr=sample_rate, hop_length=cfg.hop_length)

    silence_threshold = max(1e-4, float(np.percentile(rms, cfg.silence_percentile)))
    reference_energy = max(float(np.percentile(rms, 60)), silence_threshold * 2.0)

    points = sanitize_cut_points(cut_points, duration, cfg.min_gap_ms / 1000.0)
    for _ in range(max(1, len(points) * 2)):
        changed = False
        for boundary_index in range(1, len(points) - 1):
            left = _summarize_segment(points[boundary_index - 1], points[boundary_index], frame_times, rms, zcr, flatness, silence_threshold, reference_energy, cfg)
            right = _summarize_segment(points[boundary_index], points[boundary_index + 1], frame_times, rms, zcr, flatness, silence_threshold, reference_energy, cfg)
            boundary_strength = _boundary_strength(points[boundary_index], frame_times, combined_feature)
            boundary_is_silent = _boundary_is_silent(points[boundary_index], frame_times, rms, silence_threshold)

            if _should_merge_boundary(left, right, boundary_strength, boundary_is_silent, cfg):
                del points[boundary_index]
                changed = True
                break
        if not changed:
            break
    return sanitize_cut_points(points, duration, cfg.min_gap_ms / 1000.0)


def _normalize_feature(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    low = float(np.percentile(values, 5))
    high = float(np.percentile(values, 95))
    if high <= low:
        return np.zeros_like(values)
    normalized = (values - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def _detect_silence_boundaries(
    rms: np.ndarray,
    sample_rate: int,
    hop_length: int,
    silence_percentile: float,
    silence_min_ms: int,
) -> list[int]:
    """Locate transitions around longer low-energy spans."""
    if rms.size == 0:
        return []

    threshold = max(1e-4, float(np.percentile(rms, silence_percentile)))
    silent = rms <= threshold
    minimum_frames = max(2, int((sample_rate * (silence_min_ms / 1000.0)) / hop_length))

    boundaries: list[int] = []
    start_index: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and start_index is None:
            start_index = index
        elif not is_silent and start_index is not None:
            if index - start_index >= minimum_frames:
                if start_index > 0:
                    boundaries.append(start_index)
                if index < rms.shape[0] - 1:
                    boundaries.append(index)
            start_index = None

    if start_index is not None and silent.shape[0] - start_index >= minimum_frames and start_index > 0:
        boundaries.append(start_index)
    return boundaries


def _summarize_segment(
    start: float,
    end: float,
    frame_times: np.ndarray,
    rms: np.ndarray,
    zcr: np.ndarray,
    flatness: np.ndarray,
    silence_threshold: float,
    reference_energy: float,
    cfg: CutpointConfig,
) -> SegmentFeatureSummary:
    duration = max(0.0, end - start)
    if duration <= 0:
        return SegmentFeatureSummary(
            start=start,
            end=end,
            duration=0.0,
            rms_mean=0.0,
            rms_peak=0.0,
            zcr_mean=0.0,
            flatness_mean=0.0,
            voiced_like=False,
            breath_like=False,
            silence_like=True,
        )

    start_index = int(np.searchsorted(frame_times, start, side="left"))
    end_index = int(np.searchsorted(frame_times, end, side="right"))
    if end_index <= start_index:
        end_index = min(start_index + 1, frame_times.size)
    segment_rms = rms[start_index:end_index]
    segment_zcr = zcr[start_index:end_index]
    segment_flatness = flatness[start_index:end_index]
    if segment_rms.size == 0:
        segment_rms = np.asarray([0.0], dtype=np.float32)
        segment_zcr = np.asarray([0.0], dtype=np.float32)
        segment_flatness = np.asarray([1.0], dtype=np.float32)

    rms_mean = float(np.mean(segment_rms))
    rms_peak = float(np.max(segment_rms))
    zcr_mean = float(np.mean(segment_zcr))
    flatness_mean = float(np.mean(segment_flatness))
    energy_ratio = rms_mean / max(reference_energy, 1e-5)

    silence_like = rms_peak <= silence_threshold * 1.35
    breath_like = (
        duration <= cfg.breath_max_ms / 1000.0
        and energy_ratio <= cfg.breath_energy_ratio
        and (zcr_mean >= cfg.breath_zcr_threshold or flatness_mean >= cfg.breath_flatness_threshold)
        and not silence_like
    )
    voiced_like = (
        energy_ratio >= 0.72
        and zcr_mean <= cfg.breath_zcr_threshold * 1.1
        and flatness_mean <= cfg.breath_flatness_threshold * 1.05
    )

    return SegmentFeatureSummary(
        start=start,
        end=end,
        duration=duration,
        rms_mean=rms_mean,
        rms_peak=rms_peak,
        zcr_mean=zcr_mean,
        flatness_mean=flatness_mean,
        voiced_like=voiced_like,
        breath_like=breath_like,
        silence_like=silence_like,
    )


def _boundary_strength(boundary_time: float, frame_times: np.ndarray, combined_feature: np.ndarray) -> float:
    if frame_times.size == 0 or combined_feature.size == 0:
        return 0.0
    index = int(np.searchsorted(frame_times, boundary_time, side="left"))
    index = int(np.clip(index, 0, combined_feature.size - 1))
    left = max(0, index - 1)
    right = min(combined_feature.size, index + 2)
    return float(np.max(combined_feature[left:right]))


def _boundary_is_silent(boundary_time: float, frame_times: np.ndarray, rms: np.ndarray, silence_threshold: float) -> bool:
    if frame_times.size == 0 or rms.size == 0:
        return False
    index = int(np.searchsorted(frame_times, boundary_time, side="left"))
    index = int(np.clip(index, 0, rms.size - 1))
    left = max(0, index - 2)
    right = min(rms.size, index + 3)
    return bool(np.mean(rms[left:right]) <= silence_threshold * 1.25)


def _should_merge_boundary(
    left: SegmentFeatureSummary,
    right: SegmentFeatureSummary,
    boundary_strength: float,
    boundary_is_silent: bool,
    cfg: CutpointConfig,
) -> bool:
    if boundary_is_silent:
        return False

    micro_limit = cfg.short_segment_ms / 1000.0
    chinese_limit = cfg.chinese_merge_max_ms / 1000.0

    if left.breath_like and right.breath_like:
        return True

    if left.duration <= micro_limit and right.duration <= micro_limit and boundary_strength < cfg.weak_boundary_threshold + 0.08:
        return True

    if left.breath_like and right.voiced_like and left.duration <= cfg.breath_max_ms / 1000.0:
        return True

    if right.breath_like and left.voiced_like and right.duration <= cfg.breath_max_ms / 1000.0:
        return True

    if left.duration <= chinese_limit and not left.voiced_like and right.voiced_like and boundary_strength < cfg.weak_boundary_threshold + 0.10:
        return True

    if right.duration <= chinese_limit and not right.voiced_like and left.voiced_like and boundary_strength < cfg.weak_boundary_threshold:
        return True

    if left.duration <= micro_limit and boundary_strength < cfg.weak_boundary_threshold:
        return True

    if right.duration <= micro_limit and boundary_strength < cfg.weak_boundary_threshold:
        return True

    return False


def _report_progress(
    callback: Callable[[int, str], None] | None,
    value: int,
    message: str,
) -> None:
    if callback is None:
        return
    callback(int(np.clip(value, 0, 100)), message)
