from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from humanslice.analysis.regions import RegionConfig, analyze_segment_regions
from humanslice.audio.io import extract_clip
from humanslice.models.project import Segment, SegmentRegions, SegmentScore


@dataclass(slots=True)
class ScoreConfig:
    """Parameters for heuristic segment scoring and F0 preview generation."""

    f0_backend: str = "pyin"
    hop_length: int = 256
    f0_frame_length: int = 2048
    pitch_floor_hz: float = 65.0
    pitch_ceiling_hz: float = 1200.0
    voiced_prob_threshold: float = 0.55
    f0_smoothing_frames: int = 5
    octave_jump_threshold_cents: float = 650.0
    preferred_duration_sec: float = 0.28
    preview_max_sample_rate: int = 12000
    preview_max_points: int = 3200


@dataclass(slots=True)
class F0Track:
    """Pitch track with voiced-state confidence."""

    times: np.ndarray
    values: np.ndarray
    voiced_flag: np.ndarray
    voiced_prob: np.ndarray


def compute_f0_curve(
    samples: np.ndarray,
    sample_rate: int,
    config: ScoreConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate an F0 curve suitable for segment scoring."""
    cfg = config or ScoreConfig()
    track = _compute_f0_track_internal(
        samples=samples,
        sample_rate=sample_rate,
        frame_length=cfg.f0_frame_length,
        hop_length=cfg.hop_length,
        config=cfg,
    )
    return track.times, track.values


def compute_preview_f0_curve(
    samples: np.ndarray,
    sample_rate: int,
    config: ScoreConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a lighter F0 curve for whole-track preview.

    For long materials this downsamples the signal and increases hop size so
    importing audio stays responsive while still providing a useful pitch view.
    """

    cfg = config or ScoreConfig()
    if sample_rate <= 0 or samples.size < cfg.hop_length * 2:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    preview_samples = samples.astype(np.float32, copy=False)
    preview_sr = int(sample_rate)
    if cfg.preview_max_sample_rate > 0 and preview_sr > cfg.preview_max_sample_rate:
        import librosa

        preview_samples = librosa.resample(
            preview_samples,
            orig_sr=preview_sr,
            target_sr=cfg.preview_max_sample_rate,
            res_type="soxr_hq",
        ).astype(np.float32, copy=False)
        preview_sr = int(cfg.preview_max_sample_rate)

    if preview_samples.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    target_hop = max(cfg.hop_length * max(1, sample_rate // max(preview_sr, 1)), cfg.hop_length)
    if cfg.preview_max_points > 0:
        adaptive_hop = int(np.ceil(preview_samples.size / cfg.preview_max_points))
        target_hop = max(target_hop, adaptive_hop)

    frame_length = max(1024, 1 << int(np.ceil(np.log2(max(target_hop * 4, 1024)))))
    track = _compute_f0_track_internal(
        samples=preview_samples,
        sample_rate=preview_sr,
        frame_length=frame_length,
        hop_length=target_hop,
        config=cfg,
    )
    return track.times, track.values


def score_segment(
    samples: np.ndarray,
    sample_rate: int,
    segment: Segment,
    regions: SegmentRegions | None = None,
    config: ScoreConfig | None = None,
) -> SegmentScore:
    """Score a segment for clarity and stability using lightweight heuristics."""
    cfg = config or ScoreConfig()
    region_data = regions or segment.regions or analyze_segment_regions(samples, sample_rate, segment.start, segment.end, RegionConfig())
    clip = extract_clip(samples, sample_rate, segment.start, segment.end)
    duration = segment.duration
    if clip.size == 0 or duration <= 0:
        return SegmentScore(clarity_score=0.0, stability_score=0.0, recommended_role="weak")

    onset_clip = extract_clip(samples, sample_rate, region_data.onset.start, region_data.onset.end)
    nucleus_clip = extract_clip(samples, sample_rate, region_data.nucleus.start, region_data.nucleus.end)
    tail_clip = extract_clip(samples, sample_rate, region_data.tail.start, region_data.tail.end)

    onset_env = _rms_curve(onset_clip, cfg.f0_frame_length, cfg.hop_length)
    nucleus_env = _rms_curve(nucleus_clip, cfg.f0_frame_length, cfg.hop_length)
    tail_env = _rms_curve(tail_clip, cfg.f0_frame_length, cfg.hop_length)

    onset_attack = 0.0
    if onset_env.size > 1:
        onset_attack = float(np.max(np.maximum(0.0, np.diff(onset_env))))
        onset_attack = float(np.clip(onset_attack / max(np.max(onset_env), 1e-5), 0.0, 1.0))

    nucleus_mean = float(np.mean(nucleus_env)) if nucleus_env.size else 0.0
    nucleus_cv = float(np.std(nucleus_env) / max(nucleus_mean, 1e-5)) if nucleus_env.size else 1.0
    tail_mean = float(np.mean(tail_env)) if tail_env.size else 0.0
    silence_ratio = float(np.mean(np.abs(clip) < 0.01))

    duration_score = float(np.exp(-((duration - cfg.preferred_duration_sec) ** 2) / (2 * 0.20**2)))
    tail_penalty = float(np.clip(tail_mean / max(nucleus_mean, 1e-5), 0.0, 1.25) / 1.25)
    silence_penalty = float(np.clip(silence_ratio / 0.45, 0.0, 1.0))

    clarity_score = 100.0 * (
        0.34 * onset_attack
        + 0.22 * (1.0 - tail_penalty)
        + 0.22 * (1.0 - silence_penalty)
        + 0.22 * duration_score
    )

    f0_track = _compute_f0_track_internal(nucleus_clip, sample_rate, cfg.f0_frame_length, cfg.hop_length, cfg)
    voiced_mask = np.isfinite(f0_track.values)
    voiced = f0_track.values[voiced_mask]
    voiced_ratio = float(np.mean(f0_track.voiced_prob >= cfg.voiced_prob_threshold)) if f0_track.values.size else 0.0
    voiced_confidence = (
        float(np.mean(f0_track.voiced_prob[voiced_mask]))
        if np.any(voiced_mask)
        else 0.0
    )
    if voiced.size >= 3:
        voiced_cents = _hz_to_cents(voiced)
        median_cents = float(np.median(voiced_cents))
        pitch_deviation = float(np.median(np.abs(voiced_cents - median_cents)))
        f0_stability = float(np.clip(1.0 - (pitch_deviation / 45.0), 0.0, 1.0))
    else:
        f0_stability = 0.25

    octave_jump_penalty = _octave_jump_penalty(f0_track.values, cfg.octave_jump_threshold_cents)
    energy_stability = float(np.clip(1.0 - (nucleus_cv / 0.45), 0.0, 1.0))
    stability_score = 100.0 * (
        0.44 * f0_stability
        + 0.24 * energy_stability
        + 0.14 * duration_score
        + 0.18 * voiced_confidence
    )
    stability_score *= 1.0 - min(0.35, octave_jump_penalty * 0.35)

    recommended_role = _recommend_role(
        duration=duration,
        clarity_score=clarity_score,
        stability_score=stability_score,
        onset_attack=onset_attack,
        voiced_ratio=voiced_ratio,
    )

    return SegmentScore(
        clarity_score=float(np.clip(clarity_score, 0.0, 100.0)),
        stability_score=float(np.clip(stability_score, 0.0, 100.0)),
        recommended_role=recommended_role,
        onset_strength=onset_attack,
        voiced_ratio=voiced_ratio,
        noise_penalty=max(tail_penalty, silence_penalty, octave_jump_penalty),
    )


def _compute_f0_track_internal(
    samples: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    config: ScoreConfig,
) -> F0Track:
    if sample_rate <= 0 or samples.size < max(hop_length, 64) * 2:
        empty = np.zeros(0, dtype=np.float32)
        return F0Track(times=empty, values=empty, voiced_flag=np.zeros(0, dtype=bool), voiced_prob=empty)

    import librosa

    frame_length = int(min(max(frame_length, 512), max(samples.size, 512)))
    hop_length = int(max(64, hop_length))
    fmin = float(max(25.0, config.pitch_floor_hz))
    nyquist_margin = max(fmin * 1.2, 80.0)
    fmax = float(min(config.pitch_ceiling_hz, (sample_rate / 2.0) - nyquist_margin))
    if fmax <= fmin:
        fmax = min(sample_rate / 2.0 - 10.0, max(fmin + 40.0, fmin * 2.0))

    backend = (config.f0_backend or "pyin").lower().strip()
    if backend == "yin":
        f0, voiced_flag, voiced_prob = _compute_yin_track(samples, sample_rate, frame_length, hop_length, fmin, fmax)
    else:
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                samples,
                fmin=fmin,
                fmax=fmax,
                sr=sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
                fill_na=np.nan,
            )
        except Exception:
            f0, voiced_flag, voiced_prob = _compute_yin_track(
                samples,
                sample_rate,
                frame_length,
                hop_length,
                fmin,
                fmax,
            )
        else:
            f0 = np.asarray(f0, dtype=np.float32)
            voiced_flag = np.asarray(voiced_flag, dtype=bool)
            voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length).astype(np.float32)
    values, voiced_flag, voiced_prob = _postprocess_f0_track(
        values=f0,
        voiced_flag=voiced_flag,
        voiced_prob=voiced_prob,
        threshold=config.voiced_prob_threshold,
        smoothing_frames=config.f0_smoothing_frames,
    )
    return F0Track(times=times, values=values, voiced_flag=voiced_flag, voiced_prob=voiced_prob)


def _recommend_role(
    duration: float,
    clarity_score: float,
    stability_score: float,
    onset_attack: float,
    voiced_ratio: float,
) -> str:
    if duration < 0.05 or clarity_score < 25.0:
        return "weak"
    if onset_attack >= 0.55 and duration <= 0.28 and stability_score < 72.0:
        return "onset"
    if stability_score >= 62.0 and voiced_ratio >= 0.30 and duration >= 0.10:
        return "sustain"
    return "general"


def _rms_curve(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32)
    import librosa

    frame_length = min(frame_length, max(256, samples.size))
    return librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0].astype(np.float32)


def _compute_yin_track(
    samples: np.ndarray,
    sample_rate: int,
    frame_length: int,
    hop_length: int,
    fmin: float,
    fmax: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback F0 estimation using plain YIN plus energy-derived confidence."""
    import librosa

    values = librosa.yin(
        samples,
        fmin=fmin,
        fmax=fmax,
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
    ).astype(np.float32)
    rms = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0].astype(np.float32)
    energy_floor = max(1e-4, float(np.percentile(rms, 25)))
    energy_ceiling = max(energy_floor * 1.5, float(np.percentile(rms, 90)))
    voiced_prob = np.clip((rms - energy_floor) / max(energy_ceiling - energy_floor, 1e-5), 0.0, 1.0)
    voiced_flag = voiced_prob >= 0.5
    values[~voiced_flag] = np.nan
    return values, voiced_flag, voiced_prob.astype(np.float32)


def _postprocess_f0_track(
    values: np.ndarray,
    voiced_flag: np.ndarray,
    voiced_prob: np.ndarray,
    threshold: float,
    smoothing_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_prob = np.asarray(voiced_prob, dtype=np.float32)

    if values.size == 0:
        return values, voiced_flag, voiced_prob

    if voiced_flag.size != values.size:
        voiced_flag = np.resize(voiced_flag, values.size).astype(bool)
    if voiced_prob.size != values.size:
        voiced_prob = np.resize(voiced_prob, values.size).astype(np.float32)

    finite_mask = np.isfinite(values)
    reliable_mask = finite_mask & voiced_flag & (voiced_prob >= threshold)
    cleaned = values.copy()
    cleaned[~reliable_mask] = np.nan

    if np.count_nonzero(reliable_mask) >= 3:
        cents = _hz_to_cents(cleaned[reliable_mask])
        if smoothing_frames > 1:
            kernel = int(max(1, smoothing_frames))
            if kernel % 2 == 0:
                kernel += 1
            filtered = signal.medfilt(cents, kernel_size=kernel)
        else:
            filtered = cents
        cleaned[reliable_mask] = _cents_to_hz(filtered)

    voiced_flag = np.isfinite(cleaned)
    voiced_prob = np.where(voiced_flag, voiced_prob, 0.0).astype(np.float32)
    return cleaned.astype(np.float32, copy=False), voiced_flag, voiced_prob


def _hz_to_cents(values: np.ndarray, reference_hz: float = 55.0) -> np.ndarray:
    safe = np.maximum(np.asarray(values, dtype=np.float32), 1e-5)
    return (1200.0 * np.log2(safe / reference_hz)).astype(np.float32, copy=False)


def _cents_to_hz(values: np.ndarray, reference_hz: float = 55.0) -> np.ndarray:
    return (reference_hz * np.power(2.0, np.asarray(values, dtype=np.float32) / 1200.0)).astype(np.float32, copy=False)


def _octave_jump_penalty(values: np.ndarray, jump_threshold_cents: float) -> float:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size < 3:
        return 0.0

    cents = _hz_to_cents(finite)
    jumps = np.abs(np.diff(cents))
    if jumps.size == 0:
        return 0.0
    jump_ratio = float(np.mean(jumps >= jump_threshold_cents))
    return float(np.clip(jump_ratio, 0.0, 1.0))
