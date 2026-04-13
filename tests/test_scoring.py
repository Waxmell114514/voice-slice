from __future__ import annotations

import numpy as np

from analysis.regions import analyze_segment_regions
from analysis.scoring import ScoreConfig, compute_f0_curve, score_segment
from models.project import Segment


def test_compute_f0_curve_tracks_simple_sine_with_pyin() -> None:
    sample_rate = 22050
    duration = 0.55
    t = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    _, f0 = compute_f0_curve(samples, sample_rate, ScoreConfig(f0_backend="pyin"))
    voiced = f0[np.isfinite(f0)]

    assert voiced.size > 10
    assert abs(float(np.median(voiced)) - 220.0) < 8.0


def test_score_segment_prefers_stable_voiced_material() -> None:
    sample_rate = 22050
    duration = 0.48
    samples_count = int(duration * sample_rate)
    t = np.arange(samples_count, dtype=np.float32) / sample_rate
    envelope = np.concatenate(
        [
            np.linspace(0.0, 1.0, int(0.08 * sample_rate), endpoint=False, dtype=np.float32),
            np.ones(int(0.26 * sample_rate), dtype=np.float32),
            np.linspace(1.0, 0.0, samples_count - int(0.34 * sample_rate), endpoint=False, dtype=np.float32),
        ]
    )

    stable = envelope * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    sweep_freq = np.linspace(180.0, 320.0, samples_count, dtype=np.float32)
    phase = 2 * np.pi * np.cumsum(sweep_freq) / sample_rate
    unstable = envelope * np.sin(phase).astype(np.float32)

    stable_segment = Segment("seg_001", 0.0, duration)
    unstable_segment = Segment("seg_002", 0.0, duration)

    stable_regions = analyze_segment_regions(stable, sample_rate, 0.0, duration)
    unstable_regions = analyze_segment_regions(unstable, sample_rate, 0.0, duration)
    stable_score = score_segment(stable, sample_rate, stable_segment, stable_regions)
    unstable_score = score_segment(unstable, sample_rate, unstable_segment, unstable_regions)

    assert 0.0 <= stable_score.clarity_score <= 100.0
    assert 0.0 <= stable_score.stability_score <= 100.0
    assert stable_score.stability_score > unstable_score.stability_score
    assert stable_score.recommended_role in {"sustain", "general"}
