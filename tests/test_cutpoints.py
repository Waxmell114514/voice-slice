from __future__ import annotations

import numpy as np

from analysis.cutpoints import CutpointConfig, generate_candidate_cutpoints, refine_candidate_cutpoints


def test_generate_candidate_cutpoints_detects_multiple_regions() -> None:
    sample_rate = 22050
    silence = np.zeros(int(0.12 * sample_rate), dtype=np.float32)

    def burst(freq: float) -> np.ndarray:
        t = np.linspace(0.0, 0.18, int(0.18 * sample_rate), endpoint=False, dtype=np.float32)
        envelope = np.linspace(0.0, 1.0, t.size, dtype=np.float32)
        return 0.8 * envelope * np.sin(2 * np.pi * freq * t).astype(np.float32)

    samples = np.concatenate([silence, burst(220.0), silence, burst(330.0), silence, burst(440.0)])
    cut_points = generate_candidate_cutpoints(
        samples,
        sample_rate,
        CutpointConfig(min_gap_ms=90),
    )

    assert cut_points[0] == 0.0
    assert cut_points[-1] > 0.80
    assert len(cut_points) >= 4
    internal = cut_points[1:-1]
    assert any(0.22 <= point <= 0.38 for point in internal)
    assert any(0.52 <= point <= 0.72 for point in internal)


def test_refine_candidate_cutpoints_merges_breath_subsegments() -> None:
    sample_rate = 22050
    t_voice = np.arange(int(0.22 * sample_rate), dtype=np.float32) / sample_rate
    voice_a = 0.5 * np.sin(2 * np.pi * 220.0 * t_voice).astype(np.float32)
    voice_b = 0.5 * np.sin(2 * np.pi * 246.0 * t_voice).astype(np.float32)
    breath_a = 0.06 * np.random.default_rng(0).normal(size=int(0.08 * sample_rate)).astype(np.float32)
    breath_b = 0.05 * np.random.default_rng(1).normal(size=int(0.06 * sample_rate)).astype(np.float32)
    samples = np.concatenate([voice_a, breath_a, breath_b, voice_b])

    refined = refine_candidate_cutpoints(
        samples=samples,
        sample_rate=sample_rate,
        cut_points=[0.0, 0.22, 0.30, 0.36, samples.size / sample_rate],
        config=CutpointConfig(),
    )

    internal = refined[1:-1]
    assert any(abs(point - 0.22) < 0.02 for point in internal)
    assert any(abs(point - 0.36) < 0.02 for point in internal)
    assert not any(abs(point - 0.30) < 0.02 for point in internal)
