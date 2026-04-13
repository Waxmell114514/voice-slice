from __future__ import annotations

import numpy as np

from analysis.regions import analyze_segment_regions


def test_analyze_segment_regions_returns_ordered_subregions() -> None:
    sample_rate = 22050
    attack = np.linspace(0.0, 1.0, int(0.08 * sample_rate), endpoint=False, dtype=np.float32)
    sustain = np.ones(int(0.22 * sample_rate), dtype=np.float32)
    decay = np.linspace(1.0, 0.0, int(0.14 * sample_rate), endpoint=False, dtype=np.float32)
    envelope = np.concatenate([attack, sustain, decay])
    t = np.arange(envelope.size, dtype=np.float32) / sample_rate
    samples = envelope * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    regions = analyze_segment_regions(samples, sample_rate, 0.0, envelope.size / sample_rate)

    assert regions.onset.start == 0.0
    assert regions.onset.end < regions.nucleus.end < regions.tail.end
    assert regions.nucleus.start == regions.onset.end
    assert regions.tail.start == regions.nucleus.end
    assert (regions.nucleus.end - regions.nucleus.start) > (regions.onset.end - regions.onset.start)
