from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import numpy as np

from models.project import AudioTrack, RegionRange, Segment, SegmentRegions, SegmentScore
from services.exporter import export_segments


def test_export_segments_writes_audio_and_metadata() -> None:
    sample_rate = 22050
    duration = 0.25
    samples = 0.5 * np.sin(2 * np.pi * 220.0 * np.arange(int(duration * sample_rate)) / sample_rate).astype(np.float32)
    track = AudioTrack(path="synthetic.wav", samples=samples, sample_rate=sample_rate, duration=duration)
    output_dir = Path(f".test_export_{uuid.uuid4().hex}")

    try:
        segment = Segment(
            segment_id="seg_001",
            start=0.0,
            end=duration,
            alias="a",
            notes="main vowel",
            regions=SegmentRegions(
                onset=RegionRange(0.0, 0.05),
                nucleus=RegionRange(0.05, 0.18),
                tail=RegionRange(0.18, duration),
            ),
            score=SegmentScore(clarity_score=72.0, stability_score=83.0, recommended_role="sustain"),
        )

        result = export_segments(track, [segment], output_dir)

        assert (result["segments_dir"] / "seg_001.wav").exists()
        assert result["metadata_json"].exists()
        assert result["metadata_csv"].exists()
        assert result["oto_ini"].exists()
        assert result["utau_csv"].exists()

        metadata = json.loads(result["metadata_json"].read_text(encoding="utf-8"))
        assert metadata[0]["segment_id"] == "seg_001"
        assert metadata[0]["alias"] == "a"
        assert metadata[0]["notes"] == "main vowel"
        assert metadata[0]["recommended_role"] == "sustain"
        assert metadata[0]["utau_alias"] == "a"

        oto_lines = result["oto_ini"].read_text(encoding="utf-8").splitlines()
        assert oto_lines[0].startswith("seg_001.wav=a,")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
