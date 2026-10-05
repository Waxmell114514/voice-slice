from __future__ import annotations

from humanslice.analysis.cutpoints import CutpointConfig
from humanslice.models.project import RegionRange, Segment, SegmentRegions, SegmentScore
from humanslice.services.segment_service import (
    min_gap_sec,
    nearest_cutpoint_index,
    rebuild_segments,
    segment_index_for_time,
    segments_needing_analysis,
)


def _analyzed_segment(segment_id: str, start: float, end: float, alias: str) -> Segment:
    return Segment(
        segment_id,
        start,
        end,
        alias=alias,
        notes=f"{alias} notes",
        regions=SegmentRegions(
            onset=RegionRange(start, start + 0.05),
            nucleus=RegionRange(start + 0.05, end - 0.05),
            tail=RegionRange(end - 0.05, end),
        ),
        score=SegmentScore(clarity_score=60.0, stability_score=70.0, recommended_role="general"),
    )


def test_rebuild_segments_keeps_analysis_for_unchanged_bounds() -> None:
    previous = [_analyzed_segment("seg_001", 0.0, 0.5, "a"), _analyzed_segment("seg_002", 0.5, 1.0, "i")]

    rebuilt = rebuild_segments([0.0, 0.5, 1.0], 1.0, previous)

    assert [segment.alias for segment in rebuilt] == ["a", "i"]
    assert [segment.notes for segment in rebuilt] == ["a notes", "i notes"]
    assert rebuilt[0].score is previous[0].score
    assert rebuilt[1].regions is previous[1].regions
    assert segments_needing_analysis(rebuilt) == []


def test_rebuild_segments_drops_analysis_but_keeps_labels_when_bounds_move() -> None:
    previous = [_analyzed_segment("seg_001", 0.0, 0.5, "a"), _analyzed_segment("seg_002", 0.5, 1.0, "i")]

    rebuilt = rebuild_segments([0.0, 0.55, 1.0], 1.0, previous)

    assert [segment.alias for segment in rebuilt] == ["a", "i"]
    assert all(segment.score is None and segment.regions is None for segment in rebuilt)
    assert segments_needing_analysis(rebuilt) == rebuilt


def test_rebuild_segments_does_not_inherit_from_weak_overlap() -> None:
    previous = [_analyzed_segment("seg_001", 0.0, 1.0, "a")]

    # Splitting in the middle: each half overlaps the old segment by exactly 50%, below the threshold.
    rebuilt = rebuild_segments([0.0, 0.5, 1.0], 1.0, previous)

    assert [segment.alias for segment in rebuilt] == ["", ""]


def test_segment_and_cutpoint_lookup() -> None:
    segments = rebuild_segments([0.0, 0.3, 0.7, 1.0], 1.0)

    assert segment_index_for_time(segments, 0.1) == 0
    assert segment_index_for_time(segments, 0.5) == 1
    assert segment_index_for_time(segments, 5.0) == 2
    assert segment_index_for_time([], 0.5) == 0
    assert nearest_cutpoint_index([0.0, 0.3, 0.7, 1.0], 0.66) == 2
    assert nearest_cutpoint_index([], 0.5) == 0


def test_min_gap_sec_has_floor() -> None:
    assert min_gap_sec(CutpointConfig(min_gap_ms=120)) == 0.12
    assert min_gap_sec(CutpointConfig(min_gap_ms=0)) == 0.02
