from __future__ import annotations

from typing import Callable, Sequence

from humanslice.analysis.cutpoints import CutpointConfig, build_segments
from humanslice.analysis.regions import RegionConfig, analyze_segment_regions
from humanslice.analysis.scoring import ScoreConfig, score_segment
from humanslice.models.project import AudioTrack, Segment

# A rebuilt segment inherits alias / notes from the old segment it overlaps most,
# as long as the overlap covers at least this fraction of the longer of the two.
MATCH_MIN_OVERLAP_RATIO = 0.55
# Analysis results are reused only when both bounds moved less than this.
SAME_BOUNDS_TOLERANCE_SEC = 0.002


def min_gap_sec(config: CutpointConfig) -> float:
    """Minimum distance between two editable cutpoints, in seconds."""
    return max(0.02, config.min_gap_ms / 1000.0)


def rebuild_segments(
    cut_points: list[float],
    duration: float,
    previous_segments: Sequence[Segment] = (),
) -> list[Segment]:
    """Build segments from cutpoints, carrying user edits over from previous segments.

    Alias and notes follow the best-overlapping previous segment; regions and
    scores are kept only when the bounds are unchanged, so callers can analyze
    just the segments that still lack them.
    """
    segments = build_segments(cut_points, duration)
    for segment in segments:
        previous = match_previous_segment(segment, previous_segments)
        if previous is None:
            continue
        segment.alias = previous.alias
        segment.notes = previous.notes
        if same_bounds(segment, previous):
            segment.regions = previous.regions
            segment.score = previous.score
    return segments


def match_previous_segment(current: Segment, previous_segments: Sequence[Segment]) -> Segment | None:
    """Return the previous segment that overlaps `current` the most, if it overlaps enough."""
    best_segment: Segment | None = None
    best_ratio = 0.0
    for previous in previous_segments:
        overlap = max(0.0, min(current.end, previous.end) - max(current.start, previous.start))
        if overlap <= 0.0:
            continue
        ratio = overlap / max(current.duration, previous.duration, 1e-6)
        if ratio > best_ratio:
            best_ratio = ratio
            best_segment = previous
    return best_segment if best_ratio >= MATCH_MIN_OVERLAP_RATIO else None


def same_bounds(left: Segment, right: Segment) -> bool:
    return (
        abs(left.start - right.start) <= SAME_BOUNDS_TOLERANCE_SEC
        and abs(left.end - right.end) <= SAME_BOUNDS_TOLERANCE_SEC
    )


def segments_needing_analysis(segments: Sequence[Segment]) -> list[Segment]:
    return [segment for segment in segments if segment.regions is None or segment.score is None]


def analyze_segments(
    track: AudioTrack,
    segments: Sequence[Segment],
    region_config: RegionConfig,
    score_config: ScoreConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> None:
    """Fill in regions and scores in place; reports `(done, total)` after each segment."""
    total = len(segments)
    for index, segment in enumerate(segments, start=1):
        segment.regions = analyze_segment_regions(
            track.samples,
            track.sample_rate,
            segment.start,
            segment.end,
            region_config,
        )
        segment.score = score_segment(
            track.samples,
            track.sample_rate,
            segment,
            segment.regions,
            score_config,
        )
        if progress_callback is not None:
            progress_callback(index, total)


def segment_index_for_time(segments: Sequence[Segment], time_value: float) -> int:
    """Index of the segment containing `time_value`, falling back to the last one."""
    for index, segment in enumerate(segments):
        if segment.start <= time_value <= segment.end:
            return index
    return max(0, len(segments) - 1)


def nearest_cutpoint_index(cut_points: Sequence[float], time_value: float) -> int:
    if not cut_points:
        return 0
    return min(range(len(cut_points)), key=lambda index: abs(cut_points[index] - time_value))
