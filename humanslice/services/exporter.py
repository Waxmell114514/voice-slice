from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import soundfile as sf

from humanslice.audio.io import extract_clip
from humanslice.models.project import AudioTrack, Segment


@dataclass(slots=True)
class UtauOtoEntry:
    """A heuristic OTO entry for UTAU-like tooling."""

    file_name: str
    alias: str
    offset_ms: float
    consonant_ms: float
    cutoff_ms: float
    preutterance_ms: float
    overlap_ms: float

    def to_oto_line(self) -> str:
        return (
            f"{self.file_name}={self.alias},"
            f"{self.offset_ms:.2f},{self.consonant_ms:.2f},{self.cutoff_ms:.2f},"
            f"{self.preutterance_ms:.2f},{self.overlap_ms:.2f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "utau_alias": self.alias,
            "utau_offset_ms": round(self.offset_ms, 3),
            "utau_consonant_ms": round(self.consonant_ms, 3),
            "utau_cutoff_ms": round(self.cutoff_ms, 3),
            "utau_preutterance_ms": round(self.preutterance_ms, 3),
            "utau_overlap_ms": round(self.overlap_ms, 3),
        }

    def to_label_row(self) -> dict[str, Any]:
        """Row for utau_labels.csv; keys match `_utau_fieldnames()`."""
        return {
            "file_name": self.file_name,
            "alias": self.alias,
            "offset_ms": round(self.offset_ms, 3),
            "consonant_ms": round(self.consonant_ms, 3),
            "cutoff_ms": round(self.cutoff_ms, 3),
            "preutterance_ms": round(self.preutterance_ms, 3),
            "overlap_ms": round(self.overlap_ms, 3),
        }


def segments_to_metadata(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    """Flatten segments into export-ready metadata rows."""
    return [_metadata_row(segment, build_utau_oto_entry(segment, _wav_file_name(segment))) for segment in segments]


def export_segments(
    track: AudioTrack,
    segments: Sequence[Segment],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Export wav files plus JSON / CSV metadata and UTAU-friendly files."""
    base_dir = Path(output_dir)
    segment_dir = base_dir / "segments"
    segment_dir.mkdir(parents=True, exist_ok=True)

    oto_entries = [build_utau_oto_entry(segment, _wav_file_name(segment)) for segment in segments]
    rows: list[dict[str, Any]] = []
    for segment, oto_entry in zip(segments, oto_entries):
        clip = extract_clip(track.samples, track.sample_rate, segment.start, segment.end)
        sf.write(segment_dir / oto_entry.file_name, clip, track.sample_rate, subtype="PCM_16")
        rows.append({**_metadata_row(segment, oto_entry), "file_name": oto_entry.file_name})

    json_path = base_dir / "metadata.json"
    csv_path = base_dir / "metadata.csv"
    oto_path = base_dir / "oto.ini"
    utau_csv_path = base_dir / "utau_labels.csv"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, list(rows[0].keys()) if rows else _default_fieldnames(), rows)
    oto_path.write_text("\n".join(entry.to_oto_line() for entry in oto_entries), encoding="utf-8")
    _write_csv(utau_csv_path, _utau_fieldnames(), [entry.to_label_row() for entry in oto_entries])

    return {
        "segments_dir": segment_dir,
        "metadata_json": json_path,
        "metadata_csv": csv_path,
        "oto_ini": oto_path,
        "utau_csv": utau_csv_path,
    }


def build_utau_oto_entry(segment: Segment, file_name: str) -> UtauOtoEntry:
    """Build a lightweight OTO.ini entry from HumanSlice region heuristics."""
    duration_ms = max(segment.duration * 1000.0, 1.0)
    alias = segment.alias.strip() or segment.segment_id

    onset_ms = 0.0
    tail_ms = max(20.0, duration_ms * 0.15)
    nucleus_ms = max(duration_ms - tail_ms, 20.0)
    if segment.regions is not None:
        onset_ms = max((segment.regions.onset.end - segment.regions.onset.start) * 1000.0, 0.0)
        nucleus_ms = max((segment.regions.nucleus.end - segment.regions.nucleus.start) * 1000.0, 20.0)
        tail_ms = max((segment.regions.tail.end - segment.regions.tail.start) * 1000.0, 15.0)

    offset_ms = 0.0
    consonant_ms = min(duration_ms - 5.0, max(25.0, onset_ms + min(50.0, nucleus_ms * 0.25)))
    preutterance_ms = min(consonant_ms, max(15.0, onset_ms * 0.85 if onset_ms > 0 else consonant_ms * 0.7))
    overlap_ms = min(preutterance_ms * 0.55, max(5.0, onset_ms * 0.45 if onset_ms > 0 else preutterance_ms * 0.45))
    cutoff_ms = -min(duration_ms - 1.0, max(15.0, tail_ms))

    return UtauOtoEntry(
        file_name=file_name,
        alias=alias,
        offset_ms=offset_ms,
        consonant_ms=consonant_ms,
        cutoff_ms=cutoff_ms,
        preutterance_ms=preutterance_ms,
        overlap_ms=overlap_ms,
    )


def _wav_file_name(segment: Segment) -> str:
    return f"{segment.segment_id}.wav"


def _metadata_row(segment: Segment, oto_entry: UtauOtoEntry) -> dict[str, Any]:
    return {**segment.to_metadata_row(), **oto_entry.to_dict()}


def _write_csv(path: Path, fieldnames: list[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _default_fieldnames() -> list[str]:
    return [
        "segment_id",
        "alias",
        "notes",
        "file_name",
        "start",
        "end",
        "duration",
        "clarity_score",
        "stability_score",
        "recommended_role",
        "onset_start",
        "onset_end",
        "nucleus_start",
        "nucleus_end",
        "tail_start",
        "tail_end",
        "utau_alias",
        "utau_offset_ms",
        "utau_consonant_ms",
        "utau_cutoff_ms",
        "utau_preutterance_ms",
        "utau_overlap_ms",
    ]


def _utau_fieldnames() -> list[str]:
    return [
        "file_name",
        "alias",
        "offset_ms",
        "consonant_ms",
        "cutoff_ms",
        "preutterance_ms",
        "overlap_ms",
    ]
