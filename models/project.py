from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class RegionRange:
    """A time span inside a segment."""

    start: float
    end: float

    def to_dict(self) -> dict[str, float]:
        return {"start": round(float(self.start), 6), "end": round(float(self.end), 6)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegionRange":
        return cls(start=float(data["start"]), end=float(data["end"]))


@dataclass(slots=True)
class SegmentRegions:
    """Heuristic onset / nucleus / tail partition for a segment."""

    onset: RegionRange
    nucleus: RegionRange
    tail: RegionRange

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {
            "onset": self.onset.to_dict(),
            "nucleus": self.nucleus.to_dict(),
            "tail": self.tail.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentRegions":
        return cls(
            onset=RegionRange.from_dict(data["onset"]),
            nucleus=RegionRange.from_dict(data["nucleus"]),
            tail=RegionRange.from_dict(data["tail"]),
        )


@dataclass(slots=True)
class SegmentScore:
    """Quality scores computed from heuristic audio features."""

    clarity_score: float
    stability_score: float
    recommended_role: str
    onset_strength: float = 0.0
    voiced_ratio: float = 0.0
    noise_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "clarity_score": round(float(self.clarity_score), 4),
            "stability_score": round(float(self.stability_score), 4),
            "recommended_role": self.recommended_role,
            "onset_strength": round(float(self.onset_strength), 4),
            "voiced_ratio": round(float(self.voiced_ratio), 4),
            "noise_penalty": round(float(self.noise_penalty), 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentScore":
        return cls(
            clarity_score=float(data["clarity_score"]),
            stability_score=float(data["stability_score"]),
            recommended_role=str(data["recommended_role"]),
            onset_strength=float(data.get("onset_strength", 0.0)),
            voiced_ratio=float(data.get("voiced_ratio", 0.0)),
            noise_penalty=float(data.get("noise_penalty", 0.0)),
        )


@dataclass(slots=True)
class Segment:
    """A user-editable slice inside the loaded audio file."""

    segment_id: str
    start: float
    end: float
    regions: SegmentRegions | None = None
    score: SegmentScore | None = None
    alias: str = ""
    notes: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end - self.start))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "start": round(float(self.start), 6),
            "end": round(float(self.end), 6),
            "alias": self.alias,
            "notes": self.notes,
            "regions": self.regions.to_dict() if self.regions else None,
            "score": self.score.to_dict() if self.score else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        regions = data.get("regions")
        score = data.get("score")
        return cls(
            segment_id=str(data["segment_id"]),
            start=float(data["start"]),
            end=float(data["end"]),
            alias=str(data.get("alias", "")),
            notes=str(data.get("notes", "")),
            regions=SegmentRegions.from_dict(regions) if regions else None,
            score=SegmentScore.from_dict(score) if score else None,
        )

    def to_metadata_row(self) -> dict[str, Any]:
        """Flatten segment data for JSON / CSV metadata export."""
        regions = self.regions
        score = self.score
        return {
            "segment_id": self.segment_id,
            "alias": self.alias or self.segment_id,
            "notes": self.notes,
            "start": round(self.start, 6),
            "end": round(self.end, 6),
            "duration": round(self.duration, 6),
            "clarity_score": round(score.clarity_score, 4) if score else None,
            "stability_score": round(score.stability_score, 4) if score else None,
            "recommended_role": score.recommended_role if score else None,
            "onset_start": round(regions.onset.start, 6) if regions else None,
            "onset_end": round(regions.onset.end, 6) if regions else None,
            "nucleus_start": round(regions.nucleus.start, 6) if regions else None,
            "nucleus_end": round(regions.nucleus.end, 6) if regions else None,
            "tail_start": round(regions.tail.start, 6) if regions else None,
            "tail_end": round(regions.tail.end, 6) if regions else None,
        }


@dataclass(slots=True)
class AudioTrack:
    """Loaded audio data kept in memory for editing and preview."""

    path: str
    samples: np.ndarray
    sample_rate: int
    duration: float

    @property
    def display_name(self) -> str:
        return Path(self.path).name


@dataclass(slots=True)
class ProjectData:
    """Serializable project state for saving and reopening a session."""

    name: str = "HumanSlice"
    audio_path: str | None = None
    sample_rate: int = 0
    cut_points: list[float] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "audio_path": self.audio_path,
            "sample_rate": self.sample_rate,
            "cut_points": [round(float(point), 6) for point in self.cut_points],
            "segments": [segment.to_dict() for segment in self.segments],
            "settings": self.settings,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectData":
        return cls(
            name=str(data.get("name", "HumanSlice")),
            audio_path=data.get("audio_path"),
            sample_rate=int(data.get("sample_rate", 0)),
            cut_points=[float(point) for point in data.get("cut_points", [])],
            segments=[Segment.from_dict(item) for item in data.get("segments", [])],
            settings=dict(data.get("settings", {})),
            version=str(data.get("version", "0.1.0")),
        )
