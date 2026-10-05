from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from humanslice.analysis.cutpoints import CutpointConfig
from humanslice.analysis.regions import RegionConfig
from humanslice.analysis.scoring import ScoreConfig


@dataclass(slots=True)
class PlaybackSettings:
    """Playback-related UI settings."""

    preview_crossfade_ms: int = 35
    boundary_window_ms: int = 180
    preview_volume: int = 85


@dataclass(slots=True)
class AppSettings:
    """Top-level application settings."""

    target_sample_rate: int | None = None
    cutpoint: CutpointConfig = field(default_factory=CutpointConfig)
    regions: RegionConfig = field(default_factory=RegionConfig)
    scoring: ScoreConfig = field(default_factory=ScoreConfig)
    playback: PlaybackSettings = field(default_factory=PlaybackSettings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_sample_rate": self.target_sample_rate,
            "cutpoint": asdict(self.cutpoint),
            "regions": asdict(self.regions),
            "scoring": asdict(self.scoring),
            "playback": asdict(self.playback),
        }


def load_settings(path: str | Path | None) -> AppSettings:
    """Load optional JSON settings, falling back to defaults."""
    if path is None:
        return AppSettings()

    file_path = Path(path)
    if not file_path.exists():
        return AppSettings()

    data = json.loads(file_path.read_text(encoding="utf-8"))
    return AppSettings(
        target_sample_rate=data.get("target_sample_rate"),
        cutpoint=CutpointConfig(**_filter_fields(CutpointConfig, data.get("cutpoint", {}))),
        regions=RegionConfig(**_filter_fields(RegionConfig, data.get("regions", {}))),
        scoring=ScoreConfig(**_filter_fields(ScoreConfig, data.get("scoring", {}))),
        playback=PlaybackSettings(**_filter_fields(PlaybackSettings, data.get("playback", {}))),
    )


def _filter_fields(model_type: type, values: dict[str, Any]) -> dict[str, Any]:
    allowed = set(model_type.__dataclass_fields__.keys())
    return {key: value for key, value in values.items() if key in allowed}
