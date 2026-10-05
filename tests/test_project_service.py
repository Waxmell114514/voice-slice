from __future__ import annotations

import json
import os
from pathlib import Path

from humanslice.models.project import ProjectData, Segment
from humanslice.services.project_service import load_project, save_project


def test_save_and_load_project_roundtrip_relative_audio_path(tmp_path: Path) -> None:
    audio_path = (tmp_path / "audio" / "source.wav").resolve()
    project_path = tmp_path / "projects" / "project.json"

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    project = ProjectData(
        audio_path=str(audio_path),
        sample_rate=22050,
        cut_points=[0.0, 0.4, 0.8],
        segments=[
            Segment("seg_001", 0.0, 0.4, alias="a"),
            Segment("seg_002", 0.4, 0.8, alias="i"),
        ],
        settings={"playback": {"preview_volume": 75}},
    )

    saved_path = save_project(project, project_path)
    payload = json.loads(saved_path.read_text(encoding="utf-8"))
    expected_relative = os.path.relpath(audio_path, start=project_path.parent)

    assert payload["audio_path"] == expected_relative
    assert payload["cut_points"] == [0.0, 0.4, 0.8]
    assert payload["segments"][0]["alias"] == "a"

    loaded = load_project(project_path)

    assert loaded.audio_path == str(audio_path)
    assert loaded.sample_rate == 22050
    assert loaded.cut_points == [0.0, 0.4, 0.8]
    assert [segment.alias for segment in loaded.segments] == ["a", "i"]
