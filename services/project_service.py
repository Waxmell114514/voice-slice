from __future__ import annotations

import json
import os
from pathlib import Path

from models.project import ProjectData


def save_project(project: ProjectData, path: str | Path) -> Path:
    """Save project state to JSON, storing audio path relative when possible."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = project.to_dict()
    if project.audio_path:
        try:
            payload["audio_path"] = os.path.relpath(project.audio_path, start=file_path.parent)
        except ValueError:
            payload["audio_path"] = project.audio_path
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_path


def load_project(path: str | Path) -> ProjectData:
    """Load a saved project and resolve its audio path."""
    file_path = Path(path)
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    project = ProjectData.from_dict(payload)
    if project.audio_path:
        audio_path = Path(project.audio_path)
        if not audio_path.is_absolute():
            project.audio_path = str((file_path.parent / audio_path).resolve())
    return project
