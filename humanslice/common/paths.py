"""Per-user data directory for heavy, path-sensitive assets (MFA env, models, scratch).

Lives outside the project because several tools (MFA / Kaldi) break on paths with
spaces. Override with the ``HUMANSLICE_HOME`` environment variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def data_root() -> Path:
    override = os.environ.get("HUMANSLICE_HOME")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "humanslice"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "humanslice"
    root.mkdir(parents=True, exist_ok=True)
    return root


def scratch_dir(name: str) -> Path:
    path = data_root() / "scratch" / name
    path.mkdir(parents=True, exist_ok=True)
    return path
