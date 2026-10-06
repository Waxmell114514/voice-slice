"""Data directory for heavy assets: MFA environment, model weights, caches and scratch.

Defaults to ``<project>/data`` for a source checkout (the Hugging Face cache lives there
too, see ``configure_caches``). Override with the ``HUMANSLICE_HOME`` environment
variable. MFA / Kaldi break on paths containing spaces, so keep this path space-free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    override = os.environ.get("HUMANSLICE_HOME")
    if override:
        root = Path(override)
    elif (PROJECT_ROOT / "pyproject.toml").exists():
        root = PROJECT_ROOT / "data"
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


def configure_caches() -> None:
    """Point the Hugging Face cache into the data directory unless the user set one.

    Must run before ``huggingface_hub`` is imported (it reads these at import time),
    hence it is called from ``humanslice/__init__.py``.
    """
    if not any(key in os.environ for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")):
        os.environ["HF_HOME"] = str(data_root() / "huggingface")
