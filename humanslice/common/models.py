"""Model weight download / cache helpers (Hugging Face Hub)."""

from __future__ import annotations

import gc
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def hf_file(repo_id: str, filename: str, repo_type: str = "model") -> str:
    """Download (or reuse the cached copy of) one file from the Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def hf_snapshot(repo_id: str, allow_patterns: list[str] | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, allow_patterns=allow_patterns))


def torch_device(prefer: str | None = None) -> str:
    prefer = prefer or os.environ.get("HUMANSLICE_DEVICE")
    if prefer:
        return prefer
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def release_gpu_memory() -> None:
    """Free cached CUDA memory between pipeline stages (8 GB cards load one model at a time)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
