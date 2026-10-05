"""Vocal / accompaniment separation with python-audio-separator (UVR models).

Used for (a) material extracted from videos with BGM and (b) full song mixes, where the
vocal stem drives the analysis and the instrumental stem is mixed back under the result.
Results are cached per input file (content hash) in the per-user data directory.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from humanslice.common.paths import data_root, scratch_dir

DEFAULT_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def _digest(path: Path) -> str:
    sha = hashlib.sha1()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()[:16]


def separate(path: str | Path, model: str = DEFAULT_MODEL) -> tuple[Path, Path]:
    """Return (vocals.wav, instrumental.wav) for an audio / video file."""
    path = Path(path)
    cache = scratch_dir("separated") / f"{_digest(path)}_{Path(model).stem}"
    vocals, instrumental = cache / "vocals.wav", cache / "instrumental.wav"
    if vocals.exists() and instrumental.exists():
        return vocals, instrumental

    from audio_separator.separator import Separator

    work = cache / "work"
    work.mkdir(parents=True, exist_ok=True)
    source = path
    if path.suffix.lower() not in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        # Video container: extract audio first.
        from humanslice.common.audio import decode, save_wav

        source = work / "input.wav"
        save_wav(source, decode(path, 44100), 44100)
    models_dir = data_root() / "separator_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    separator = Separator(model_file_dir=str(models_dir), output_dir=str(work), log_level=30)
    separator.load_model(model)
    outputs = [work / Path(name).name for name in separator.separate(str(source))]
    for output in outputs:
        lowered = output.name.lower()
        if "(vocals)" in lowered:
            shutil.move(output, vocals)
        elif "(instrumental)" in lowered or "(other)" in lowered:
            shutil.move(output, instrumental)
    shutil.rmtree(work, ignore_errors=True)
    if not vocals.exists():
        raise RuntimeError(f"Separation produced no vocal stem for {path}")
    return vocals, instrumental


def mix(vocal_path: str | Path, instrumental_path: str | Path, out_path: str | Path, vocal_gain_db: float = 0.0) -> Path:
    """Sum a rendered vocal with the instrumental (same timeline), peak-limited to -1 dBFS."""
    import numpy as np

    from humanslice.common.audio import load_wav, save_wav

    vocal, sr = load_wav(vocal_path)
    inst, _ = load_wav(instrumental_path, sr)
    n = max(vocal.size, inst.size)
    out = np.zeros(n, dtype=np.float64)
    out[: inst.size] += inst
    out[: vocal.size] += vocal * 10 ** (vocal_gain_db / 20)
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.89:
        out *= 0.89 / peak
    save_wav(out_path, out.astype(np.float32), sr, subtype="PCM_24")
    return Path(out_path)
