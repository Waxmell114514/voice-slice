"""Phone-level forced alignment with Montreal Forced Aligner (run in its own conda env).

MFA is the only aligner with published *speech* phone-boundary accuracy (< 15 ms mean on
English/Japanese benchmarks). It runs on CPU in a separate environment (default
``<project>/.envs/mfa``) driven through ``conda run``. Utterances are aligned as toned
pinyin tokens (``ni3 hao3``) against a syllable lexicon derived from MFA's own
dictionary (see ``mfa_lexicon``), so each word interval is exactly one syllable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanslice.common.audio import save_wav
from humanslice.common.paths import data_root, scratch_dir
from humanslice.common.textgrid import TextGrid, read_textgrid
from humanslice.corpus.mfa_lexicon import build_lexicon, is_consonant, pronunciations

DEFAULT_ENV = data_root() / "envs" / "mfa"
MODELS_DIR = data_root() / "mfa_models"
RELEASES = "https://github.com/MontrealCorpusTools/mfa-models/releases/download/"
MODEL_FILES = {
    "zh": {
        "acoustic": ("mandarin_mfa.zip", "acoustic-mandarin_mfa-v3.0.0/mandarin_mfa.zip"),
        "dictionary": ("mandarin_china_mfa.dict", "dictionary-mandarin_china_mfa-v3.0.0/mandarin_china_mfa.dict"),
    },
}
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>", "<unk>"}


class MFAError(RuntimeError):
    pass


@dataclass(slots=True)
class AlignItem:
    utt_id: str
    audio_16k: np.ndarray
    tokens: list[str]  # toned pinyin, one per syllable, e.g. ["ni3", "hao3"]


@dataclass(slots=True)
class SyllableTiming:
    text: str
    start: float
    end: float
    cv_boundary: float  # end of the initial consonant (start for zero-initial)
    phones: list[str]


class MFARunner:
    def __init__(self, env_path: str | Path | None = None, language: str = "zh", jobs: int | None = None) -> None:
        self.env_path = Path(env_path or os.environ.get("HUMANSLICE_MFA_ENV", DEFAULT_ENV))
        self.language = language
        self.jobs = jobs or max(1, (os.cpu_count() or 4) // 2)
        if not (self.env_path / "conda-meta").exists():
            raise MFAError(
                f"MFA environment not found at {self.env_path}. Create it with:\n"
                f"  conda create -p {self.env_path} -c conda-forge montreal-forced-aligner python=3.12"
            )
        if language not in MODEL_FILES:
            raise MFAError(f"No MFA model configured for language {language!r}")

    # ----------------------------------------------------------------- CLI

    @staticmethod
    def _conda() -> str:
        for candidate in (os.environ.get("CONDA_EXE"), shutil.which("conda"), r"C:\ProgramData\miniconda3\Scripts\conda.exe"):
            if candidate and Path(candidate).exists():
                return candidate
        raise MFAError("conda executable not found (set CONDA_EXE)")

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        cmd = [self._conda(), "run", "-p", str(self.env_path), "--no-capture-output", "mfa", *args]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        if result.returncode != 0:
            raise MFAError(f"mfa {' '.join(args[:2])} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        return result

    def model_path(self, kind: str) -> Path:
        filename, release_path = MODEL_FILES[self.language][kind]
        path = MODELS_DIR / filename
        if not path.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".part")
            with urllib.request.urlopen(RELEASES + release_path, timeout=600) as response:
                tmp.write_bytes(response.read())
            tmp.replace(path)
        return path

    # ----------------------------------------------------------- aligning

    def align(self, items: list[AlignItem], sample_rate: int = 16000) -> dict[str, TextGrid]:
        """Align utterances; returns TextGrids (tiers ``words`` / ``phones``) by utt_id.

        Work happens in a space-free scratch directory (MFA quotes paths containing
        spaces when launching Kaldi binaries, which breaks on Windows).
        """
        acoustic = self.model_path("acoustic")
        lexicon = build_lexicon(self.model_path("dictionary"), MODELS_DIR / f"{self.language}_syllable_lexicon.json")
        root = Path(tempfile.mkdtemp(prefix="align_", dir=scratch_dir("mfa")))
        try:
            return self._align_in(root, items, acoustic, lexicon, sample_rate)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def _align_in(self, root: Path, items, acoustic: Path, lexicon, sample_rate: int) -> dict[str, TextGrid]:
        corpus_dir = root / "corpus" / "speaker"
        out_dir = root / "output"
        corpus_dir.mkdir(parents=True)

        needed: set[str] = set()
        for item in items:
            save_wav(corpus_dir / f"{item.utt_id}.wav", item.audio_16k, sample_rate)
            (corpus_dir / f"{item.utt_id}.lab").write_text(" ".join(item.tokens), encoding="utf-8")
            needed.update(item.tokens)
        dictionary = root / "syllables.dict"
        lines = [f"{token}\t{phones}" for token in sorted(needed) for phones in pronunciations(lexicon, token)]
        dictionary.write_text("\n".join(lines) + "\n", encoding="utf-8")

        self.run(
            "align",
            str(corpus_dir.parent),
            str(dictionary),
            str(acoustic),
            str(out_dir),
            "--clean",
            "--overwrite",
            "--single_speaker",
            "--no_tokenization",
            "--fine_tune",
            "--num_jobs",
            str(self.jobs),
            "--temporary_directory",
            str(root / "tmp"),
        )
        return {path.stem: read_textgrid(path) for path in out_dir.rglob("*.TextGrid")}


def syllable_timings(grid: TextGrid) -> list[SyllableTiming | None]:
    """One entry per aligned word (= syllable token); unaligned tokens give ``None``."""
    words = []
    for interval in grid.tier("words"):
        if interval.text.strip() in SILENCE_LABELS - {"<unk>"}:
            continue
        # MFA occasionally emits a spurious, very long repeat of the last word over trailing
        # non-speech; genuine repeats (天天 tian1 tian1) are short and are kept.
        if words and interval.text == words[-1].text and interval.end - interval.start > 1.5:
            continue
        words.append(interval)
    phones = [interval for interval in grid.tier("phones") if interval.text.strip() not in SILENCE_LABELS]
    result: list[SyllableTiming | None] = []
    for word in words:
        inside = [p for p in phones if p.start >= word.start - 1e-4 and p.end <= word.end + 1e-4]
        if not inside or word.text.strip() == "<unk>":
            result.append(None)
            continue
        boundary = word.start
        if is_consonant(inside[0].text.strip()) and len(inside) > 1:
            boundary = inside[0].end
        result.append(
            SyllableTiming(
                text=word.text.strip(),
                start=word.start,
                end=word.end,
                cv_boundary=boundary,
                phones=[p.text.strip() for p in inside],
            )
        )
    return result
