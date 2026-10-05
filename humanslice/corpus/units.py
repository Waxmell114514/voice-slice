"""Unit database: every aligned syllable occurrence in the speech material.

A *bank* directory holds the decoded material and everything derived from it::

    <bank>/
      bank.json            metadata (language, sources, sample rate, pipeline settings)
      audio/<source>.wav   44.1 kHz mono master audio per material file
      utterances.json      speech chunks with transcripts
      units.json           aligned syllable units + features
      work/                intermediate files (16 kHz copies, aligner input / output)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

MASTER_SAMPLE_RATE = 44100


@dataclass(slots=True)
class Utterance:
    """A contiguous speech chunk of one source file, with its transcript."""

    utt_id: str
    source_id: str
    start: float  # seconds in the source master audio
    end: float
    text: str = ""
    text_origin: str = ""  # "asr", "label", ...

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class Unit:
    """One spoken syllable, located in a source's master audio."""

    unit_id: str
    source_id: str
    utt_id: str
    start: float  # syllable start (seconds, source master timeline)
    end: float  # syllable end
    cv_boundary: float  # initial / final boundary; equals start for zero-initial syllables
    char: str
    pinyin: str  # toneless, "v" for ü
    tone: int
    initial: str
    final: str
    prev_id: str | None = None  # contiguous neighbour in the same utterance (no pause between)
    next_id: str | None = None
    # Acoustic features (filled by corpus.features)
    f0_midi: float | None = None  # median pitch of the voiced final, MIDI note number (float)
    f0_std: float | None = None  # pitch spread over the final, semitones
    voiced_ratio: float = 0.0  # fraction of voiced frames in the final
    nucleus_start: float | None = None  # most stable part of the final
    nucleus_end: float | None = None
    energy_db: float | None = None  # mean RMS of the final, dBFS
    # Alignment bookkeeping
    align_source: str = ""  # "mfa", "qwen", "label"
    align_error: float | None = None  # disagreement between aligners, seconds (if cross-checked)
    rejected: str = ""  # non-empty reason when QC rejects the unit

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def consonant_duration(self) -> float:
        return self.cv_boundary - self.start

    @property
    def final_duration(self) -> float:
        return self.end - self.cv_boundary

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, 5)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Unit":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass(slots=True)
class Bank:
    """In-memory view of a bank directory."""

    root: Path
    meta: dict[str, Any] = field(default_factory=dict)
    utterances: list[Utterance] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)

    # ------------------------------------------------------------ paths

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    def master_path(self, source_id: str) -> Path:
        return self.audio_dir / f"{source_id}.wav"

    # ------------------------------------------------------------- I/O

    @classmethod
    def open(cls, root: str | Path) -> "Bank":
        bank = cls(root=Path(root))
        meta_path = bank.root / "bank.json"
        if meta_path.exists():
            bank.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        utt_path = bank.root / "utterances.json"
        if utt_path.exists():
            bank.utterances = [Utterance(**item) for item in json.loads(utt_path.read_text(encoding="utf-8"))]
        units_path = bank.root / "units.json"
        if units_path.exists():
            bank.units = [Unit.from_dict(item) for item in json.loads(units_path.read_text(encoding="utf-8"))]
        return bank

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        _write_json(self.root / "bank.json", self.meta)
        _write_json(self.root / "utterances.json", [asdict(item) for item in self.utterances])
        _write_json(self.root / "units.json", [unit.to_dict() for unit in self.units])

    # ---------------------------------------------------------- queries

    def usable_units(self) -> list[Unit]:
        return [unit for unit in self.units if not unit.rejected]

    def units_by_pinyin(self) -> dict[str, list[Unit]]:
        index: dict[str, list[Unit]] = {}
        for unit in self.usable_units():
            index.setdefault(unit.pinyin, []).append(unit)
        return index

    def unit_map(self) -> dict[str, Unit]:
        return {unit.unit_id: unit for unit in self.units}

    def utterance_map(self) -> dict[str, Utterance]:
        return {utt.utt_id: utt for utt in self.utterances}


def link_contiguous(units: Iterable[Unit], max_gap: float = 0.03) -> None:
    """Set prev_id / next_id for units that follow each other without a pause."""
    ordered = sorted(units, key=lambda unit: (unit.source_id, unit.utt_id, unit.start))
    for left, right in zip(ordered, ordered[1:]):
        if left.utt_id == right.utt_id and right.start - left.end <= max_gap:
            left.next_id = right.unit_id
            right.prev_id = left.unit_id


def _write_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
