"""Score extracted from a target vocal: syllables, notes, pitch curve, loudness."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from humanslice.pitch.f0 import FRAME_PERIOD, F0Curve


@dataclass(slots=True)
class Note:
    start: float  # seconds; for the first note of a syllable this is the vowel onset (the beat)
    end: float
    pitch: int  # MIDI note number
    syllable: int  # index into Score.syllables
    slur: bool = False  # continuation of the previous note's syllable (melisma)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class SungSyllable:
    char: str
    pinyin: str
    tone: int
    initial: str
    final: str
    start: float  # consonant onset
    vowel_start: float  # consonant / vowel boundary (note onset)
    end: float


@dataclass(slots=True)
class Score:
    syllables: list[SungSyllable] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    f0_hz: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    loudness_db: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    frame_period: float = FRAME_PERIOD
    duration: float = 0.0
    tempo: float = 120.0
    source: str = ""

    @property
    def f0(self) -> F0Curve:
        return F0Curve(self.f0_hz, self.frame_period)

    def notes_of(self, syllable_index: int) -> list[Note]:
        return [note for note in self.notes if note.syllable == syllable_index]

    def save(self, path: str | Path) -> None:
        payload = {
            "source": self.source,
            "duration": self.duration,
            "tempo": self.tempo,
            "frame_period": self.frame_period,
            "syllables": [asdict(s) for s in self.syllables],
            "notes": [asdict(n) for n in self.notes],
            "f0_hz": np.round(self.f0_hz, 2).tolist(),
            "loudness_db": np.round(self.loudness_db, 2).tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Score":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            syllables=[SungSyllable(**s) for s in data["syllables"]],
            notes=[Note(**n) for n in data["notes"]],
            f0_hz=np.asarray(data["f0_hz"], dtype=np.float32),
            loudness_db=np.asarray(data["loudness_db"], dtype=np.float32),
            frame_period=data.get("frame_period", FRAME_PERIOD),
            duration=data["duration"],
            tempo=data.get("tempo", 120.0),
            source=data.get("source", ""),
        )
