"""Export a unit bank as an OpenUtau-ready Mandarin CVVC voicebank.

Aliases generated (all optional except CV; OpenUtau's ZH CVVC phonemizer falls back to CV):

- ``ba``      CV: best occurrence of each syllable
- ``- ba``    phrase-initial CV, from an occurrence that follows a pause
- ``a b``     VC transition, from two syllables that are contiguous in the material
- ``a R``     phrase ending, from an occurrence followed by a pause

oto.ini values follow the classic UTAU semantics (ms, relative to the wav file):
offset = usable start, consonant = unstretched length, cutoff < 0 = length from offset,
preutterance = point aligned to the note start, overlap = crossfade with the previous note.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from humanslice.common.audio import load_wav, save_wav
from humanslice.corpus.units import Bank, Unit
from humanslice.export.presamp_zh import CONSONANT_GROUP, PRESAMP_INI, VOWEL_GROUP, utau_spelling

PHONEMIZER = "OpenUtau.Plugin.Builtin.ChineseCVVCPhonemizer"
PAD_BEFORE = 0.25  # seconds of context kept before the alias offset
PAD_AFTER = 0.15
TONE_PREFERENCE = {1: 1.0, 4: 0.85, 2: 0.8, 3: 0.55, 5: 0.5}


@dataclass(slots=True)
class OtoEntry:
    wav: str
    alias: str
    offset: float  # all in ms
    consonant: float
    cutoff: float
    preutter: float
    overlap: float

    def line(self) -> str:
        return (
            f"{self.wav}={self.alias},{self.offset:.1f},{self.consonant:.1f},"
            f"{self.cutoff:.1f},{self.preutter:.1f},{self.overlap:.1f}"
        )


def unit_quality(unit: Unit, speaker_median: float | None) -> float:
    """Heuristic score for choosing the occurrence that best represents a syllable."""
    if unit.rejected:
        return -1.0
    duration_score = min(unit.final_duration, 0.35) / 0.35
    stability = math.exp(-(unit.f0_std or 2.0) / 1.0)
    nucleus = 0.0
    if unit.nucleus_start is not None and unit.nucleus_end is not None:
        nucleus = min(unit.nucleus_end - unit.nucleus_start, 0.25) / 0.25
    pitch = 0.5
    if unit.f0_midi is not None and speaker_median is not None:
        pitch = math.exp(-abs(unit.f0_midi - speaker_median) / 4.0)
    energy = 0.5 if unit.energy_db is None else float(np.clip((unit.energy_db + 45.0) / 30.0, 0.0, 1.0))
    return (
        0.25 * duration_score
        + 0.20 * stability
        + 0.15 * nucleus
        + 0.15 * pitch
        + 0.10 * energy
        + 0.10 * TONE_PREFERENCE.get(unit.tone, 0.5)
        + 0.05 * unit.voiced_ratio
    )


def speaker_median_pitch(units: list[Unit]) -> float | None:
    values = [unit.f0_midi for unit in units if unit.f0_midi is not None and not unit.rejected]
    return float(np.median(values)) if values else None


def _safe_name(alias: str) -> str:
    """ASCII wav name per alias: ``ba``, ``head_ba`` (- ba), ``vc_a_b`` (a b), ``end_a`` (a R)."""
    if alias.startswith("- "):
        alias = "head " + alias[2:]
    elif alias.endswith(" R"):
        alias = "end " + alias[:-2]
    elif " " in alias:
        alias = "vc " + alias
    return re.sub(r"[^0-9A-Za-z]+", "_", alias).strip("_") or "x"


class _Writer:
    def __init__(self, bank: Bank, out_dir: Path) -> None:
        self.bank = bank
        self.out_dir = out_dir
        self.entries: list[OtoEntry] = []
        self._audio_cache: dict[str, np.ndarray] = {}
        self.sample_rate = int(bank.meta.get("sample_rate", 44100))

    def _source(self, source_id: str) -> np.ndarray:
        if source_id not in self._audio_cache:
            if len(self._audio_cache) > 8:
                self._audio_cache.clear()
            self._audio_cache[source_id], _ = load_wav(self.bank.master_path(source_id), self.sample_rate)
        return self._audio_cache[source_id]

    def add(self, alias: str, source_id: str, region_start: float, region_end: float, *, consonant: float,
            preutter: float, overlap: float) -> None:
        """Cut [region_start, region_end] (+ padding) to a wav; times of the oto are absolute seconds."""
        audio = self._source(source_id)
        file_start = max(0.0, region_start - PAD_BEFORE)
        file_end = min(audio.size / self.sample_rate, region_end + PAD_AFTER)
        clip = audio[int(file_start * self.sample_rate):int(file_end * self.sample_rate)]
        fade = min(int(0.005 * self.sample_rate), clip.size // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip = clip.copy()
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]
        wav_name = f"{_safe_name(alias)}.wav"
        save_wav(self.out_dir / wav_name, clip, self.sample_rate)
        offset_ms = (region_start - file_start) * 1000.0
        length_ms = (region_end - region_start) * 1000.0
        self.entries.append(
            OtoEntry(
                wav=wav_name,
                alias=alias,
                offset=offset_ms,
                consonant=min(consonant * 1000.0, length_ms),
                cutoff=-length_ms,
                preutter=preutter * 1000.0,
                overlap=overlap * 1000.0,
            )
        )


def export_voicebank(bank: Bank, out_dir: str | Path, name: str | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    units = bank.usable_units()
    unit_map = bank.unit_map()
    median = speaker_median_pitch(units)
    writer = _Writer(bank, out)

    by_syllable: dict[str, list[Unit]] = {}
    for unit in units:
        by_syllable.setdefault(utau_spelling(unit.pinyin), []).append(unit)

    for syllable, candidates in sorted(by_syllable.items()):
        ranked = sorted(candidates, key=lambda u: unit_quality(u, median), reverse=True)
        _add_cv(writer, syllable, ranked[0])
        initial = [u for u in ranked if u.prev_id is None]
        if initial:
            _add_cv(writer, f"- {syllable}", initial[0])

    # VC transitions and endings, best example per alias.
    best_vc: dict[str, tuple[float, Unit, Unit]] = {}
    best_end: dict[str, tuple[float, Unit]] = {}
    for unit in units:
        vowel = VOWEL_GROUP.get(utau_spelling(unit.pinyin))
        if vowel is None:
            continue
        score = unit_quality(unit, median)
        nxt = unit_map.get(unit.next_id) if unit.next_id else None
        if nxt is not None and not nxt.rejected:
            consonant = CONSONANT_GROUP.get(utau_spelling(nxt.pinyin))
            if consonant:
                alias = f"{vowel} {consonant}"
                pair_score = score + unit_quality(nxt, median)
                if alias not in best_vc or pair_score > best_vc[alias][0]:
                    best_vc[alias] = (pair_score, unit, nxt)
        elif unit.next_id is None:
            alias = f"{vowel} R"
            if alias not in best_end or score > best_end[alias][0]:
                best_end[alias] = (score, unit)

    for alias, (_, left, right) in sorted(best_vc.items()):
        tail = min(0.12, 0.5 * left.final_duration)
        start = left.end - tail
        writer.add(
            alias,
            left.source_id,
            start,
            right.cv_boundary + 0.01,
            consonant=right.cv_boundary - start,
            preutter=right.start - start,
            overlap=tail * 0.5,
        )
    for alias, (_, unit) in sorted(best_end.items()):
        tail = min(0.15, 0.6 * unit.final_duration)
        start = unit.end - tail
        writer.add(alias, unit.source_id, start, unit.end + 0.08, consonant=tail * 0.5, preutter=0.03, overlap=0.03)

    _write_metadata(out, writer.entries, name or out.name, bank)
    return out


def _add_cv(writer: _Writer, alias: str, unit: Unit) -> None:
    preutter = max(unit.cv_boundary - unit.start, 0.0)
    lead = 0.0
    if preutter < 0.02:  # zero initial: start slightly before the vowel onset
        lead = 0.02
    start = unit.start - lead
    nucleus_start = unit.nucleus_start if unit.nucleus_start is not None else unit.cv_boundary
    nucleus_end = unit.nucleus_end if unit.nucleus_end is not None else unit.end
    end = max(nucleus_end, unit.cv_boundary + 0.6 * unit.final_duration)
    fixed = max(nucleus_start - start, preutter + lead + 0.02)
    overlap = (preutter * 0.5) if preutter >= 0.02 else 0.01
    writer.add(alias, unit.source_id, start, end, consonant=fixed, preutter=preutter + lead, overlap=overlap)


def _write_metadata(out: Path, entries: list[OtoEntry], name: str, bank: Bank) -> None:
    (out / "oto.ini").write_text("\n".join(entry.line() for entry in entries) + "\n", encoding="utf-8")
    (out / "presamp.ini").write_text(PRESAMP_INI, encoding="utf-8")
    (out / "character.txt").write_text(f"name={name}\nauthor=HumanSlice (auto-generated)\n", encoding="utf-8")
    (out / "character.yaml").write_text(
        f"name: {name}\n"
        "text_file_encoding: utf-8\n"
        f"default_phonemizer: {PHONEMIZER}\n"
        "singer_type: utau\n",
        encoding="utf-8",
    )
    sources = len(bank.meta.get("sources", {}))
    (out / "readme.txt").write_text(
        f"Auto-generated by HumanSlice from {sources} material file(s).\n"
        f"{len(entries)} aliases. Lyrics: toneless pinyin (v for ü), e.g. ni hao / lv / nue.\n",
        encoding="utf-8",
    )
