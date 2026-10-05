"""Unit selection: choose one speech unit per sung syllable (Viterbi over target + join costs).

Target cost  pitch distance to the note, stretch factor, unit quality, tone fit, and a
             penalty when the syllable itself is missing and a similar one stands in.
Join cost    zero for units that are contiguous in the material — this is what keeps whole
             spoken words/phrases intact (the 人力 character) — otherwise a base cost plus
             source / pitch-shift / loudness mismatch terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from humanslice.corpus.units import Bank, Unit
from humanslice.export.utau import speaker_median_pitch, unit_quality
from humanslice.song.score import Score
from humanslice.text.zh import split_pinyin

# Fallback tiers when the exact syllable is missing from the material.
TIER_EXACT, TIER_SAME_FINAL, TIER_SAME_VOWEL, TIER_ANY = 0, 1, 2, 3
TIER_PENALTY = {TIER_EXACT: 0.0, TIER_SAME_FINAL: 2.5, TIER_SAME_VOWEL: 5.0, TIER_ANY: 15.0}


@dataclass(slots=True)
class SelectOptions:
    max_candidates: int = 40
    join_weight: float = 1.0  # higher = stronger preference for contiguous original phrases
    pitch_weight: float = 1.0
    stretch_weight: float = 1.0
    transpose: int | None = None  # semitones; None = automatic
    octave_only: bool = True  # keep the key (needed when mixing with the original instrumental)


@dataclass(slots=True)
class Target:
    index: int  # syllable index in the score
    pinyin: str
    pitch: float  # duration-weighted note pitch after transposition
    vowel_duration: float
    phrase_start: bool
    phrase_end: bool
    joined_to_previous: bool  # no rest between this and the previous syllable


@dataclass(slots=True)
class Selection:
    transpose: int
    units: list[Unit]  # one per score syllable
    tiers: list[int] = field(default_factory=list)
    cost: float = 0.0

    def summary(self) -> dict[str, float]:
        n = max(len(self.units), 1)
        contiguous = sum(
            1 for a, b in zip(self.units, self.units[1:]) if a.next_id is not None and a.next_id == b.unit_id
        )
        return {
            "syllables": len(self.units),
            "exact_ratio": sum(1 for tier in self.tiers if tier == TIER_EXACT) / n,
            "contiguous_joins": contiguous,
            "cost": self.cost,
        }


def choose_transpose(score: Score, units: list[Unit], octave_only: bool = True) -> int:
    """Shift (semitones) that brings the melody into the speaker's natural pitch range."""
    median = speaker_median_pitch(units)
    if median is None or not score.notes:
        return 0
    pitches = np.array([note.pitch for note in score.notes], dtype=float)
    weights = np.array([note.duration for note in score.notes])
    shifts = range(-36, 37, 12) if octave_only else range(-36, 37)
    best, best_cost = 0, math.inf
    for shift in shifts:
        distance = np.abs(pitches + shift - median)
        cost = float(np.sum(weights * (0.1 * distance + np.maximum(0.0, distance - 4.0) ** 2)))
        if cost < best_cost - 1e-9:
            best, best_cost = shift, cost
    return best


def build_targets(score: Score, transpose: int, rest_gap: float = 0.15) -> list[Target]:
    targets: list[Target] = []
    for index, syllable in enumerate(score.syllables):
        notes = score.notes_of(index)
        if notes:
            weights = np.array([max(note.duration, 1e-3) for note in notes])
            pitch = float(np.average([note.pitch for note in notes], weights=weights)) + transpose
        else:
            pitch = (targets[-1].pitch if targets else 60.0)
        previous_end = score.syllables[index - 1].end if index > 0 else -1.0
        next_start = score.syllables[index + 1].start if index + 1 < len(score.syllables) else math.inf
        joined = index > 0 and syllable.start - previous_end <= rest_gap
        targets.append(
            Target(
                index=index,
                pinyin=syllable.pinyin,
                pitch=pitch,
                vowel_duration=max(syllable.end - syllable.vowel_start, 0.02),
                phrase_start=not joined,
                phrase_end=next_start - syllable.end > rest_gap,
                joined_to_previous=joined,
            )
        )
    return targets


class UnitSelector:
    def __init__(self, bank: Bank, options: SelectOptions | None = None) -> None:
        self.options = options or SelectOptions()
        self.units = bank.usable_units()
        self.units = [unit for unit in self.units if unit.f0_midi is not None]
        self.median = speaker_median_pitch(self.units)
        self._quality = {unit.unit_id: unit_quality(unit, self.median) for unit in self.units}
        self._by_pinyin: dict[str, list[Unit]] = {}
        self._by_final: dict[str, list[Unit]] = {}
        self._by_vowel: dict[str, list[Unit]] = {}
        for unit in self.units:
            self._by_pinyin.setdefault(unit.pinyin, []).append(unit)
            self._by_final.setdefault(unit.final, []).append(unit)
            self._by_vowel.setdefault(_vowel_nucleus(unit.final), []).append(unit)

    # ------------------------------------------------------------ candidates

    def candidates(self, target: Target) -> list[tuple[Unit, int]]:
        if target.pinyin in self._by_pinyin:
            return [(unit, TIER_EXACT) for unit in self._by_pinyin[target.pinyin]]
        initial, final = split_pinyin(target.pinyin)
        if final in self._by_final:
            return [(unit, TIER_SAME_FINAL) for unit in self._by_final[final]]
        vowel = _vowel_nucleus(final)
        if vowel in self._by_vowel:
            return [(unit, TIER_SAME_VOWEL) for unit in self._by_vowel[vowel]]
        return [(unit, TIER_ANY) for unit in self.units]

    def target_cost(self, unit: Unit, tier: int, target: Target) -> float:
        opt = self.options
        distance = abs((unit.f0_midi or self.median or 60.0) - target.pitch)
        pitch_cost = 0.15 * distance + 0.15 * max(0.0, distance - 3.0) ** 2
        available = max(unit.final_duration, 0.03)
        stretch = math.log(max(target.vowel_duration / available, 1e-3))
        stretch_cost = 0.6 * max(0.0, stretch) ** 2 + 0.1 * max(0.0, -stretch)
        quality_cost = 1.5 * (1.0 - max(self._quality.get(unit.unit_id, 0.0), 0.0))
        tone_cost = 0.0
        if target.vowel_duration > 0.5:
            tone_cost = {3: 0.3, 5: 0.4}.get(unit.tone, 0.0)
        context = 0.0
        if target.phrase_start and unit.prev_id is None:
            context -= 0.15
        if target.phrase_end and unit.next_id is None:
            context -= 0.15
        return (
            opt.pitch_weight * pitch_cost
            + opt.stretch_weight * stretch_cost
            + quality_cost
            + tone_cost
            + context
            + TIER_PENALTY[tier]
        )

    def join_cost(self, left: Unit, right: Unit, left_target: Target, right_target: Target) -> float:
        if not right_target.joined_to_previous:
            return 0.0
        if left.next_id is not None and left.next_id == right.unit_id:
            return 0.0
        cost = 0.6
        if left.source_id != right.source_id:
            cost += 0.15
        shift_left = left_target.pitch - (left.f0_midi or left_target.pitch)
        shift_right = right_target.pitch - (right.f0_midi or right_target.pitch)
        cost += 0.03 * abs(shift_left - shift_right)
        if left.energy_db is not None and right.energy_db is not None:
            cost += 0.02 * abs(left.energy_db - right.energy_db)
        return self.options.join_weight * cost

    # ---------------------------------------------------------------- search

    def select(self, score: Score) -> Selection:
        transpose = self.options.transpose
        if transpose is None:
            transpose = choose_transpose(score, self.units, self.options.octave_only)
        targets = build_targets(score, transpose)
        if not targets:
            return Selection(transpose, [], [], 0.0)

        lattice: list[list[tuple[Unit, int, float]]] = []
        for target in targets:
            scored = [(unit, tier, self.target_cost(unit, tier, target)) for unit, tier in self.candidates(target)]
            scored.sort(key=lambda item: item[2])
            # Always keep contiguous continuations of the previous column's candidates.
            keep = scored[: self.options.max_candidates]
            if lattice:
                wanted = {u.next_id for u, _, _ in lattice[-1] if u.next_id}
                keep_ids = {u.unit_id for u, _, _ in keep}
                keep += [item for item in scored if item[0].unit_id in wanted and item[0].unit_id not in keep_ids]
            lattice.append(keep)

        best = [cost for _, _, cost in lattice[0]]
        back: list[list[int]] = [[-1] * len(lattice[0])]
        for column in range(1, len(lattice)):
            current, previous = lattice[column], lattice[column - 1]
            scores, pointers = [], []
            for unit, _, target_cost in current:
                options = [
                    best[j] + self.join_cost(prev_unit, unit, targets[column - 1], targets[column])
                    for j, (prev_unit, _, _) in enumerate(previous)
                ]
                j = int(np.argmin(options))
                scores.append(options[j] + target_cost)
                pointers.append(j)
            best, back = scores, back + [pointers]

        index = int(np.argmin(best))
        total = float(best[index])
        chosen: list[tuple[Unit, int]] = []
        for column in range(len(lattice) - 1, -1, -1):
            unit, tier, _ = lattice[column][index]
            chosen.append((unit, tier))
            index = back[column][index]
        chosen.reverse()
        return Selection(transpose, [unit for unit, _ in chosen], [tier for _, tier in chosen], total)


def _vowel_nucleus(final: str) -> str:
    """Main vowel of a phonetic final, ignoring medials and codas (``iang -> a``, ``uei -> e``)."""
    core = final.lstrip("iuv") or final
    for vowel in ("a", "o", "e", "i", "u", "v"):
        if vowel in core:
            return vowel
    return core[:1]
