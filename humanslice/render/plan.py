"""Render plan: where each selected unit goes on the output timeline and how its time is warped.

For every sung syllable the chosen unit is laid out as:

    [consonant at natural length][vowel onset][nucleus stretched / looped][tail]
                                 ^ note onset (score vowel_start)

The consonant precedes the beat (先行発声), only the stable nucleus is stretched, and
when the needed stretch exceeds ``max_stretch`` the nucleus is traversed back and forth
(ping-pong) instead of being smeared. Neighbouring syllables overlap by a short
crossfade so a backend can blend them in the parameter domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanslice.corpus.units import Unit
from humanslice.render.select import Selection
from humanslice.song.score import Score

CROSSFADE = 0.03
MAX_CONSONANT = 0.18
RELEASE = 0.06  # tail kept after a phrase-final syllable


@dataclass(slots=True)
class Placement:
    unit: Unit
    syllable: int
    out_start: float  # region on the output timeline (seconds, before transposition-independent)
    out_end: float
    onset: float  # note onset = start of the vowel on the output timeline
    knots_out: np.ndarray  # piecewise-linear time map: output time -> source time (absolute, in source)
    knots_src: np.ndarray
    pingpong: tuple[float, float, float, float, int] | None = None  # (out_a, out_b, src_a, src_b, half_cycles)
    fade_in: float = CROSSFADE
    fade_out: float = CROSSFADE

    def source_time(self, out_times: np.ndarray) -> np.ndarray:
        src = np.interp(out_times, self.knots_out, self.knots_src)
        if self.pingpong is not None:
            out_a, out_b, src_a, src_b, half_cycles = self.pingpong
            inside = (out_times >= out_a) & (out_times < out_b)
            if inside.any():
                # Bounce across the nucleus an odd number of times so the walk ends at src_b,
                # where the (linear) tail mapping continues seamlessly.
                phase = (out_times[inside] - out_a) / max(out_b - out_a, 1e-6) * half_cycles
                tri = 1.0 - np.abs((phase % 2.0) - 1.0)
                src[inside] = src_a + tri * (src_b - src_a)
        return src

    def weight(self, out_times: np.ndarray) -> np.ndarray:
        w = np.ones_like(out_times)
        if self.fade_in > 0:
            w = np.minimum(w, np.clip((out_times - self.out_start) / self.fade_in, 0.0, 1.0))
        if self.fade_out > 0:
            w = np.minimum(w, np.clip((self.out_end - out_times) / self.fade_out, 0.0, 1.0))
        w[(out_times < self.out_start) | (out_times > self.out_end)] = 0.0
        return w


def plan(score: Score, selection: Selection, max_stretch: float = 2.5) -> list[Placement]:
    placements: list[Placement] = []
    syllables = score.syllables
    for index, (syllable, unit) in enumerate(zip(syllables, selection.units)):
        consonant = min(unit.consonant_duration, MAX_CONSONANT)
        onset = syllable.vowel_start
        out_start = onset - consonant
        nxt = syllables[index + 1] if index + 1 < len(syllables) else None
        next_unit = selection.units[index + 1] if nxt is not None else None
        joined_next = nxt is not None and nxt.start - syllable.end <= 0.15
        if joined_next:
            next_consonant = min(next_unit.consonant_duration, MAX_CONSONANT)
            out_end = max(nxt.vowel_start - next_consonant, onset + 0.03) + CROSSFADE
        else:
            out_end = syllable.end + RELEASE

        knots_out = [out_start, onset]
        knots_src = [unit.cv_boundary - consonant, unit.cv_boundary]
        pingpong = None

        vowel_out = out_end - onset
        final_len = unit.end - unit.cv_boundary
        ns = unit.nucleus_start if unit.nucleus_start is not None else unit.cv_boundary
        ne = unit.nucleus_end if unit.nucleus_end is not None else unit.end
        if ne - ns < 0.04:  # no usable nucleus: use the middle half of the final
            ns = unit.cv_boundary + 0.25 * final_len
            ne = unit.cv_boundary + 0.75 * final_len
        attack = ns - unit.cv_boundary
        tail = min(unit.end - ne, 0.2)  # offglide / coda after the held vowel
        if vowel_out <= attack + (ne - ns) + tail:
            # Short note: compress the final uniformly (keeps codas such as -n / -i),
            # but never faster than 2x; beyond that the end is truncated.
            used = min(attack + (ne - ns) + tail, 2.0 * vowel_out)
            knots_out += [out_end]
            knots_src += [unit.cv_boundary + used]
        else:
            nucleus_out = vowel_out - attack - tail
            ratio = nucleus_out / (ne - ns)
            a_out, b_out = onset + attack, out_end - tail
            knots_out += [a_out, b_out, out_end]
            knots_src += [ns, ne, ne + tail]
            if ratio > max_stretch:
                half_cycles = max(1, int(round(ratio / max_stretch)))
                half_cycles += 1 - half_cycles % 2  # odd
                pingpong = (a_out, b_out, ns, ne, half_cycles)

        placements.append(
            Placement(
                unit=unit,
                syllable=index,
                out_start=out_start,
                out_end=out_end,
                onset=onset,
                knots_out=_strictly_increasing(knots_out),
                knots_src=np.asarray(knots_src, dtype=float),
                pingpong=pingpong,
                fade_in=CROSSFADE if index > 0 and placements and placements[-1].out_end > out_start else 0.01,
                fade_out=CROSSFADE if joined_next else RELEASE,
            )
        )
    return placements


def _strictly_increasing(values: list[float], eps: float = 1e-4) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    for i in range(1, out.size):
        out[i] = max(out[i], out[i - 1] + eps)
    return out


def phrases(placements: list[Placement], gap: float = 0.08) -> list[list[Placement]]:
    """Group placements whose output regions touch or overlap; groups render independently."""
    groups: list[list[Placement]] = []
    for placement in sorted(placements, key=lambda p: p.out_start):
        if groups and placement.out_start <= max(p.out_end for p in groups[-1]) + gap:
            groups[-1].append(placement)
        else:
            groups.append([placement])
    return groups
