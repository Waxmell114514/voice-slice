"""Score -> OpenUtau project (USTX, YAML) for manual touch-up.

Notes start at the vowel onset (the voicebank's preutterance places the consonant
before it), melisma notes use the ``+`` extender lyric, note pitch bends are flat and
vibrato is off: the singer's real pitch (incl. vibrato) is written as a ``pitd`` curve
(cents relative to the note, every 5 ticks), as OpenUtau's own RMVPE import does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from humanslice.export.presamp_zh import utau_spelling
from humanslice.export.utau import PHONEMIZER
from humanslice.pitch.f0 import hz_to_midi, interpolate_unvoiced
from humanslice.song.score import Score

RESOLUTION = 480
CURVE_STEP = 5  # ticks


def seconds_to_ticks(seconds: float, tempo: float) -> int:
    return int(round(seconds * tempo / 60.0 * RESOLUTION))


def score_notes_in_ticks(score: Score, transpose: int = 0) -> list[dict]:
    """Notes as dicts with tick positions; overlapping / zero-length notes are fixed up."""
    notes = []
    previous_end = 0
    for note in sorted(score.notes, key=lambda n: n.start):
        start = max(seconds_to_ticks(note.start, score.tempo), previous_end)
        end = max(seconds_to_ticks(note.end, score.tempo), start + 15)
        syllable = score.syllables[note.syllable]
        lyric = "+" if note.slur else utau_spelling(syllable.pinyin)
        notes.append({"position": start, "duration": end - start, "tone": note.pitch + transpose, "lyric": lyric})
        previous_end = end
    return notes


def pitd_curve(score: Score, notes: list[dict], transpose: int = 0, tolerance: float = 3.0) -> tuple[list[int], list[int]]:
    """PITD points (ticks, cents) following the target F0 inside notes; 0 in rests."""
    if not notes or not np.any(score.f0_hz > 0):
        return [], []
    filled_midi = hz_to_midi(interpolate_unvoiced(score.f0_hz))
    xs: list[int] = []
    ys: list[int] = []
    for note in notes:
        start, end = note["position"], note["position"] + note["duration"]
        ticks = np.arange(start - start % CURVE_STEP, end, CURVE_STEP)
        seconds = ticks * 60.0 / (score.tempo * RESOLUTION)
        frames = np.clip(np.round(seconds / score.frame_period).astype(int), 0, filled_midi.size - 1)
        cents = np.clip(np.round((filled_midi[frames] + transpose - note["tone"]) * 100.0), -1200, 1200)
        if xs and ticks.size and ticks[0] <= xs[-1]:
            keep = ticks > xs[-1]
            ticks, cents = ticks[keep], cents[keep]
        xs.extend(int(t) for t in ticks)
        ys.extend(int(c) for c in cents)
    points = _simplify(np.array(xs, dtype=float), np.array(ys, dtype=float), tolerance)
    return [int(xs[i]) for i in points], [int(ys[i]) for i in points]


def _simplify(x: np.ndarray, y: np.ndarray, tolerance: float) -> list[int]:
    """Ramer-Douglas-Peucker on a polyline; returns kept indices (iterative)."""
    if x.size <= 2:
        return list(range(x.size))
    keep = np.zeros(x.size, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, x.size - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg_x, seg_y = x[a + 1:b], y[a + 1:b]
        line = y[a] + (y[b] - y[a]) * (seg_x - x[a]) / max(x[b] - x[a], 1e-9)
        errors = np.abs(seg_y - line)
        index = int(np.argmax(errors))
        if errors[index] > tolerance:
            mid = a + 1 + index
            keep[mid] = True
            stack.extend([(a, mid), (mid, b)])
    return list(np.flatnonzero(keep))


def build_ustx(score: Score, singer: str, transpose: int = 0, name: str = "HumanSlice") -> str:
    notes = score_notes_in_ticks(score, transpose)
    xs, ys = pitd_curve(score, notes, transpose)
    part_end = max((n["position"] + n["duration"] for n in notes), default=RESOLUTION * 4)
    tempo = float(score.tempo)
    lines = [
        f"name: {_q(name)}",
        "comment: ''",
        "output_dir: Vocal",
        "cache_dir: UCache",
        "ustx_version: '0.6'",
        f"resolution: {RESOLUTION}",
        f"bpm: {tempo}",
        "beat_per_bar: 4",
        "beat_unit: 4",
        "time_signatures:",
        "- bar_position: 0",
        "  beat_per_bar: 4",
        "  beat_unit: 4",
        "tempos:",
        "- position: 0",
        f"  bpm: {tempo}",
        "tracks:",
        f"- singer: {_q(singer)}",
        f"  phonemizer: {PHONEMIZER}",
        "  renderer_settings:",
        "    renderer: WORLDLINE-R",
        "  track_name: Vocal",
        "  mute: false",
        "  solo: false",
        "  volume: 0",
        "  pan: 0",
        "voice_parts:",
        f"- name: {_q(name)}",
        "  comment: ''",
        "  track_no: 0",
        "  position: 0",
        f"  duration: {part_end}",
        "  notes:",
    ]
    for note in notes:
        lines += [
            f"  - position: {note['position']}",
            f"    duration: {note['duration']}",
            f"    tone: {note['tone']}",
            f"    lyric: {_q(note['lyric'])}",
            "    pitch:",
            "      data:",
            "      - {x: -25, y: 0, shape: io}",
            "      - {x: 25, y: 0, shape: io}",
            "      snap_first: false",
            "    vibrato: {length: 0, period: 175, depth: 25, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0}",
            "    note_expressions: []",
            "    phoneme_expressions: []",
            "    phoneme_overrides: []",
        ]
    lines.append("  curves:")
    if xs:
        lines += [
            "  - abbr: pitd",
            f"    xs: [{', '.join(map(str, xs))}]",
            f"    ys: [{', '.join(map(str, ys))}]",
        ]
    else:
        lines[-1] = "  curves: []"
    lines.append("wave_parts: []")
    return "\n".join(lines) + "\n"


def write_ustx(score: Score, path: str | Path, singer: str, transpose: int = 0) -> Path:
    path = Path(path)
    path.write_text(build_ustx(score, singer, transpose, name=path.stem), encoding="utf-8")
    return path


def _q(text: str) -> str:
    """Single-quoted YAML scalar (avoids yes/no/on/off and numeric coercion)."""
    return "'" + str(text).replace("'", "''") + "'"
