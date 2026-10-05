"""End-to-end rendering: score + unit bank -> jinriki vocal wav (+ selection report)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from humanslice.common.audio import save_wav
from humanslice.corpus.units import Bank
from humanslice.render.plan import plan
from humanslice.render.select import SelectOptions, Selection, UnitSelector
from humanslice.render.world import WorldOptions, render_world
from humanslice.song.score import Score


@dataclass(slots=True)
class RenderOptions:
    select: SelectOptions = field(default_factory=SelectOptions)
    world: WorldOptions = field(default_factory=WorldOptions)
    max_stretch: float = 2.5
    backend: str = "nsf"  # "nsf" (PC-NSF-HiFiGAN) or "world"


@dataclass(slots=True)
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    selection: Selection
    report: dict


def note_curve(score: Score) -> np.ndarray:
    """Per-frame MIDI of the active note (held through rests) on the score's F0 grid."""
    frames = max(score.f0_hz.size, int(score.duration / score.frame_period) + 1)
    curve = np.full(frames, np.nan)
    for note in score.notes:
        a = int(note.start / score.frame_period)
        b = int(note.end / score.frame_period)
        curve[a:b] = note.pitch
    valid = np.isfinite(curve)
    if not valid.any():
        return np.full(frames, 60.0)
    idx = np.arange(frames)
    return np.interp(idx, idx[valid], curve[valid])


def render(bank: Bank, score: Score, options: RenderOptions | None = None, progress: Callable[[str], None] = print) -> RenderResult:
    options = options or RenderOptions()
    selector = UnitSelector(bank, options.select)
    if not selector.units:
        raise ValueError("The unit bank has no usable units")
    selection = selector.select(score)
    progress(
        f"[render] transpose {selection.transpose:+d} st, speaker median "
        f"{selector.median:.1f}, {selection.summary()['contiguous_joins']} contiguous joins"
    )
    placements = plan(score, selection, max_stretch=options.max_stretch)
    if options.backend == "nsf":
        from humanslice.render.nsf import render_nsf as synthesize
    elif options.backend == "world":
        synthesize = render_world
    else:
        raise ValueError(f"Unknown backend {options.backend!r}")
    audio, sr = synthesize(
        bank,
        placements,
        score.f0_hz,
        score.frame_period,
        score.duration,
        selection.transpose,
        loudness_db=score.loudness_db,
        note_curve=note_curve(score),
        options=options.world,
        progress=progress,
    )
    return RenderResult(audio, sr, selection, build_report(score, selection))


def build_report(score: Score, selection: Selection) -> dict:
    rows = []
    for syllable, unit, tier in zip(score.syllables, selection.units, selection.tiers):
        rows.append(
            {
                "char": syllable.char,
                "pinyin": syllable.pinyin,
                "time": round(syllable.vowel_start, 3),
                "unit": unit.unit_id,
                "unit_char": unit.char,
                "unit_pinyin": unit.pinyin,
                "source": unit.source_id,
                "source_time": round(unit.start, 3),
                "tier": tier,
                "shift_st": None if unit.f0_midi is None else round(
                    np.mean([n.pitch for n in score.notes if n.syllable == syllable_index(score, syllable)] or [0])
                    + selection.transpose - unit.f0_midi, 2),
            }
        )
    return {"transpose": selection.transpose, "summary": selection.summary(), "syllables": rows}


def syllable_index(score: Score, syllable) -> int:
    return score.syllables.index(syllable)


def write_outputs(result: RenderResult, out_wav: str | Path) -> Path:
    out_wav = Path(out_wav)
    save_wav(out_wav, result.audio, result.sample_rate, subtype="PCM_24")
    out_wav.with_suffix(".report.json").write_text(json.dumps(result.report, ensure_ascii=False, indent=1), encoding="utf-8")
    return out_wav
