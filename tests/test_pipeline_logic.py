"""Model-free tests for the jinriki pipeline: TextGrid I/O, notes, selection, planning, exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from humanslice.common.textgrid import Interval, TextGrid, read_textgrid, write_textgrid
from humanslice.corpus.mfa import syllable_timings
from humanslice.corpus.units import Bank, Unit, link_contiguous
from humanslice.export.ustx import build_ustx
from humanslice.pitch.f0 import fix_octave_errors, midi_to_hz
from humanslice.render.plan import plan
from humanslice.render.select import SelectOptions, Selection, UnitSelector, choose_transpose
from humanslice.song.notes import segment_syllable, tuning_offset
from humanslice.song.score import Note, Score, SungSyllable


def _unit(uid: str, pinyin: str, initial: str, final: str, start: float, f0: float, utt: str = "u0") -> Unit:
    return Unit(
        unit_id=uid, source_id="s0", utt_id=utt, start=start, end=start + 0.3,
        cv_boundary=start + (0.08 if initial else 0.0), char="字", pinyin=pinyin, tone=1,
        initial=initial, final=final, f0_midi=f0, f0_std=0.3, voiced_ratio=0.9,
        nucleus_start=start + 0.12, nucleus_end=start + 0.25, energy_db=-20.0,
    )


def _score(pinyins: list[str], pitches: list[int], gap: float = 0.0) -> Score:
    syllables, notes, t = [], [], 0.5
    for index, (pinyin, pitch) in enumerate(zip(pinyins, pitches)):
        syllables.append(SungSyllable("字", pinyin, 1, "", pinyin, t, t + 0.05, t + 0.5))
        notes.append(Note(t + 0.05, t + 0.5, pitch, index))
        t += 0.5 + gap
    f0 = np.zeros(int((t + 1) / 0.01), dtype=np.float32)
    for note in notes:
        f0[int(note.start / 0.01):int(note.end / 0.01)] = midi_to_hz(note.pitch)
    return Score(syllables, notes, f0, np.full(f0.size, -20.0, dtype=np.float32), 0.01, t + 1, 120.0, "x.wav")


def test_textgrid_roundtrip_and_syllable_timings(tmp_path: Path) -> None:
    grid = TextGrid(0.0, 1.0, {
        "words": [Interval(0.0, 0.1, ""), Interval(0.1, 0.4, "ni3"), Interval(0.4, 0.8, "a1"), Interval(0.8, 1.0, "")],
        "phones": [Interval(0.0, 0.1, ""), Interval(0.1, 0.18, "ɲ"), Interval(0.18, 0.4, "i˨˩˦"),
                   Interval(0.4, 0.45, "ʔ"), Interval(0.45, 0.8, "a˥"), Interval(0.8, 1.0, "")],
    })
    write_textgrid(grid, tmp_path / "x.TextGrid")
    loaded = read_textgrid(tmp_path / "x.TextGrid")
    assert [i.text for i in loaded.tier("words")] == ["", "ni3", "a1", ""]
    timings = syllable_timings(loaded)
    assert [t.text for t in timings] == ["ni3", "a1"]
    assert abs(timings[0].cv_boundary - 0.18) < 1e-9
    assert abs(timings[1].cv_boundary - 0.45) < 1e-9  # glottal onset counts as the consonant part


def test_octave_fix_keeps_real_leaps() -> None:
    hz = np.full(200, 220.0, dtype=np.float32)
    hz[50:55] = 440.0  # short octave error
    hz[120:] = 440.0  # genuine leap
    fixed = fix_octave_errors(hz)
    assert np.allclose(fixed[50:55], 220.0)
    assert np.allclose(fixed[130:], 440.0)


def test_segment_syllable_splits_melisma() -> None:
    hz = np.zeros(200, dtype=np.float32)
    hz[0:60] = midi_to_hz(60)
    hz[60:120] = midi_to_hz(64)
    notes = segment_syllable(hz, 0.0, 1.2)
    assert [pitch for _, _, pitch in notes] == [60, 64]
    assert abs(notes[1][0] - 0.6) < 0.06
    assert abs(tuning_offset(hz)) < 0.05


def test_selection_prefers_contiguous_units_and_transposes() -> None:
    units = [_unit("a1", "ni", "n", "i", 0.0, 48.0), _unit("a2", "hao", "h", "ao", 0.3, 48.0),
             _unit("b1", "ni", "n", "i", 5.0, 48.5, utt="u1"), _unit("b2", "hao", "h", "ao", 7.0, 48.5, utt="u2")]
    link_contiguous(units)
    bank = Bank(root=Path("."), units=units)
    score = _score(["ni", "hao"], [60, 60])
    assert choose_transpose(score, units) == -12
    selection = UnitSelector(bank, SelectOptions()).select(score)
    assert [u.unit_id for u in selection.units] == ["a1", "a2"]
    assert selection.transpose == -12


def test_plan_places_consonant_before_onset_and_loops_long_vowels() -> None:
    unit = _unit("a1", "ni", "n", "i", 0.0, 48.0)
    score = _score(["ni"], [48])
    score.syllables[0].end = 2.5
    score.notes[0].end = 2.5
    placement = plan(score, Selection(0, [unit], [0]))[0]
    assert abs(placement.onset - placement.out_start - 0.08) < 1e-6
    assert placement.pingpong is not None
    out = np.linspace(placement.out_start, placement.out_end, 400)
    src = placement.source_time(out)
    assert src.min() >= unit.start - 1e-6 and src.max() <= unit.end + 0.05


def test_ustx_contains_notes_extenders_and_pitd() -> None:
    score = _score(["ni", "hao"], [60, 62])
    score.notes.append(Note(score.notes[-1].end, score.notes[-1].end + 0.3, 64, 1, slur=True))
    text = build_ustx(score, singer="Bank", transpose=-12)
    assert "ustx_version: '0.6'" in text
    assert "lyric: 'ni'" in text and "lyric: '+'" in text
    assert "tone: 48" in text
    assert "abbr: pitd" in text


def test_voicebank_wav_names_do_not_collide() -> None:
    from humanslice.export.utau import _safe_name

    names = {_safe_name(alias) for alias in ("ba", "- ba", "a b", "a R", "ai b", "a ba")}
    assert len(names) == 6


def test_repair_spans_keeps_every_syllable() -> None:
    from humanslice.song.analyze import repair_spans

    spans = repair_spans([(0.0, 0.5), (0.5, 0.5), (0.5, 0.5), (0.6, 1.0)], 0.0, 1.2)
    assert all(b - a > 0.03 for a, b in spans)
    assert all(spans[i][1] <= spans[i + 1][0] + 1e-9 for i in range(len(spans) - 1))


def test_tuning_offset_detects_detuned_singer() -> None:
    hz = np.repeat(midi_to_hz(np.array([60.3, 62.3, 64.3, 65.3])), 80).astype(np.float32)
    assert abs(tuning_offset(hz) - 0.3) < 0.05


def test_nucleus_prefers_main_vowel_position() -> None:
    from humanslice.corpus.build import describe_unit

    frames = 40
    f0 = np.full(frames, 150.0, dtype=np.float32)
    energy = np.full(frames, -20.0, dtype=np.float32)
    flux = np.full(frames, 0.5, dtype=np.float32)
    flux[4:12] = 0.05  # steady early part (the "a" of "ao")
    flux[30:38] = 0.04  # slightly steadier off-glide at the end
    unit = _unit("x", "hao", "h", "ao", 0.0, 50.0)
    unit.cv_boundary, unit.end = 0.0, frames * 0.01
    describe_unit(unit, f0, energy, offset=0.0, flux=flux)
    assert unit.nucleus_start < 0.15


def test_syllable_timings_drops_spurious_long_repeat() -> None:
    grid = TextGrid(0.0, 14.0, {
        "words": [Interval(0.0, 0.3, "ni3"), Interval(0.3, 0.6, "tian1"), Interval(0.6, 0.9, "tian1"),
                  Interval(0.9, 13.0, "tian1")],
        "phones": [Interval(0.0, 0.1, "n"), Interval(0.1, 0.3, "i˨˩˦"), Interval(0.3, 0.4, "tʰ"),
                   Interval(0.4, 0.6, "e˥"), Interval(0.6, 0.7, "tʰ"), Interval(0.7, 0.9, "e˥"),
                   Interval(0.9, 13.0, "e˥")],
    })
    assert [t.text for t in syllable_timings(grid)] == ["ni3", "tian1", "tian1"]
