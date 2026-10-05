"""Score -> Standard MIDI File with pinyin lyric events (melody for DAWs / other editors)."""

from __future__ import annotations

from pathlib import Path

from humanslice.export.ustx import RESOLUTION, score_notes_in_ticks
from humanslice.song.score import Score


def write_midi(score: Score, path: str | Path, transpose: int = 0) -> Path:
    import mido

    midi = mido.MidiFile(ticks_per_beat=RESOLUTION, charset="utf-8")
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(score.tempo), time=0))
    track.append(mido.MetaMessage("track_name", name="Vocal", time=0))
    now = 0
    for note in score_notes_in_ticks(score, transpose):
        delta = note["position"] - now
        if note["lyric"] != "+":
            track.append(mido.MetaMessage("lyrics", text=note["lyric"], time=delta))
            delta = 0
        pitch = max(0, min(127, note["tone"]))
        track.append(mido.Message("note_on", note=pitch, velocity=100, time=delta))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=note["duration"]))
        now = note["position"] + note["duration"]
    path = Path(path)
    midi.save(path)
    return path
