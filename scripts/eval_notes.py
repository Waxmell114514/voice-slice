"""Frame-level note pitch accuracy of a predicted score against a ground-truth score.

Usage: python scripts/eval_notes.py GT_SCORE.json [PRED_SCORE.json]
Without PRED, notes are re-segmented from the GT syllable timing (isolates the note segmenter).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanslice.song.notes import segment_syllable, tuning_offset  # noqa: E402
from humanslice.song.score import Note, Score  # noqa: E402


def frame_notes(notes: list[Note], frames: int, period: float = 0.01) -> np.ndarray:
    out = np.full(frames, np.nan)
    for note in notes:
        out[int(note.start / period):int(note.end / period)] = note.pitch
    return out


def main() -> int:
    gt = Score.load(sys.argv[1])
    if len(sys.argv) > 2:
        pred_notes = Score.load(sys.argv[2]).notes
    else:
        tuning = tuning_offset(gt.f0_hz)
        pred_notes = []
        for index, syllable in enumerate(gt.syllables):
            for number, (a, b, pitch) in enumerate(segment_syllable(gt.f0_hz, syllable.vowel_start, syllable.end, tuning)):
                pred_notes.append(Note(a, b, pitch, index, number > 0))
    frames = int(gt.duration / 0.01) + 1
    truth, pred = frame_notes(gt.notes, frames), frame_notes(pred_notes, frames)
    both = np.isfinite(truth) & np.isfinite(pred)
    diff = np.abs(truth[both] - pred[both])
    print(f"GT notes {len(gt.notes)}, predicted {len(pred_notes)}")
    print(f"frames covered: {both.sum()}/{np.isfinite(truth).sum()} ({both.sum() / max(np.isfinite(truth).sum(), 1) * 100:.1f}%)")
    print(f"exact pitch: {np.mean(diff == 0) * 100:.1f}%   within 1 st: {np.mean(diff <= 1) * 100:.1f}%   octave errors: {np.mean(diff >= 11) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
