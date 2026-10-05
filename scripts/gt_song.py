"""Join a GTSinger song's segments into one acapella and build its ground-truth Score.

Usage: python scripts/gt_song.py devdata/songs/ZH-Alto-1_成都_Breathy -o out/chengdu
Writes <out>/vocal.wav, <out>/lyrics.txt and <out>/score_gt.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanslice.common.audio import frame_rms_db, load_wav, resample, save_wav  # noqa: E402
from humanslice.pitch.f0 import FRAME_PERIOD, clean_f0  # noqa: E402
from humanslice.song.score import Note, Score, SungSyllable  # noqa: E402
from humanslice.text.zh import split_pinyin, text_to_syllables  # noqa: E402

GAP = 0.6


def build(folder: Path, out: Path, sample_rate: int = 44100) -> Score:
    out.mkdir(parents=True, exist_ok=True)
    pieces, syllables, notes, lyrics = [], [], [], []
    offset = 0.0
    for wav in sorted(folder.glob("*.wav")):
        audio, _ = load_wav(wav, sample_rate)
        words = json.loads(wav.with_suffix(".json").read_text(encoding="utf-8"))
        chars = [w for w in words if not w["word"].startswith("<")]
        readings = [s for s in text_to_syllables("".join(w["word"] for w in chars))]
        for word, reading in zip(chars, readings):
            if reading is None:
                continue
            vowel_start = word["ph_end"][0] if len(word["ph"]) >= 2 else word["start_time"]
            index = len(syllables)
            syllables.append(
                SungSyllable(word["word"], reading.pinyin, reading.tone, *split_pinyin(reading.pinyin),
                             offset + word["start_time"], offset + vowel_start, offset + word["end_time"])
            )
            for number, (pitch, start, end) in enumerate(zip(word["note"], word["note_start"], word["note_end"])):
                if pitch <= 0:
                    continue
                notes.append(Note(offset + max(start, vowel_start) if number == 0 else offset + start,
                                  offset + end, int(pitch), index, slur=number > 0))
        lyrics.append("".join(w["word"] for w in chars))
        pieces += [audio, np.zeros(int(GAP * sample_rate), dtype=np.float32)]
        offset += audio.size / sample_rate + GAP
    vocal = np.concatenate(pieces)
    save_wav(out / "vocal.wav", vocal, sample_rate)
    (out / "lyrics.txt").write_text("\n".join(lyrics), encoding="utf-8")

    from humanslice.pitch.rmvpe import RMVPE

    audio_16k = resample(vocal, sample_rate, 16000)
    f0, _ = RMVPE().infer(audio_16k)
    score = Score(syllables, notes, clean_f0(f0), frame_rms_db(audio_16k, 16000, FRAME_PERIOD), FRAME_PERIOD,
                  vocal.size / sample_rate, 120.0, str(out / "vocal.wav"))
    score.save(out / "score_gt.json")
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    score = build(Path(args.folder), Path(args.output))
    print(f"{len(score.syllables)} syllables, {len(score.notes)} notes, {score.duration:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
