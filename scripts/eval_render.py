"""Intelligibility of rendered vocals: Qwen3-ASR transcript vs lyrics, pinyin error rate.

Pinyin (toneless) is compared instead of characters so homophones count as correct —
for a jinriki the question is whether the syllables are heard, not which Hanzi.

Usage: python scripts/eval_render.py LYRICS.txt AUDIO.wav [AUDIO2.wav ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanslice.common.audio import load_wav, split_on_silence  # noqa: E402
from humanslice.corpus.asr import QwenASR  # noqa: E402
from humanslice.text.zh import text_to_syllables  # noqa: E402


def pinyin_sequence(text: str) -> list[str]:
    return [s.pinyin for s in text_to_syllables(text) if s is not None]


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def main() -> int:
    lyrics = Path(sys.argv[1]).read_text(encoding="utf-8")
    reference = pinyin_sequence(lyrics)
    asr = QwenASR()
    for path in sys.argv[2:]:
        audio, sr = load_wav(path, 16000)
        chunks = split_on_silence(audio, sr, max_chunk=25.0, min_silence=0.4, pad=0.15)
        texts = asr.transcribe([audio[int(a * sr):int(b * sr)] for a, b in chunks], "zh")
        hypothesis = pinyin_sequence("".join(texts))
        errors = edit_distance(reference, hypothesis)
        print(f"{Path(path).name:28} syllable error rate {errors / max(len(reference), 1) * 100:5.1f}%  "
              f"({errors}/{len(reference)})")
        print("   heard:", "".join(texts)[:120])
    return 0


if __name__ == "__main__":
    sys.exit(main())
