"""Measure syllable / initial-final boundary accuracy against GTSinger ground truth.

Usage: python scripts/eval_alignment.py [--aligner mfa|qwen] [--limit 60] [--dir devdata/corpus]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from humanslice.common.audio import load_wav  # noqa: E402
from humanslice.text.zh import text_to_syllables  # noqa: E402


def ground_truth(json_path: Path) -> list[dict]:
    words = json.loads(json_path.read_text(encoding="utf-8"))
    result = []
    for word in words:
        if word["word"].startswith("<"):
            continue
        boundary = word["ph_end"][0] if len(word["ph"]) >= 2 else word["start_time"]
        result.append({"char": word["word"], "start": word["start_time"], "end": word["end_time"], "cv": boundary})
    return result


def report(name: str, errors: list[float]) -> str:
    e = np.abs(np.asarray(errors)) * 1000
    signed = np.asarray(errors) * 1000
    return (f"{name:10} n={e.size:5d} bias={signed.mean():+6.1f}ms mean={e.mean():6.1f}ms median={np.median(e):6.1f}ms "
            f"<20ms={np.mean(e < 20) * 100:5.1f}% <50ms={np.mean(e < 50) * 100:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligner", default="mfa", choices=["mfa", "qwen"])
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--dir", default="devdata/corpus")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))[: args.limit]
    items, truths = [], {}
    for index, json_path in enumerate(files):
        wav = json_path.with_suffix(".wav")
        if not wav.exists():
            continue
        truth = ground_truth(json_path)
        text = "".join(t["char"] for t in truth)
        syllables = [s for s in text_to_syllables(text) if s is not None]
        if len(syllables) != len(truth):
            continue
        audio, _ = load_wav(wav, 16000)
        utt_id = f"utt{index:04d}"
        items.append((utt_id, audio, syllables))
        truths[utt_id] = truth

    predictions: dict[str, list[tuple[float, float, float] | None]] = {}
    if args.aligner == "mfa":
        from humanslice.corpus.mfa import AlignItem, MFARunner, syllable_timings

        runner = MFARunner()
        grids = runner.align([AlignItem(u, a, [f"{s.pinyin}{s.tone}" for s in syl]) for u, a, syl in items])
        for utt_id, _, syl in items:
            if utt_id in grids:
                timings = syllable_timings(grids[utt_id])
                if len(timings) == len(syl):
                    predictions[utt_id] = [None if t is None else (t.start, t.end, t.cv_boundary) for t in timings]
    else:
        from humanslice.corpus.asr import QwenAligner
        from humanslice.corpus.build import estimate_cv_boundary

        aligner = QwenAligner()
        for utt_id, audio, syl in items:
            tokens = aligner.align(audio, "".join(s.char for s in syl))
            if len(tokens) == len(syl):
                predictions[utt_id] = [
                    (t.start, t.end, estimate_cv_boundary(audio, t.start, t.end, s)) for t, s in zip(tokens, syl)
                ]

    starts, ends, cvs = [], [], []
    for utt_id, predicted in predictions.items():
        for truth, pred in zip(truths[utt_id], predicted):
            if pred is None:
                continue
            starts.append(pred[0] - truth["start"])
            ends.append(pred[1] - truth["end"])
            if truth["cv"] > truth["start"]:
                cvs.append(pred[2] - truth["cv"])
    print(f"{args.aligner}: {len(predictions)}/{len(items)} utterances aligned")
    print(report("start", starts))
    print(report("end", ends))
    print(report("cv", cvs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
