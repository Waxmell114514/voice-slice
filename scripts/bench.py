"""Benchmark render quality across songs: ASR syllable error rate + pitch accuracy.

Usage: python scripts/bench.py BANK [--tag name] [--backend nsf|world] [--songs out/chengdu out/yunyan]
Each song dir needs vocal.wav + lyrics.txt; score_pred.json is created (and cached) if missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_render import edit_distance, pinyin_sequence  # noqa: E402

from humanslice.common.audio import load_wav, split_on_silence  # noqa: E402
from humanslice.corpus.units import Bank  # noqa: E402
from humanslice.pitch.f0 import clean_f0, hz_to_midi  # noqa: E402
from humanslice.render.engine import RenderOptions, render, write_outputs  # noqa: E402
from humanslice.render.select import SelectOptions  # noqa: E402
from humanslice.song.analyze import SongOptions, analyze_song  # noqa: E402
from humanslice.song.score import Score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bank")
    parser.add_argument("--tag", default="bench")
    parser.add_argument("--backend", default="nsf")
    parser.add_argument("--join-weight", type=float, default=1.0)
    parser.add_argument("--songs", nargs="+", default=["out/chengdu", "out/yunyan"])
    args = parser.parse_args()

    bank = Bank.open(args.bank)
    renders = []
    for song in map(Path, args.songs):
        score_path = song / "score_pred.json"
        if not score_path.exists():
            lyrics = (song / "lyrics.txt").read_text(encoding="utf-8")
            analyze_song(song / "vocal.wav", SongOptions(lyrics=lyrics), progress=lambda m: None).save(score_path)
        score = Score.load(score_path)
        options = RenderOptions(select=SelectOptions(join_weight=args.join_weight), backend=args.backend)
        result = render(bank, score, options, progress=lambda m: None)
        out = write_outputs(result, song / f"jinriki_{args.tag}.wav")
        renders.append((song, out, score, result.selection.transpose))

    from humanslice.corpus.asr import QwenASR
    from humanslice.pitch.rmvpe import RMVPE

    asr = QwenASR()
    rmvpe = RMVPE()
    total_err = total_ref = 0
    for song, out, score, transpose in renders:
        audio, sr = load_wav(out, 16000)
        chunks = split_on_silence(audio, sr, max_chunk=25.0, min_silence=0.4, pad=0.15)
        heard = "".join(asr.transcribe([audio[int(a * sr):int(b * sr)] for a, b in chunks], "zh"))
        reference = pinyin_sequence((song / "lyrics.txt").read_text(encoding="utf-8"))
        errors = edit_distance(reference, pinyin_sequence(heard))
        total_err += errors
        total_ref += len(reference)
        f0, _ = rmvpe.infer(audio)
        rendered = hz_to_midi(clean_f0(f0))
        target = hz_to_midi(score.f0_hz) + transpose
        n = min(rendered.size, target.size)
        both = np.isfinite(rendered[:n]) & np.isfinite(target[:n])
        cents = np.abs(rendered[:n][both] - target[:n][both]) * 100
        print(f"{song.name:10} SER {errors / len(reference) * 100:5.1f}%  pitch median {np.median(cents):4.0f}c  "
              f">50c {np.mean(cents > 50) * 100:4.1f}%   heard: {heard[:60]}")
    print(f"TOTAL [{args.tag}] SER {total_err / total_ref * 100:.1f}% ({total_err}/{total_ref})")
    (Path("out") / f"bench_{args.tag}.json").write_text(json.dumps({"ser": total_err / total_ref}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
