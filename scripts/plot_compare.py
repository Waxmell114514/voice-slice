"""Plot spectrogram + F0 of the target vocal vs a rendered vocal for a time window.

Usage: python scripts/plot_compare.py TARGET.wav RENDER.wav OUT.png [--start 0] [--dur 10] [--transpose -12]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanslice.common.audio import load_wav, resample  # noqa: E402
from humanslice.pitch.f0 import clean_f0, hz_to_midi  # noqa: E402
from humanslice.pitch.rmvpe import RMVPE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("target")
    parser.add_argument("render")
    parser.add_argument("out")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--dur", type=float, default=10.0)
    parser.add_argument("--transpose", type=int, default=0)
    args = parser.parse_args()
    rmvpe = RMVPE()
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    curves = {}
    for ax, path, title in ((axes[0], args.target, "target"), (axes[1], args.render, "render")):
        audio, sr = load_wav(path, 22050)
        a, b = int(args.start * sr), int((args.start + args.dur) * sr)
        clip = audio[a:b]
        ax.specgram(clip + 1e-7, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-120)
        ax.set_ylim(0, 8000)
        ax.set_title(f"{title}: {Path(path).name}")
        f0, _ = rmvpe.infer(resample(clip, sr, 16000))
        curves[title] = hz_to_midi(clean_f0(f0))
    t = np.arange(curves["target"].size) * 0.01
    axes[2].plot(t, curves["target"] + args.transpose, label=f"target {args.transpose:+d}", lw=1)
    t2 = np.arange(curves["render"].size) * 0.01
    axes[2].plot(t2, curves["render"], label="render", lw=1)
    axes[2].set_ylabel("MIDI")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=80)
    both = np.isfinite(curves["target"][: curves["render"].size]) & np.isfinite(curves["render"][: curves["target"].size])
    n = min(curves["target"].size, curves["render"].size)
    diff = (curves["render"][:n] - curves["target"][:n] - args.transpose)[both[:n]]
    print(f"pitch error over {diff.size} frames: median |Δ| = {np.median(np.abs(diff)) * 100:.0f} cents, "
          f">50c: {np.mean(np.abs(diff) > 0.5) * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
