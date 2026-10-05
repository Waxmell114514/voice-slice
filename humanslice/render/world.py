"""WORLD vocoder backend: parameter-domain concatenation of the planned units.

For each phrase, spectral envelopes (log domain) and aperiodicities of the placed units
are warped onto the output frame grid and blended with the placements' crossfade
weights; the excitation follows the target vocal's F0 (transposed). Voicing comes from
the source frames in consonants and is forced on in vowels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from humanslice.common.audio import load_wav
from humanslice.corpus.units import Bank, Unit
from humanslice.pitch.f0 import interpolate_unvoiced
from humanslice.render.plan import Placement, phrases

FRAME_MS = 5.0
FRAME = FRAME_MS / 1000.0


@dataclass(slots=True)
class UnitFeatures:
    t0: float  # source time of frame 0
    f0: np.ndarray
    sp: np.ndarray
    ap: np.ndarray

    def index(self, src_times: np.ndarray) -> np.ndarray:
        return np.clip(np.round((src_times - self.t0) / FRAME).astype(int), 0, self.f0.size - 1)


class WorldAnalyzer:
    def __init__(self, bank: Bank, margin: float = 0.12) -> None:
        self.bank = bank
        self.fs = int(bank.meta.get("sample_rate", 44100))
        self.margin = margin
        self._sources: dict[str, np.ndarray] = {}
        self._cache: dict[str, UnitFeatures] = {}

    def _source(self, source_id: str) -> np.ndarray:
        if source_id not in self._sources:
            if len(self._sources) > 16:
                self._sources.clear()
            self._sources[source_id], _ = load_wav(self.bank.master_path(source_id), self.fs)
        return self._sources[source_id]

    def features(self, unit: Unit) -> UnitFeatures:
        cached = self._cache.get(unit.unit_id)
        if cached is not None:
            return cached
        import pyworld as pw

        audio = self._source(unit.source_id)
        a = max(0, int((unit.start - self.margin) * self.fs))
        b = min(audio.size, int((unit.end + self.margin) * self.fs))
        x = audio[a:b].astype(np.float64)
        f0, t = pw.harvest(x, self.fs, f0_floor=55.0, f0_ceil=1100.0, frame_period=FRAME_MS)
        sp = pw.cheaptrick(x, f0, t, self.fs)
        ap = pw.d4c(x, f0, t, self.fs)
        features = UnitFeatures(t0=a / self.fs, f0=f0, sp=sp, ap=ap)
        self._cache[unit.unit_id] = features
        return features


@dataclass(slots=True)
class WorldOptions:
    level_follow: float = 0.7  # how much of the target vocal's dynamics to impose
    max_gain_db: float = 12.0
    pitch_follow: float = 1.0  # 1 = exact target F0 (incl. vibrato); 0 = flat note pitches
    loop_smoothing: float = 0.7  # 0 = raw ping-pong loop, 1 = static averaged vowel


def render_world(
    bank: Bank,
    placements: list[Placement],
    target_f0: np.ndarray,
    target_period: float,
    duration: float,
    transpose: int,
    loudness_db: np.ndarray | None = None,
    note_curve: np.ndarray | None = None,
    options: WorldOptions | None = None,
    progress=None,
) -> tuple[np.ndarray, int]:
    import pyworld as pw

    options = options or WorldOptions()
    analyzer = WorldAnalyzer(bank)
    fs = analyzer.fs
    out = np.zeros(int(np.ceil((duration + 1.0) * fs)), dtype=np.float64)

    # Target pitch on the output grid (Hz), gaps bridged so vowels are always voiced.
    filled = interpolate_unvoiced(target_f0) if np.any(target_f0 > 0) else np.full_like(target_f0, 220.0)
    target_midi = 69 + 12 * np.log2(np.maximum(filled, 1e-3) / 440.0)
    if note_curve is not None and options.pitch_follow < 1.0:
        target_midi = note_curve + options.pitch_follow * (target_midi - note_curve)
    target_midi = target_midi + transpose

    loud = None
    if loudness_db is not None and loudness_db.size:
        voiced_level = loudness_db[target_f0[: loudness_db.size] > 0] if np.any(target_f0 > 0) else loudness_db
        loud = loudness_db - float(np.median(voiced_level))
    energies = [p.unit.energy_db for p in placements if p.unit.energy_db is not None]
    reference = float(np.median(energies)) if energies else -20.0

    groups = phrases(placements)
    for number, group in enumerate(groups, start=1):
        t_start = min(p.out_start for p in group) - 0.02
        t_end = max(p.out_end for p in group) + 0.02
        times = t_start + np.arange(int(np.ceil((t_end - t_start) / FRAME)) + 1) * FRAME
        acc_sp = acc_ap = None
        acc_w = np.zeros(times.size)
        acc_voiced = np.zeros(times.size)
        for placement in group:
            mask = (times >= placement.out_start) & (times <= placement.out_end)
            if not mask.any():
                continue
            ts = times[mask]
            w = placement.weight(ts)
            feats = analyzer.features(placement.unit)
            idx = feats.index(placement.source_time(ts))
            log_sp = np.log(np.maximum(feats.sp[idx], 1e-16))
            if placement.pingpong is not None:
                # Looping a short nucleus makes the timbre wobble at the loop rate; pull the
                # looped frames halfway towards the nucleus' average envelope.
                out_a, out_b, src_a, src_b, _ = placement.pingpong
                loop = (ts >= out_a) & (ts < out_b)
                nucleus = feats.index(np.arange(src_a, src_b, FRAME))
                mean_env = np.log(np.maximum(feats.sp[nucleus], 1e-16)).mean(axis=0)
                log_sp[loop] = (1.0 - options.loop_smoothing) * log_sp[loop] + options.loop_smoothing * mean_env
            gain_db = np.clip(reference - (placement.unit.energy_db or reference), -options.max_gain_db, options.max_gain_db)
            gains = np.full(ts.size, gain_db)
            if loud is not None:
                li = np.clip(np.round(ts / target_period).astype(int), 0, loud.size - 1)
                gains = gains + np.clip(options.level_follow * loud[li], -15.0, 6.0)
            log_sp = log_sp + (gains * np.log(10.0) / 10.0)[:, None]
            voiced = (feats.f0[idx] > 0) | (ts >= placement.onset)
            if acc_sp is None:
                acc_sp = np.zeros((times.size, log_sp.shape[1]))
                acc_ap = np.zeros((times.size, log_sp.shape[1]))
            acc_sp[mask] += w[:, None] * log_sp
            acc_ap[mask] += w[:, None] * feats.ap[idx]
            acc_w[mask] += w
            acc_voiced[mask] += w * voiced
        if acc_sp is None:
            continue

        present = acc_w > 1e-4
        norm = np.where(present, acc_w, 1.0)
        sp = np.exp(acc_sp / norm[:, None])
        sp *= (np.minimum(acc_w, 1.0) ** 2)[:, None]  # fades at phrase edges
        sp[~present] = 1e-16
        ap = np.where(present[:, None], acc_ap / norm[:, None], 1.0)
        ap = np.clip(ap, 0.0, 1.0)
        fi = np.clip(np.round(times / target_period).astype(int), 0, target_midi.size - 1)
        f0 = 440.0 * 2.0 ** ((target_midi[fi] - 69.0) / 12.0)
        f0[(acc_voiced / norm < 0.5) | ~present] = 0.0

        y = pw.synthesize(
            np.ascontiguousarray(f0, dtype=np.float64),
            np.ascontiguousarray(sp, dtype=np.float64),
            np.ascontiguousarray(ap, dtype=np.float64),
            fs,
            FRAME_MS,
        )
        a = int(round(t_start * fs))
        if a < 0:
            y, a = y[-a:], 0
        b = min(out.size, a + y.size)
        out[a:b] += y[: b - a]
        if progress is not None and (number % 10 == 0 or number == len(groups)):
            progress(f"[render] phrase {number}/{len(groups)}")

    out = out[: int(np.ceil(duration * fs))]
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 0:
        out *= 10 ** (-1.0 / 20) / peak
    return out.astype(np.float32), fs
