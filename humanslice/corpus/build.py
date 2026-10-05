"""Speech material -> unit database pipeline.

Stages (each persisted in the bank directory, re-runs skip finished stages):

1. ingest      decode every material file to a 44.1 kHz mono master (``audio/<sid>.wav``)
2. segment     split into utterances at pauses (or one utterance per file with a sidecar label)
3. transcribe  Qwen3-ASR for utterances without a label
4. align       MFA phone alignment (preferred) or Qwen3-ForcedAligner character timestamps
5. features    RMVPE pitch, vowel nucleus, energy per unit
6. qc          reject units that are too short, unvoiced or badly aligned
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from humanslice.common.audio import MEDIA_EXTENSIONS, decode, frame_rms_db, load_wav, save_wav, split_on_silence
from humanslice.corpus.units import MASTER_SAMPLE_RATE, Bank, Unit, Utterance, link_contiguous
from humanslice.pitch.f0 import FRAME_PERIOD, clean_f0, hz_to_midi, voiced_runs
from humanslice.text.zh import Syllable, main_vowel_position, text_to_syllables

log = logging.getLogger(__name__)
ANALYSIS_SAMPLE_RATE = 16000
LABEL_SUFFIXES = (".lab", ".txt")

Progress = Callable[[str], None]


@dataclass(slots=True)
class BuildOptions:
    language: str = "zh"
    aligner: str = "auto"  # "mfa", "qwen", "auto" (MFA if its environment exists)
    asr_model: str | None = None
    device: str | None = None
    max_utterance_sec: float = 20.0
    use_labels: bool = True  # sidecar .lab/.txt transcripts next to material files
    separate: bool = False  # strip BGM / noise with a vocal separation model before analysis
    min_unit_sec: float = 0.06
    min_final_sec: float = 0.04


def build_bank(
    sources: Iterable[str | Path],
    bank_dir: str | Path,
    options: BuildOptions | None = None,
    progress: Progress = print,
    force: bool = False,
    refeature: bool = False,
) -> Bank:
    options = options or BuildOptions()
    bank = Bank.open(bank_dir)
    if force:
        bank.utterances, bank.units = [], []
    bank.meta.setdefault("language", options.language)
    bank.meta.setdefault("sample_rate", MASTER_SAMPLE_RATE)

    files = _collect_files(sources)
    if not files:
        raise ValueError("No audio / video material found")
    new_sources = ingest(bank, files, progress, separate=options.separate)
    if new_sources or not bank.utterances:
        segment(bank, options, progress, only_sources=new_sources if bank.utterances else None)
        bank.save()
    transcribe(bank, options, progress)
    bank.save()
    attempted = set(bank.meta.get("aligned_utterances", [])) | {unit.utt_id for unit in bank.units}
    pending = {utt.utt_id for utt in bank.utterances} - attempted
    if pending:
        align(bank, options, progress, only_utts=pending)
        features(bank, options, progress, only_utts=pending)
        bank.meta["aligned_utterances"] = sorted(attempted | pending)
    if refeature:
        features(bank, options, progress, only_utts={unit.utt_id for unit in bank.units} - pending)
    quality_control(bank, options)
    link_contiguous(bank.units)
    bank.save()
    usable = len(bank.usable_units())
    progress(f"Bank ready: {len(bank.units)} units ({usable} usable), {len(bank.units_by_pinyin())} distinct syllables")
    return bank


# ---------------------------------------------------------------- stage 1


def _collect_files(sources: Iterable[str | Path]) -> list[Path]:
    files: list[Path] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in MEDIA_EXTENSIONS))
        elif path.suffix.lower() in MEDIA_EXTENSIONS:
            files.append(path)
    return files


def source_id_for(path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", path.stem)[:32].strip("_") or "src"
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def ingest(bank: Bank, files: list[Path], progress: Progress, separate: bool = False) -> list[str]:
    sources = bank.meta.setdefault("sources", {})
    added: list[str] = []
    for index, path in enumerate(files, start=1):
        sid = source_id_for(path)
        if sid in sources and bank.master_path(sid).exists():
            continue
        decoded_from = path
        if separate:
            from humanslice.common.separate import separate as separate_vocals

            decoded_from, _ = separate_vocals(path)
        samples = decode(decoded_from, MASTER_SAMPLE_RATE)
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        if peak > 1.0:
            samples = samples / peak
        save_wav(bank.master_path(sid), samples, MASTER_SAMPLE_RATE, subtype="FLOAT")
        label = _read_label(path)
        sources[sid] = {"path": str(path.resolve()), "duration": samples.size / MASTER_SAMPLE_RATE, "label": label}
        added.append(sid)
        if index % 25 == 0 or index == len(files):
            progress(f"[ingest] {index}/{len(files)} files")
    return added


def _read_label(path: Path) -> str | None:
    for suffix in LABEL_SUFFIXES:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace").strip()
    return None


# ---------------------------------------------------------------- stage 2


def segment(bank: Bank, options: BuildOptions, progress: Progress, only_sources: list[str] | None = None) -> None:
    sources = bank.meta["sources"]
    targets = only_sources if only_sources is not None else list(sources)
    if only_sources is None:
        bank.utterances = []
    for sid in targets:
        info = sources[sid]
        label = info.get("label") if options.use_labels else None
        if label and info["duration"] <= 60.0:
            bank.utterances.append(Utterance(f"{sid}_u000", sid, 0.0, info["duration"], label, "label"))
            continue
        samples, sr = load_wav(bank.master_path(sid), ANALYSIS_SAMPLE_RATE)
        spans = split_on_silence(samples, sr, max_chunk=options.max_utterance_sec)
        for number, (start, end) in enumerate(spans):
            bank.utterances.append(Utterance(f"{sid}_u{number:03d}", sid, start, end))
    progress(f"[segment] {len(bank.utterances)} utterances")


def utterance_audio(bank: Bank, utt: Utterance, sample_rate: int = ANALYSIS_SAMPLE_RATE, cache: dict | None = None) -> np.ndarray:
    key = (utt.source_id, sample_rate)
    if cache is not None and key in cache:
        samples = cache[key]
    else:
        samples, _ = load_wav(bank.master_path(utt.source_id), sample_rate)
        if cache is not None:
            cache.clear()
            cache[key] = samples
    return samples[int(utt.start * sample_rate):int(utt.end * sample_rate)]


# ---------------------------------------------------------------- stage 3


def transcribe(bank: Bank, options: BuildOptions, progress: Progress) -> None:
    pending = [utt for utt in bank.utterances if not utt.text]
    if not pending:
        return
    from humanslice.corpus.asr import ASR_MODEL, QwenASR

    progress(f"[transcribe] {len(pending)} utterances with {options.asr_model or ASR_MODEL}")
    asr = QwenASR(options.asr_model or ASR_MODEL, device=options.device)
    cache: dict = {}
    try:
        batch_size = 8
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            texts = asr.transcribe([utterance_audio(bank, utt, cache=cache) for utt in batch], options.language)
            for utt, text in zip(batch, texts):
                utt.text, utt.text_origin = text, "asr"
            progress(f"[transcribe] {min(offset + batch_size, len(pending))}/{len(pending)}")
    finally:
        asr.close()


# ---------------------------------------------------------------- stage 4


def _resolve_aligner(options: BuildOptions) -> str:
    if options.aligner != "auto":
        return options.aligner
    from humanslice.corpus.mfa import DEFAULT_ENV

    return "mfa" if (DEFAULT_ENV / "conda-meta").exists() else "qwen"


def align(bank: Bank, options: BuildOptions, progress: Progress, only_utts: set[str]) -> None:
    utts = [utt for utt in bank.utterances if utt.utt_id in only_utts and utt.text]
    syllables = {utt.utt_id: text_to_syllables(utt.text) for utt in utts}
    method = _resolve_aligner(options)
    progress(f"[align] {len(utts)} utterances with {method}")
    cache: dict = {}
    if method == "mfa":
        timings = _align_mfa(bank, utts, syllables, options, cache)
    else:
        timings = _align_qwen(bank, utts, syllables, options, cache)

    counter = len(bank.units)
    for utt in utts:
        for syllable, timing in zip(_hanzi_only(syllables[utt.utt_id]), timings.get(utt.utt_id, [])):
            if timing is None:
                continue
            start, end, boundary = timing
            counter += 1
            bank.units.append(
                Unit(
                    unit_id=f"u{counter:06d}",
                    source_id=utt.source_id,
                    utt_id=utt.utt_id,
                    start=utt.start + start,
                    end=utt.start + end,
                    cv_boundary=utt.start + (boundary if syllable.initial else start),
                    char=syllable.char,
                    pinyin=syllable.pinyin,
                    tone=syllable.tone,
                    initial=syllable.initial,
                    final=syllable.final,
                    align_source=method,
                )
            )
    progress(f"[align] {len(bank.units)} units")


def _hanzi_only(syllables: list[Syllable | None]) -> list[Syllable]:
    return [syllable for syllable in syllables if syllable is not None]


def _align_mfa(bank, utts, syllables, options, cache) -> dict[str, list[tuple[float, float, float] | None]]:
    from humanslice.corpus.mfa import AlignItem, MFARunner, syllable_timings

    runner = MFARunner(language=options.language)
    items = [
        AlignItem(
            utt.utt_id,
            utterance_audio(bank, utt, cache=cache),
            [f"{s.pinyin}{s.tone}" for s in _hanzi_only(syllables[utt.utt_id])],
        )
        for utt in utts
    ]
    items = [item for item in items if item.tokens]
    grids = runner.align(items)
    result: dict[str, list[tuple[float, float, float] | None]] = {}
    for item in items:
        grid = grids.get(item.utt_id)
        if grid is None:
            continue
        timings = syllable_timings(grid)
        if len(timings) != len(item.tokens):
            log.warning("MFA word count mismatch for %s (%d vs %d)", item.utt_id, len(timings), len(item.tokens))
            continue
        result[item.utt_id] = [None if t is None else (t.start, t.end, t.cv_boundary) for t in timings]
    return result


def _align_qwen(bank, utts, syllables, options, cache) -> dict[str, list[tuple[float, float, float] | None]]:
    from humanslice.corpus.asr import QwenAligner

    aligner = QwenAligner(device=options.device)
    result: dict[str, list[tuple[float, float, float] | None]] = {}
    try:
        for utt in utts:
            chars = _hanzi_only(syllables[utt.utt_id])
            if not chars:
                continue
            audio = utterance_audio(bank, utt, cache=cache)
            tokens = aligner.align(audio, "".join(s.char for s in chars), options.language)
            if len(tokens) != len(chars):
                log.warning("Qwen aligner token mismatch for %s", utt.utt_id)
                continue
            result[utt.utt_id] = [
                (token.start, token.end, estimate_cv_boundary(audio, token.start, token.end, syllable))
                for token, syllable in zip(tokens, chars)
            ]
    finally:
        aligner.close()
    return result


def estimate_cv_boundary(audio_16k: np.ndarray, start: float, end: float, syllable: Syllable) -> float:
    """Heuristic initial/final boundary when no phone aligner is available.

    Voiceless initials end where periodic energy begins (low zero-crossing rate + energy rise);
    sonorant initials (m n l r) get a fixed short share of the syllable.
    """
    if not syllable.initial:
        return start
    if syllable.initial in ("m", "n", "l", "r"):
        return start + min(0.06, 0.25 * (end - start))
    sr = ANALYSIS_SAMPLE_RATE
    clip = audio_16k[int(start * sr):int(end * sr)]
    if clip.size < 160:
        return start + 0.3 * (end - start)
    hop = 80
    frames = np.lib.stride_tricks.sliding_window_view(clip, 320)[::hop]
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    energy = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-10)
    voiced = (zcr < 0.15) & (energy > energy.max() - 20)
    run = voiced_runs(voiced)
    if not run:
        return start + 0.3 * (end - start)
    first = max(run, key=lambda r: r[1] - r[0])[0]
    return start + min(first * hop / sr, 0.7 * (end - start))


# ---------------------------------------------------------------- stage 5


def features(bank: Bank, options: BuildOptions, progress: Progress, only_utts: set[str]) -> None:
    from humanslice.common.models import release_gpu_memory
    from humanslice.pitch.rmvpe import RMVPE

    by_utt: dict[str, list[Unit]] = {}
    for unit in bank.units:
        if unit.utt_id in only_utts:
            by_utt.setdefault(unit.utt_id, []).append(unit)
    utt_map = bank.utterance_map()
    progress(f"[features] pitch / energy for {len(by_utt)} utterances")
    rmvpe = RMVPE(device=options.device)
    cache: dict = {}
    try:
        for count, (utt_id, units) in enumerate(by_utt.items(), start=1):
            utt = utt_map[utt_id]
            audio = utterance_audio(bank, utt, cache=cache)
            f0, _ = rmvpe.infer(audio)
            f0 = clean_f0(f0)
            energy = frame_rms_db(audio, ANALYSIS_SAMPLE_RATE, FRAME_PERIOD)
            flux = spectral_flux(audio)
            for unit in units:
                describe_unit(unit, f0, energy, offset=utt.start, flux=flux)
            if count % 100 == 0:
                progress(f"[features] {count}/{len(by_utt)}")
    finally:
        del rmvpe
        release_gpu_memory()


def spectral_flux(audio_16k: np.ndarray, period: float = FRAME_PERIOD) -> np.ndarray:
    """Frame-to-frame change of a 40-band log-mel spectrum (10 ms grid, frame i centred at i*period)."""
    import librosa

    hop = int(round(period * ANALYSIS_SAMPLE_RATE))
    mel = librosa.feature.melspectrogram(
        y=audio_16k.astype(np.float32), sr=ANALYSIS_SAMPLE_RATE, n_fft=512, hop_length=hop, n_mels=40, fmax=7600
    )
    log_mel = np.log(mel + 1e-6)
    change = np.mean(np.abs(np.diff(log_mel, axis=1)), axis=0)
    return np.concatenate([[change[0] if change.size else 0.0], change]).astype(np.float32)


def describe_unit(
    unit: Unit,
    f0: np.ndarray,
    energy_db: np.ndarray,
    offset: float,
    flux: np.ndarray | None = None,
    window: int = 6,
) -> None:
    """Fill pitch / energy / nucleus features from frame-level analysis of the utterance.

    The nucleus is the *spectrally* steadiest stretch of the final (the main vowel of a
    diphthong such as the ``a`` of ``iao``): looping it on long notes holds one vowel
    instead of sliding back and forth through the glide.
    """
    a = int(round((unit.cv_boundary - offset) / FRAME_PERIOD))
    b = int(round((unit.end - offset) / FRAME_PERIOD))
    a, b = max(0, a), min(f0.size, max(b, a + 1))
    final_f0 = f0[a:b]
    final_energy = energy_db[a:min(b, energy_db.size)]
    voiced = final_f0 > 0
    unit.voiced_ratio = float(voiced.mean()) if final_f0.size else 0.0
    unit.energy_db = float(np.mean(final_energy)) if final_energy.size else None
    unit.nucleus_start, unit.nucleus_end = unit.cv_boundary, unit.end
    if voiced.sum() < 2:
        unit.f0_midi = unit.f0_std = None
        return
    midi = hz_to_midi(final_f0[voiced])
    unit.f0_midi = float(np.median(midi))
    unit.f0_std = float(np.std(midi))

    loud = final_energy > (np.max(final_energy) - 15.0) if final_energy.size == final_f0.size else voiced
    if flux is None or flux.size < b or b - a < window + 2:
        return
    final_flux = flux[a:b].astype(np.float64)
    final_flux[~(voiced & loud)] = np.inf  # never sustain on unvoiced / quiet frames
    if not np.isfinite(final_flux).any():
        return
    scores = np.convolve(np.where(np.isfinite(final_flux), final_flux, 1e3), np.ones(window) / window, mode="valid")
    # Prefer the main vowel's usual position (avoids holding an off-glide or nasal coda).
    centres = (np.arange(scores.size) + window / 2) / final_flux.size
    best = int(np.argmin(scores * (1.0 + 2.0 * np.abs(centres - main_vowel_position(unit.final)))))
    # Grow the steady window while neighbouring frames stay close to its stability level.
    limit = max(float(scores[best]) * 1.6, float(scores[best]) + 0.05)
    max_len = max(window, int(0.45 * final_flux.size))
    s, e = best, best + window
    while e - s < max_len:
        grow_left = s > 0 and final_flux[s - 1] <= limit
        grow_right = e < final_flux.size and final_flux[e] <= limit
        if not (grow_left or grow_right):
            break
        if grow_left and (not grow_right or final_flux[s - 1] <= final_flux[e]):
            s -= 1
        else:
            e += 1
    unit.nucleus_start = offset + (a + s) * FRAME_PERIOD
    unit.nucleus_end = min(unit.end, offset + (a + e) * FRAME_PERIOD)


# ---------------------------------------------------------------- stage 6


def quality_control(bank: Bank, options: BuildOptions) -> None:
    for unit in bank.units:
        reason = ""
        if unit.duration < options.min_unit_sec:
            reason = "too_short"
        elif unit.final_duration < options.min_final_sec:
            reason = "final_too_short"
        elif unit.voiced_ratio < 0.3:
            reason = "unvoiced"
        elif unit.align_error is not None and unit.align_error > 0.08:
            reason = "misaligned"
        unit.rejected = reason
