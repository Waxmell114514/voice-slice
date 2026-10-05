"""Target vocal -> Score (syllable timing, notes, F0, loudness).

1. pitch      RMVPE F0 (10 ms) with octave-error cleanup
2. lyrics     user-supplied lyrics (preferred) or Qwen3-ASR on pause-separated chunks
3. timing     Qwen3-ForcedAligner character timestamps, consonant/vowel split by voicing onset
4. notes      per-syllable F0 segmentation into (slur) notes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from humanslice.common.audio import frame_rms_db, load_wav, resample, split_on_silence
from humanslice.corpus.build import estimate_cv_boundary
from humanslice.pitch.f0 import FRAME_PERIOD, clean_f0
from humanslice.song.notes import segment_syllable, tuning_offset
from humanslice.song.score import Note, Score, SungSyllable
from humanslice.text.zh import Syllable, text_to_syllables

log = logging.getLogger(__name__)
ANALYSIS_SR = 16000
ALIGN_MAX_SEC = 280.0  # Qwen3-ForcedAligner accepts up to 5 minutes per call


@dataclass(slots=True)
class SongOptions:
    language: str = "zh"
    lyrics: str | None = None
    device: str | None = None
    tempo: float = 120.0
    phone_aligner: str = "auto"  # "auto" (MFA when installed), "mfa", "none"


def analyze_song(path: str | Path, options: SongOptions | None = None, progress: Callable[[str], None] = print) -> Score:
    options = options or SongOptions()
    audio, sr = load_wav(path)
    audio_16k = resample(audio, sr, ANALYSIS_SR)
    duration = audio.size / sr

    progress("[song] pitch (RMVPE)")
    from humanslice.common.models import release_gpu_memory
    from humanslice.pitch.rmvpe import RMVPE

    rmvpe = RMVPE(device=options.device)
    f0, _ = rmvpe.infer(audio_16k)
    del rmvpe
    release_gpu_memory()
    f0 = clean_f0(f0)
    loudness = frame_rms_db(audio_16k, ANALYSIS_SR, FRAME_PERIOD)

    chunks = _chunks(audio_16k, duration)
    if options.lyrics:
        texts = None
        syllables = [s for s in text_to_syllables(options.lyrics) if s is not None]
    else:
        progress(f"[song] lyrics (Qwen3-ASR, {len(chunks)} chunks)")
        from humanslice.corpus.asr import QwenASR

        asr = QwenASR(device=options.device)
        try:
            texts = asr.transcribe([audio_16k[int(a * ANALYSIS_SR):int(b * ANALYSIS_SR)] for a, b in chunks], options.language)
        finally:
            asr.close()
        syllables = []
        for text in texts:
            syllables.extend(s for s in text_to_syllables(text) if s is not None)
        progress(f"[song] recognized: {''.join(texts)}")

    progress(f"[song] aligning {len(syllables)} syllables")
    timed = _align(audio_16k, duration, syllables, chunks, texts, options)

    sung: list[SungSyllable] = []
    for syllable, (start, end) in timed:
        vowel_start = estimate_cv_boundary(audio_16k, start, end, syllable)
        sung.append(SungSyllable(syllable.char, syllable.pinyin, syllable.tone, syllable.initial, syllable.final,
                                 start, vowel_start, end))
    if options.phone_aligner != "none":
        refined = refine_with_mfa(audio_16k, sung, options)
        progress(f"[song] MFA refined {refined}/{len(sung)} syllables")
    sung = _refine_ends(sung, f0)

    tuning = tuning_offset(f0)
    notes: list[Note] = []
    for index, syllable in enumerate(sung):
        pieces = segment_syllable(f0, syllable.vowel_start, syllable.end, tuning)
        if not pieces:
            pitch = notes[-1].pitch if notes else 60
            pieces = [(syllable.vowel_start, syllable.end, pitch)]
        for number, (a, b, pitch) in enumerate(pieces):
            notes.append(Note(start=a, end=b, pitch=pitch, syllable=index, slur=number > 0))
    progress(f"[song] {len(sung)} syllables, {len(notes)} notes, tuning {tuning:+.2f} st")
    return Score(sung, notes, f0, loudness, FRAME_PERIOD, duration, options.tempo, str(path))


def _chunks(audio_16k: np.ndarray, duration: float) -> list[tuple[float, float]]:
    spans = split_on_silence(audio_16k, ANALYSIS_SR, max_chunk=25.0, min_silence=0.4, pad=0.15)
    return spans or [(0.0, duration)]


def _align(audio_16k, duration, syllables, chunks, texts, options) -> list[tuple[Syllable, tuple[float, float]]]:
    from humanslice.corpus.asr import QwenAligner

    aligner = QwenAligner(device=options.device)
    result: list[tuple[Syllable, tuple[float, float]]] = []
    try:
        if texts is not None:
            # ASR path: each chunk carries its own text.
            for (a, b), text in zip(chunks, texts):
                chunk_syllables = [s for s in text_to_syllables(text) if s is not None]
                result.extend(_align_piece(aligner, audio_16k, a, b, chunk_syllables, options.language))
        elif duration <= ALIGN_MAX_SEC:
            result = _align_piece(aligner, audio_16k, 0.0, duration, syllables, options.language)
        else:
            # Long song with known lyrics: distribute syllables over chunks proportionally to voiced time.
            result = _align_long(aligner, audio_16k, chunks, syllables, options.language)
    finally:
        aligner.close()
    return result


def _align_piece(aligner, audio_16k, a, b, syllables, language):
    if not syllables:
        return []
    clip = audio_16k[int(a * ANALYSIS_SR):int(b * ANALYSIS_SR)]
    tokens = aligner.align(clip, "".join(s.char for s in syllables), language)
    if len(tokens) != len(syllables):
        log.warning("aligner returned %d tokens for %d syllables", len(tokens), len(syllables))
    spans = [(a + t.start, a + t.end) for t in tokens[: len(syllables)]]
    spans += [(b, b)] * (len(syllables) - len(spans))
    return list(zip(syllables, repair_spans(spans, a, b)))


def repair_spans(spans: list[tuple[float, float]], lo: float, hi: float, min_dur: float = 0.06) -> list[tuple[float, float]]:
    """Make spans monotonic and give zero-length ones time borrowed from their neighbours.

    The aligner occasionally collapses a character (fast passages, melisma); dropping it
    would lose lyrics, so it gets a share of the gap or of the longer neighbour instead.
    """
    out = [list(span) for span in spans]
    for i, span in enumerate(out):
        previous_end = out[i - 1][1] if i else lo
        span[0] = max(span[0], previous_end)
        span[1] = max(span[1], span[0])
    for i, span in enumerate(out):
        if span[1] - span[0] >= min_dur:
            continue
        right_limit = out[i + 1][0] if i + 1 < len(out) else hi
        if right_limit - span[0] >= min_dur:
            span[1] = min(right_limit, span[0] + max(min_dur, (right_limit - span[0]) * 0.5))
        elif i > 0 and out[i - 1][1] - out[i - 1][0] > 2 * min_dur:
            # Take the tail of the previous syllable.
            cut = max(out[i - 1][0] + min_dur, span[1] - min_dur)
            out[i - 1][1] = cut
            span[0], span[1] = cut, max(span[1], cut + min_dur)
        elif i + 1 < len(out) and out[i + 1][1] - out[i + 1][0] > 2 * min_dur:
            span[1] = span[0] + min_dur
            out[i + 1][0] = max(out[i + 1][0], span[1])
    return [(span[0], span[1]) for span in out]


def _align_long(aligner, audio_16k, chunks, syllables, language):
    groups: list[list[tuple[float, float]]] = [[]]
    total = 0.0
    for a, b in chunks:
        if total + (b - a) > ALIGN_MAX_SEC and groups[-1]:
            groups.append([])
            total = 0.0
        groups[-1].append((a, b))
        total += b - a
    weights = np.array([sum(b - a for a, b in group) for group in groups])
    counts = np.round(np.cumsum(weights) / weights.sum() * len(syllables)).astype(int)
    result, previous = [], 0
    for group, upto in zip(groups, counts):
        result.extend(_align_piece(aligner, audio_16k, group[0][0], group[-1][1], syllables[previous:upto], language))
        previous = upto
    return result


def refine_with_mfa(audio_16k: np.ndarray, sung: list[SungSyllable], options: SongOptions, max_shift: float = 0.12) -> int:
    """Re-align syllable / consonant boundaries line by line with MFA (in place).

    Lines are pause-separated chunks; each takes the syllables whose aligner midpoint
    falls inside it. Lines whose MFA result doesn't match the syllable count keep the
    Qwen timing.
    """
    from humanslice.corpus.mfa import DEFAULT_ENV, AlignItem, MFAError, MFARunner, syllable_timings

    if options.phone_aligner == "auto" and not (DEFAULT_ENV / "conda-meta").exists():
        return 0
    lines = split_on_silence(audio_16k, ANALYSIS_SR, max_chunk=20.0, min_silence=0.3, pad=0.12)
    members: list[list[int]] = [[] for _ in lines]
    for index, syllable in enumerate(sung):
        middle = 0.5 * (syllable.start + syllable.end)
        for number, (a, b) in enumerate(lines):
            if a <= middle < b:
                members[number].append(index)
                break
    items = []
    for number, ((a, b), indices) in enumerate(zip(lines, members)):
        if indices:
            tokens = [f"{sung[i].pinyin}{sung[i].tone}" for i in indices]
            items.append(AlignItem(f"line{number:04d}", audio_16k[int(a * ANALYSIS_SR):int(b * ANALYSIS_SR)], tokens))
    try:
        grids = MFARunner(language=options.language).align(items)
    except MFAError as exc:
        log.warning("MFA refinement skipped: %s", exc)
        return 0
    refined = 0
    for number, ((a, _), indices) in enumerate(zip(lines, members)):
        grid = grids.get(f"line{number:04d}")
        if grid is None or not indices:
            continue
        timings = syllable_timings(grid)
        if len(timings) != len(indices):
            continue
        for index, timing in zip(indices, timings):
            if timing is None:
                continue
            syllable = sung[index]
            start, end = a + timing.start, a + timing.end
            # MFA (trained on speech) occasionally derails on long sung vowels; only trust it
            # where it roughly agrees with the character-level aligner.
            if abs(start - syllable.start) > max_shift or abs(end - syllable.end) > 1.25 * max_shift:
                continue
            syllable.start, syllable.end = start, end
            syllable.vowel_start = a + (timing.cv_boundary if syllable.initial else timing.start)
            refined += 1
    return refined


def _refine_ends(sung: list[SungSyllable], f0: np.ndarray, max_unvoiced_gap: float = 0.06) -> list[SungSyllable]:
    """Make syllables non-overlapping and let each one last as long as its voicing does.

    Aligners often end a held note early; the syllable is extended along the voiced run
    that contains its end (tolerating short unvoiced dips) up to the next syllable.
    """
    gap_frames = int(round(max_unvoiced_gap / FRAME_PERIOD))
    voiced = f0 > 0
    for index, current in enumerate(sung):
        limit = sung[index + 1].start if index + 1 < len(sung) else f0.size * FRAME_PERIOD
        current.end = min(current.end, limit)
        frame = int(current.end / FRAME_PERIOD)
        last_voiced, unvoiced_run = frame, 0
        while frame < min(int(limit / FRAME_PERIOD), voiced.size):
            if voiced[frame]:
                last_voiced, unvoiced_run = frame, 0
            else:
                unvoiced_run += 1
                if unvoiced_run > gap_frames:
                    break
            frame += 1
        current.end = max(current.end, min(limit, (last_voiced + 1) * FRAME_PERIOD))
    for syllable in sung:
        syllable.vowel_start = min(max(syllable.vowel_start, syllable.start), syllable.end - 0.02)
    return sung
