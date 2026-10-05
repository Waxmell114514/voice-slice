"""One-shot jinriki pipeline shared by the CLI (``humanslice make``) and the GUI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from humanslice.render.engine import RenderOptions


@dataclass(slots=True)
class MakeOptions:
    material: list[str]
    vocal: str
    output: str
    lyrics: str | None = None  # text, or path to .txt / .lrc
    bank: str | None = None  # reuse / store the unit bank here (default <output>/bank)
    name: str | None = None  # voicebank name
    separate: bool = False  # the song is a full mix
    separate_material: bool = False
    aligner: str = "auto"
    tempo: float = 120.0
    vocal_gain_db: float = 0.0
    device: str | None = None
    render: RenderOptions = field(default_factory=RenderOptions)


def read_lyrics(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.suffix.lower() in (".txt", ".lrc", ".lab") and path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        return re.sub(r"\[[^\]]*\]", "", text)  # strip LRC timestamps
    return value


def run_make(options: MakeOptions, progress: Callable[[str], None] = print) -> dict[str, Path]:
    from humanslice.corpus.build import BuildOptions, build_bank
    from humanslice.export.midi import write_midi
    from humanslice.export.utau import export_voicebank
    from humanslice.export.ustx import write_ustx
    from humanslice.render.engine import render, write_outputs
    from humanslice.song.analyze import SongOptions, analyze_song

    out = Path(options.output)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    progress("== 1/4 素材 -> 单元库")
    bank = build_bank(
        options.material,
        Path(options.bank) if options.bank else out / "bank",
        BuildOptions(aligner=options.aligner, device=options.device, separate=options.separate_material),
        progress=progress,
    )
    name = options.name or "HumanSlice"
    results["voicebank"] = export_voicebank(bank, out / "voicebank" / name, name=name)

    progress("== 2/4 歌曲分析")
    vocal, instrumental = Path(options.vocal), None
    if options.separate:
        from humanslice.common.separate import separate

        progress("[song] separating vocals / accompaniment")
        vocal, instrumental = separate(options.vocal)
    score = analyze_song(
        vocal,
        SongOptions(lyrics=read_lyrics(options.lyrics), device=options.device, tempo=options.tempo),
        progress=progress,
    )
    score.save(out / "score.json")

    progress("== 3/4 渲染")
    result = render(bank, score, options.render, progress=progress)
    results["vocal"] = write_outputs(result, out / "jinriki.wav")
    results["report"] = results["vocal"].with_suffix(".report.json")
    if instrumental is not None and instrumental.exists():
        from humanslice.common.separate import mix

        results["mix"] = mix(results["vocal"], instrumental, out / "jinriki_mix.wav", options.vocal_gain_db)

    progress("== 4/4 导出工程")
    transpose = result.selection.transpose
    results["ustx"] = write_ustx(score, out / "project.ustx", singer=name, transpose=transpose)
    results["midi"] = write_midi(score, out / "melody.mid", transpose=transpose)
    summary = result.report["summary"]
    progress(
        f"完成：移调 {transpose:+d}，{summary['syllables']} 字，"
        f"同音节命中 {summary['exact_ratio'] * 100:.0f}%，素材原句连接 {summary['contiguous_joins']} 处"
    )
    return results
