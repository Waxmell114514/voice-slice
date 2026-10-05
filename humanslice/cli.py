"""Command line interface.

    humanslice bank build MATERIAL... -o BANK          speech material -> unit bank
    humanslice bank utau BANK -o VOICEBANK_DIR          unit bank -> OpenUtau voicebank
    humanslice song analyze VOCAL -o SCORE.json         acapella -> score (+ --ustx / --midi)
    humanslice render BANK SCORE.json -o OUT.wav        score + bank -> jinriki vocal
    humanslice make --material DIR --vocal VOCAL -o OUT one-shot pipeline
"""

from __future__ import annotations

import argparse
import logging
import sys


def cmd_bank_build(args: argparse.Namespace) -> int:
    from humanslice.corpus.build import BuildOptions, build_bank

    options = BuildOptions(aligner=args.aligner, use_labels=not args.no_labels, device=args.device, separate=args.separate)
    build_bank(args.material, args.output, options, force=args.force, refeature=args.refeature)
    return 0


def cmd_bank_utau(args: argparse.Namespace) -> int:
    from humanslice.corpus.units import Bank
    from humanslice.export.utau import export_voicebank

    bank = Bank.open(args.bank)
    out = export_voicebank(bank, args.output, name=args.name)
    print(f"Voicebank written to {out}")
    return 0


def cmd_song_analyze(args: argparse.Namespace) -> int:
    from humanslice.pipeline import read_lyrics
    from humanslice.song.analyze import SongOptions, analyze_song

    options = SongOptions(lyrics=read_lyrics(args.lyrics), device=args.device, tempo=args.tempo)
    vocal = args.vocal
    if args.separate:
        from humanslice.common.separate import separate

        vocal, instrumental = separate(args.vocal)
        print(f"Separated: {vocal}  /  {instrumental}")
    score = analyze_song(vocal, options)
    score.save(args.output)
    print(f"Score written to {args.output}")
    _export_score(score, args)
    return 0


def _export_score(score, args: argparse.Namespace) -> None:
    if getattr(args, "ustx", None):
        from humanslice.export.ustx import write_ustx

        write_ustx(score, args.ustx, singer=args.singer or "HumanSlice", transpose=args.transpose or 0)
        print(f"USTX written to {args.ustx}")
    if getattr(args, "midi", None):
        from humanslice.export.midi import write_midi

        write_midi(score, args.midi)
        print(f"MIDI written to {args.midi}")


def _render_options(args: argparse.Namespace):
    from humanslice.render.engine import RenderOptions
    from humanslice.render.select import SelectOptions
    from humanslice.render.world import WorldOptions

    return RenderOptions(
        select=SelectOptions(
            transpose=args.transpose,
            octave_only=not args.any_key,
            join_weight=args.join_weight,
        ),
        world=WorldOptions(pitch_follow=args.pitch_follow, level_follow=args.level_follow),
        max_stretch=args.max_stretch,
        backend=args.backend,
    )


def cmd_render(args: argparse.Namespace) -> int:
    from humanslice.corpus.units import Bank
    from humanslice.render.engine import render, write_outputs
    from humanslice.song.score import Score

    result = render(Bank.open(args.bank), Score.load(args.score), _render_options(args))
    out = write_outputs(result, args.output)
    print(f"Rendered {out} ({result.report['summary']})")
    return 0


def cmd_make(args: argparse.Namespace) -> int:
    from humanslice.pipeline import MakeOptions, run_make

    results = run_make(
        MakeOptions(
            material=args.material,
            vocal=args.vocal,
            output=args.output,
            lyrics=args.lyrics,
            bank=args.bank,
            name=args.name,
            separate=args.separate,
            separate_material=args.separate_material,
            aligner=args.aligner,
            tempo=args.tempo,
            vocal_gain_db=args.vocal_gain,
            device=args.device,
            render=_render_options(args),
        )
    )
    for key, path in results.items():
        print(f"{key:10} {path}")
    return 0


def _add_render_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--transpose", type=int, default=None, help="semitones (default: automatic octave shift)")
    parser.add_argument("--any-key", action="store_true", help="allow non-octave automatic transposition")
    parser.add_argument("--join-weight", type=float, default=1.0, help="preference for contiguous original phrases")
    parser.add_argument("--pitch-follow", type=float, default=1.0, help="1 = follow target F0 exactly, 0 = flat notes")
    parser.add_argument("--level-follow", type=float, default=0.7, help="how much target dynamics to impose")
    parser.add_argument("--max-stretch", type=float, default=2.5, help="beyond this the vowel nucleus is looped")
    parser.add_argument("--backend", default="nsf", choices=["nsf", "world"],
                        help="nsf: PC-NSF-HiFiGAN (CC BY-NC-SA weights, best quality); world: WORLD vocoder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="humanslice", description="Automatic jinriki (人力) vocal maker")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda / cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    bank = sub.add_parser("bank", help="speech material unit bank").add_subparsers(dest="bank_command", required=True)
    build = bank.add_parser("build", help="build / extend a unit bank from speech material")
    build.add_argument("material", nargs="+", help="audio / video files or folders")
    build.add_argument("-o", "--output", required=True, help="bank directory")
    build.add_argument("--aligner", default="auto", choices=["auto", "mfa", "qwen"])
    build.add_argument("--no-labels", action="store_true", help="ignore sidecar .lab/.txt transcripts")
    build.add_argument("--separate", action="store_true", help="remove BGM / noise with a vocal separation model")
    build.add_argument("--force", action="store_true", help="re-run segmentation and alignment")
    build.add_argument("--refeature", action="store_true", help="recompute pitch / nucleus features only")
    build.set_defaults(func=cmd_bank_build)

    utau = bank.add_parser("utau", help="export an OpenUtau voicebank")
    utau.add_argument("bank")
    utau.add_argument("-o", "--output", required=True)
    utau.add_argument("--name", default=None)
    utau.set_defaults(func=cmd_bank_utau)

    song = sub.add_parser("song", help="target vocal analysis").add_subparsers(dest="song_command", required=True)
    analyze = song.add_parser("analyze", help="acapella -> score")
    analyze.add_argument("vocal")
    analyze.add_argument("-o", "--output", required=True, help="score .json")
    analyze.add_argument("--lyrics", default=None, help="lyrics text or .txt/.lrc file (recommended)")
    analyze.add_argument("--tempo", type=float, default=120.0, help="BPM used for USTX / MIDI tick grid")
    analyze.add_argument("--ustx", default=None, help="also write an OpenUtau project")
    analyze.add_argument("--singer", default=None, help="voicebank name referenced by the USTX")
    analyze.add_argument("--transpose", type=int, default=0, help="transpose notes in the USTX / MIDI")
    analyze.add_argument("--midi", default=None, help="also write a MIDI file with lyrics")
    analyze.add_argument("--separate", action="store_true", help="input is a full mix: extract the vocal first")
    analyze.set_defaults(func=cmd_song_analyze)

    rend = sub.add_parser("render", help="render a score with a unit bank")
    rend.add_argument("bank")
    rend.add_argument("score")
    rend.add_argument("-o", "--output", required=True)
    _add_render_flags(rend)
    rend.set_defaults(func=cmd_render)

    make = sub.add_parser("make", help="material + acapella -> jinriki vocal, voicebank and USTX")
    make.add_argument("--material", nargs="+", required=True)
    make.add_argument("--vocal", required=True)
    make.add_argument("--lyrics", default=None)
    make.add_argument("-o", "--output", required=True)
    make.add_argument("--bank", default=None, help="reuse / store the unit bank here")
    make.add_argument("--name", default=None, help="voicebank name")
    make.add_argument("--aligner", default="auto", choices=["auto", "mfa", "qwen"])
    make.add_argument("--tempo", type=float, default=120.0)
    make.add_argument("--separate", action="store_true", help="--vocal is a full song mix: split vocals/accompaniment")
    make.add_argument("--separate-material", action="store_true", help="remove BGM from the speech material")
    make.add_argument("--vocal-gain", type=float, default=0.0, help="dB applied to the vocal in the final mix")
    _add_render_flags(make)
    make.set_defaults(func=cmd_make)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
