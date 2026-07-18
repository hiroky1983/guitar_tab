"""guitartab CLI — transcribe / separate / tab / quantize / eval サブコマンド。

    python -m guitartab transcribe --url <YouTube URL>
    python -m guitartab separate   --url <YouTube URL> | --input <audio>
    python -m guitartab tab        <notes.json> [--out-dir DIR]
    python -m guitartab quantize   <notes.json> [--audio FILE] [--out FILE]
    python -m guitartab midi       <notes.json> [--rhythm rhythm.json] [--out FILE]
    python -m guitartab musicxml   <tab.json> [--rhythm rhythm.json] [--out FILE]
    python -m guitartab eval       [--eval-data eval_data] [--engine basicpitch]
    python -m guitartab eval --rhythm [--eval-data eval_data/guitarset]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from guitartab.eval.benchmark import discover_items, format_table, run_benchmark
from guitartab.eval.metrics import DEFAULT_ONSET_TOLERANCE_SEC
from guitartab.pipeline import (
    DEFAULT_WORK_ROOT,
    MIDI_FILENAME,
    MUSICXML_FILENAME,
    RHYTHM_FILENAME,
    TAB_FILENAME,
    TAB_TEXT_FILENAME,
    run_transcribe_pipeline,
    stage_download,
    stage_midi,
    stage_musicxml,
    stage_quantize,
    stage_separate,
    stage_tab,
)
from guitartab.tab.render_ascii import DEFAULT_LINE_WIDTH, DEFAULT_TIME_STEP_SEC
from guitartab.tab.render_midi import DEFAULT_TEMPO_BPM
from guitartab.transcribe.base import TranscriberEngine
from guitartab.transcribe.basicpitch import BasicPitchEngine
from guitartab.transcribe.muscriptor import MuScriptorEngine
from guitartab.transcribe.yourmt3 import YourMT3Engine

ENGINE_NAMES = ["basicpitch", "yourmt3", "muscriptor"]


def build_engine(name: str, args: argparse.Namespace) -> TranscriberEngine:
    if name == "basicpitch":
        return BasicPitchEngine(
            venv_python=args.basicpitch_python,
            onset_threshold=args.bp_onset_threshold,
            frame_threshold=args.bp_frame_threshold,
            minimum_note_length=args.bp_minimum_note_length,
            minimum_frequency=args.bp_minimum_frequency,
            maximum_frequency=args.bp_maximum_frequency,
            melodia_trick=False if args.bp_no_melodia_trick else None,
        )
    if name == "yourmt3":
        return YourMT3Engine(
            venv_python=args.yourmt3_python,
            home=args.yourmt3_home,
            device=args.yourmt3_device,
        )
    if name == "muscriptor":
        instruments = None
        if args.ms_instruments is not None:
            instruments = [s.strip() for s in args.ms_instruments.split(",") if s.strip()]
            if not instruments:
                raise SystemExit(
                    "--ms-instruments must contain at least one instrument name"
                )
        return MuScriptorEngine(
            venv_python=args.muscriptor_python,
            instruments=instruments,
            cfg_coef=args.ms_cfg_coef,
            batch_size=args.ms_batch_size,
            device=args.ms_device,
        )
    raise SystemExit(f"unknown engine: {name} (available: {', '.join(ENGINE_NAMES)})")


def _add_common_engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--basicpitch-python",
        type=Path,
        default=None,
        metavar="PYTHON",
        help="basic-pitch 専用 venv の python パス "
        "(default: $GUITARTAB_BASICPITCH_PYTHON or .venv-basicpitch/bin/python)",
    )
    ym = p.add_argument_group("yourmt3 params")
    ym.add_argument(
        "--yourmt3-python",
        type=Path,
        default=None,
        metavar="PYTHON",
        help="YourMT3 専用 venv の python パス "
        "(default: $GUITARTAB_YOURMT3_PYTHON or .venv-yourmt3/bin/python)",
    )
    ym.add_argument(
        "--yourmt3-home",
        type=Path,
        default=None,
        metavar="DIR",
        help="YourMT3 コード+チェックポイントのディレクトリ "
        "(default: $GUITARTAB_YOURMT3_HOME or third_party/yourmt3)",
    )
    ym.add_argument(
        "--yourmt3-device",
        default=None,
        metavar="DEV",
        help="推論デバイス cpu|mps (default: $GUITARTAB_YOURMT3_DEVICE or cpu)",
    )
    ms = p.add_argument_group("muscriptor params")
    ms.add_argument(
        "--muscriptor-python",
        type=Path,
        default=None,
        metavar="PYTHON",
        help="MuScriptor 専用 venv の python パス "
        "(default: $GUITARTAB_MUSCRIPTOR_PYTHON or .venv-muscriptor/bin/python)",
    )
    ms.add_argument(
        "--ms-instruments",
        default=None,
        metavar="LIST",
        help="生成条件付けプロンプト（カンマ区切り、"
        "default: acoustic_guitar,distorted_electric_guitar）",
    )
    ms.add_argument(
        "--ms-cfg-coef",
        type=float,
        default=None,
        metavar="C",
        help="classifier-free guidance 係数 (default: 1.5。1.6 以上は縮退暴走の実測あり)",
    )
    ms.add_argument("--ms-batch-size", type=int, default=None, metavar="N")
    ms.add_argument(
        "--ms-device",
        default=None,
        metavar="DEV",
        help="推論デバイス mps|cpu (default: $GUITARTAB_MUSCRIPTOR_DEVICE or mps)",
    )
    # basic_pitch.inference.predict() のネイティブ推論パラメータ。
    # 未指定（None）は predict() のデフォルト = 従来動作。
    bp = p.add_argument_group("basic-pitch predict() params")
    bp.add_argument("--bp-onset-threshold", type=float, default=None, metavar="P")
    bp.add_argument("--bp-frame-threshold", type=float, default=None, metavar="P")
    bp.add_argument(
        "--bp-minimum-note-length", type=float, default=None, metavar="MS"
    )
    bp.add_argument("--bp-minimum-frequency", type=float, default=None, metavar="HZ")
    bp.add_argument("--bp-maximum-frequency", type=float, default=None, metavar="HZ")
    bp.add_argument(
        "--bp-no-melodia-trick",
        action="store_true",
        help="melodia trick を無効化する（default: 有効）",
    )


def cmd_transcribe(args: argparse.Namespace) -> int:
    engine = build_engine(args.engine, args)
    notes_path = run_transcribe_pipeline(
        args.url,
        engine,
        work_root=args.work,
        separate=not args.no_separate,
        quantize=not args.no_quantize,
        force=args.force,
    )
    print(notes_path)
    return 0


def cmd_separate(args: argparse.Namespace) -> int:
    if args.input is not None:
        source = Path(args.input)
        if not source.exists():
            raise SystemExit(f"input not found: {source}")
        stems_dir = args.work / source.stem / "stems"
        from guitartab.separate import separate_guitar

        guitar = separate_guitar(source, stems_dir, force=args.force)
    else:
        source = stage_download(args.url, args.work, force=args.force)
        guitar = stage_separate(source, force=args.force)
    print(guitar)
    return 0


def cmd_tab(args: argparse.Namespace) -> int:
    notes_path = args.notes
    if not notes_path.exists():
        raise SystemExit(f"notes.json not found: {notes_path}")
    out_dir = args.out_dir if args.out_dir is not None else notes_path.parent
    if "eval_data" in out_dir.resolve().parts:
        raise SystemExit(
            "output directory is inside eval_data/ (frozen GT area); "
            "use --out-dir to write elsewhere"
        )
    tab_path = out_dir / TAB_FILENAME
    tab_txt_path = out_dir / TAB_TEXT_FILENAME
    stage_tab(
        notes_path,
        tab_path,
        tab_txt_path,
        time_step_sec=args.time_step,
        line_width=args.width,
        force=args.force,
    )
    print(tab_path)
    print(tab_txt_path)
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    notes_path = args.notes
    if not notes_path.exists():
        raise SystemExit(f"notes.json not found: {notes_path}")
    out_path = args.out if args.out is not None else notes_path.parent / RHYTHM_FILENAME
    if "eval_data" in out_path.resolve().parts:
        raise SystemExit(
            "output path is inside eval_data/ (frozen GT area); "
            "use --out to write elsewhere"
        )
    audio_path = args.audio
    if audio_path is not None and not audio_path.exists():
        raise SystemExit(f"audio not found: {audio_path}")
    stage_quantize(notes_path, out_path, audio_path=audio_path, force=args.force)
    print(out_path)
    return 0


def cmd_midi(args: argparse.Namespace) -> int:
    notes_path = args.notes
    if not notes_path.exists():
        raise SystemExit(f"notes.json not found: {notes_path}")
    midi_path = args.out if args.out is not None else notes_path.parent / MIDI_FILENAME
    if "eval_data" in midi_path.resolve().parts:
        raise SystemExit(
            "output path is inside eval_data/ (frozen GT area); "
            "use --out to write elsewhere"
        )
    stage_midi(
        notes_path,
        midi_path,
        tempo_bpm=args.tempo,
        rhythm_path=args.rhythm,
        force=args.force,
    )
    print(midi_path)
    return 0


def cmd_musicxml(args: argparse.Namespace) -> int:
    tab_path = args.tab
    if not tab_path.exists():
        raise SystemExit(f"tab.json not found: {tab_path}")
    out_path = args.out if args.out is not None else tab_path.parent / MUSICXML_FILENAME
    if "eval_data" in out_path.resolve().parts:
        raise SystemExit(
            "output path is inside eval_data/ (frozen GT area); "
            "use --out to write elsewhere"
        )
    stage_musicxml(tab_path, out_path, rhythm_path=args.rhythm, force=args.force)
    print(out_path)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    if args.rhythm:
        return _cmd_eval_rhythm(args)
    items = discover_items(args.eval_data)
    if not items:
        print(
            f"no benchmark items found under {args.eval_data} "
            "(expected <dir>/ with audio + gt.json|gt.jams|ground_truth.json; "
            "note: eval_data/gt/ has no audio so it is not a bench item)",
            file=sys.stderr,
        )
        return 1
    engines = [build_engine(name, args) for name in args.engine]
    result = run_benchmark(
        engines,
        items,
        onset_tolerance_sec=args.onset_tolerance,
        out_dir=args.out,
    )
    print(format_table(result))
    had_errors = any(errs for errs in result.errors.values())
    return 1 if had_errors else 0


def _cmd_eval_rhythm(args: argparse.Namespace) -> int:
    """リズム量子化ベンチ（M4）。engine 指定時は合成ベンチを E2E モードで評価。"""
    from guitartab.eval.rhythm_benchmark import (
        discover_rhythm_items,
        format_rhythm_table,
        run_rhythm_benchmark,
    )
    if args.rhythm_estimator == "beatthis":
        from guitartab.rhythm.beatthis import BeatThisTempoEstimator

        estimator = BeatThisTempoEstimator()
    else:
        from guitartab.rhythm.estimate import LibrosaConstantTempoEstimator

        estimator = LibrosaConstantTempoEstimator()

    items = discover_rhythm_items(args.eval_data)
    if not items:
        print(
            f"no rhythm benchmark items found under {args.eval_data} "
            "(expected GuitarSet layout or rhythm_synth clips)",
            file=sys.stderr,
        )
        return 1
    engine = None
    if args.engine:
        engine = build_engine(args.engine[0], args)
    result = run_rhythm_benchmark(
        estimator,
        items,
        use_audio=not args.rhythm_no_audio,
        engine=engine,
    )
    print(format_rhythm_table(result))
    return 1 if result.errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="guitartab",
        description="YouTube URL からギター TAB 譜を生成するパイプライン (v2)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_tr = sub.add_parser("transcribe", help="download → separate → transcribe")
    p_tr.add_argument("--url", required=True, help="YouTube URL")
    p_tr.add_argument("--work", type=Path, default=DEFAULT_WORK_ROOT)
    p_tr.add_argument("--engine", default="basicpitch", choices=ENGINE_NAMES)
    p_tr.add_argument(
        "--no-separate",
        action="store_true",
        help="Demucs 分離をスキップして source.wav を直接転写する",
    )
    p_tr.add_argument(
        "--no-quantize",
        action="store_true",
        help="quantize（テンポ推定+格子スナップ）をスキップして固定 120BPM 近似で出力する",
    )
    p_tr.add_argument("--force", action="store_true", help="キャッシュを無視して再実行")
    _add_common_engine_args(p_tr)
    p_tr.set_defaults(func=cmd_transcribe)

    p_sep = sub.add_parser("separate", help="Demucs でギターステムを抽出")
    src = p_sep.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL")
    src.add_argument("--input", type=Path, help="ローカル音声ファイル")
    p_sep.add_argument("--work", type=Path, default=DEFAULT_WORK_ROOT)
    p_sep.add_argument("--force", action="store_true")
    p_sep.set_defaults(func=cmd_separate)

    p_tab = sub.add_parser("tab", help="notes.json → 運指割当(tab.json) + ASCII tab(tab.txt)")
    p_tab.add_argument("notes", type=Path, help="入力 notes.json")
    p_tab.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="出力先ディレクトリ (default: notes.json と同じディレクトリ)",
    )
    p_tab.add_argument(
        "--time-step",
        type=float,
        default=DEFAULT_TIME_STEP_SEC,
        metavar="SEC",
        help="ASCII tab の 1 カラムあたりの秒数",
    )
    p_tab.add_argument(
        "--width", type=int, default=DEFAULT_LINE_WIDTH, help="ASCII tab の折り返し桁数"
    )
    p_tab.add_argument("--force", action="store_true", help="キャッシュを無視して再実行")
    p_tab.set_defaults(func=cmd_tab)

    p_q = sub.add_parser(
        "quantize",
        help="notes.json → テンポ推定+格子スナップ (rhythm.json、M4a)",
    )
    p_q.add_argument("notes", type=Path, help="入力 notes.json")
    p_q.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="拍推定に使う音声（ギターステム等。省略時はノート onset のみで推定）",
    )
    p_q.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"出力パス (default: notes.json と同じディレクトリの {RHYTHM_FILENAME})",
    )
    p_q.add_argument("--force", action="store_true", help="キャッシュを無視して再実行")
    p_q.set_defaults(func=cmd_quantize)

    p_mid = sub.add_parser("midi", help="notes.json → MIDI (output.mid)")
    p_mid.add_argument("notes", type=Path, help="入力 notes.json")
    p_mid.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"出力 MIDI パス (default: notes.json と同じディレクトリの {MIDI_FILENAME})",
    )
    p_mid.add_argument(
        "--tempo",
        type=float,
        default=DEFAULT_TEMPO_BPM,
        metavar="BPM",
        help="固定テンポ (default: %(default)s BPM。量子化は行わない)",
    )
    p_mid.add_argument(
        "--rhythm",
        type=Path,
        default=None,
        metavar="FILE",
        help="rhythm.json を使って実テンポ・量子化 tick でレンダリングする",
    )
    p_mid.add_argument("--force", action="store_true", help="キャッシュを無視して再実行")
    p_mid.set_defaults(func=cmd_midi)

    p_mxl = sub.add_parser(
        "musicxml",
        help="tab.json → MusicXML (output.musicxml, MuseScore / Guitar Pro 連携用)",
    )
    p_mxl.add_argument("tab", type=Path, help="入力 tab.json")
    p_mxl.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"出力 MusicXML パス (default: tab.json と同じディレクトリの {MUSICXML_FILENAME})",
    )
    p_mxl.add_argument(
        "--rhythm",
        type=Path,
        default=None,
        metavar="FILE",
        help="rhythm.json を使って実テンポ・量子化 tick でレンダリングする",
    )
    p_mxl.add_argument("--force", action="store_true", help="キャッシュを無視して再実行")
    p_mxl.set_defaults(func=cmd_musicxml)

    p_ev = sub.add_parser("eval", help="eval_data/ のベンチセットを一括評価")
    # デフォルトは dev セットのみ。holdout（1回限りの最終判定用）を日常のチューニングで
    # 汚染しないため、eval_data/ 全体をデフォルトにしない。
    p_ev.add_argument("--eval-data", type=Path, default=Path("eval_data/guitarset"))
    p_ev.add_argument(
        "--engine",
        action="append",
        choices=ENGINE_NAMES,
        help="評価するエンジン（複数指定可、default: basicpitch）",
    )
    p_ev.add_argument(
        "--onset-tolerance",
        type=float,
        default=DEFAULT_ONSET_TOLERANCE_SEC,
        metavar="SEC",
    )
    p_ev.add_argument(
        "--out",
        type=Path,
        default=None,
        help="推定 notes.json を残すディレクトリ（eval_data 配下は指定禁止）",
    )
    p_ev.add_argument(
        "--rhythm",
        action="store_true",
        help="リズム量子化ベンチ（M4）を実行する。GT ノート onset + 音声で"
        "テンポ・拍を評価（合成ベンチでは GPA も）。--engine 指定時は"
        "合成ベンチをフル転写経由（E2E）で評価する",
    )
    p_ev.add_argument(
        "--rhythm-estimator",
        choices=["librosa", "beatthis"],
        default="librosa",
        help="--rhythm のテンポ推定器（librosa = M4a 候補 A / beatthis = M4b 候補 B。"
        "beatthis は専用 venv .venv-beatthis が必要）",
    )
    p_ev.add_argument(
        "--rhythm-no-audio",
        action="store_true",
        help="--rhythm で音声を使わずノート onset のみで推定する（フォールバック経路の評価）",
    )
    _add_common_engine_args(p_ev)
    p_ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if args.command == "eval":
        if args.engine is None and not args.rhythm:
            args.engine = ["basicpitch"]
        if args.out is not None:
            out_r, ev_r = args.out.resolve(), args.eval_data.resolve()
            if out_r == ev_r or ev_r in out_r.parents:
                raise SystemExit("--out must not be inside eval_data/ (frozen GT area)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
