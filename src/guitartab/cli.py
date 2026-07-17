"""guitartab CLI — transcribe / separate / eval サブコマンド。

    python -m guitartab transcribe --url <YouTube URL>
    python -m guitartab separate   --url <YouTube URL> | --input <audio>
    python -m guitartab eval       [--eval-data eval_data] [--engine basicpitch]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from guitartab.eval.benchmark import discover_items, format_table, run_benchmark
from guitartab.eval.metrics import DEFAULT_ONSET_TOLERANCE_SEC
from guitartab.pipeline import (
    DEFAULT_WORK_ROOT,
    run_transcribe_pipeline,
    stage_download,
    stage_separate,
)
from guitartab.transcribe.base import TranscriberEngine
from guitartab.transcribe.basicpitch import BasicPitchEngine

ENGINE_NAMES = ["basicpitch", "muscriptor"]


def build_engine(name: str, args: argparse.Namespace) -> TranscriberEngine:
    if name == "basicpitch":
        return BasicPitchEngine(venv_python=args.basicpitch_python)
    if name == "muscriptor":
        raise SystemExit(
            "muscriptor engine is not implemented yet (M0: pending Apple Silicon "
            "verification, see docs/DESIGN.md)"
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


def cmd_transcribe(args: argparse.Namespace) -> int:
    engine = build_engine(args.engine, args)
    notes_path = run_transcribe_pipeline(
        args.url,
        engine,
        work_root=args.work,
        separate=not args.no_separate,
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


def cmd_eval(args: argparse.Namespace) -> int:
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

    p_ev = sub.add_parser("eval", help="eval_data/ のベンチセットを一括評価")
    p_ev.add_argument("--eval-data", type=Path, default=Path("eval_data"))
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
    _add_common_engine_args(p_ev)
    p_ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    if args.command == "eval":
        if args.engine is None:
            args.engine = ["basicpitch"]
        if args.out is not None:
            out_r, ev_r = args.out.resolve(), args.eval_data.resolve()
            if out_r == ev_r or ev_r in out_r.parents:
                raise SystemExit("--out must not be inside eval_data/ (frozen GT area)")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
