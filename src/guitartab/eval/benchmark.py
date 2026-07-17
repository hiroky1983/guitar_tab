"""ベンチセット一括評価とエンジン比較表の出力。

ベンチセットの配置規約（2種類に対応）:

1. アイテムディレクトリ形式 — 各ディレクトリが1アイテム:

    eval_data/items/<item_id>/
      audio.wav             # 音声（audio.* を優先。なければ唯一の音声ファイル）
      gt.jams | gt.json | ground_truth.json   # ground truth

2. GuitarSet 形式 — annotations/ と audio/ が並列で、ファイル名 prefix で対応:

    eval_data/guitarset/
      annotations/00_BN3-119-G_solo.jams
      audio/00_BN3-119-G_solo_mic.wav      # jams の stem で前方一致

eval_data/gt/ は凍結 GT 置き場（音声なし → ベンチ対象外）。

eval_data/gt/ は凍結・読み取り専用。本モジュールは eval_data/ 配下に一切
書き込まない（推定結果のダンプは out_dir 指定時のみ、eval_data 外に書く）。
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from guitartab.eval.loaders import load_ground_truth
from guitartab.eval.metrics import (
    DEFAULT_ONSET_TOLERANCE_SEC,
    NoteMetrics,
    evaluate_notes,
)
from guitartab.transcribe.base import TranscriberEngine, save_notes

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".ogg"}
GT_JSON_NAMES = {"gt.json", "ground_truth.json", "notes_gt.json"}


@dataclass(frozen=True)
class BenchItem:
    item_id: str
    audio_path: Path
    gt_path: Path


@dataclass
class BenchmarkResult:
    onset_tolerance_sec: float
    # {engine_name: {item_id: NoteMetrics}}
    metrics: dict[str, dict[str, NoteMetrics]] = field(default_factory=dict)
    # {engine_name: {item_id: error_message}}
    errors: dict[str, dict[str, str]] = field(default_factory=dict)


def _find_gt(directory: Path) -> Path | None:
    jams = sorted(directory.glob("*.jams"))
    if jams:
        return jams[0]
    for name in sorted(GT_JSON_NAMES):
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def _find_audio(directory: Path) -> Path | None:
    audio_files = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    if not audio_files:
        return None
    preferred = [p for p in audio_files if p.stem == "audio"]
    return preferred[0] if preferred else audio_files[0]


def _discover_guitarset_layout(directory: Path, eval_root: Path) -> list[BenchItem]:
    """annotations/*.jams と audio/ をファイル名 prefix で対応付ける。"""
    ann_dir = directory / "annotations"
    audio_dir = directory / "audio"
    if not (ann_dir.is_dir() and audio_dir.is_dir()):
        return []
    audio_files = sorted(
        p for p in audio_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )
    items: list[BenchItem] = []
    for jams in sorted(ann_dir.glob("*.jams")):
        matches = [a for a in audio_files if a.stem.startswith(jams.stem)]
        if not matches:
            continue  # 音声未取得のアノテーションはスキップ
        items.append(
            BenchItem(
                item_id=f"{directory.relative_to(eval_root)}/{jams.stem}",
                audio_path=matches[0],
                gt_path=jams,
            )
        )
    return items


def discover_items(eval_root: Path) -> list[BenchItem]:
    """eval_root 以下を再帰的に走査し、音声+GT のペアを集める（配置規約は冒頭参照）。"""
    eval_root = Path(eval_root)
    items: list[BenchItem] = []
    if not eval_root.exists():
        return items
    for directory in sorted(d for d in [eval_root, *eval_root.rglob("*")] if d.is_dir()):
        guitarset_items = _discover_guitarset_layout(directory, eval_root)
        if guitarset_items:
            items.extend(guitarset_items)
            continue
        gt = _find_gt(directory)
        audio = _find_audio(directory)
        if gt is not None and audio is not None:
            items.append(
                BenchItem(
                    item_id=str(directory.relative_to(eval_root)),
                    audio_path=audio,
                    gt_path=gt,
                )
            )
    return items


def run_benchmark(
    engines: list[TranscriberEngine],
    items: list[BenchItem],
    *,
    onset_tolerance_sec: float = DEFAULT_ONSET_TOLERANCE_SEC,
    out_dir: Path | None = None,
) -> BenchmarkResult:
    """全アイテム × 全エンジンを評価する。

    out_dir を指定すると推定 notes.json を out_dir/{engine}/{item_id}.json に残す
    （eval_data/ 配下を out_dir に指定しないこと）。
    """
    result = BenchmarkResult(onset_tolerance_sec=onset_tolerance_sec)
    for engine in engines:
        result.metrics[engine.name] = {}
        result.errors[engine.name] = {}
        for item in items:
            try:
                ref_notes = load_ground_truth(item.gt_path)
                est_notes = engine.transcribe(item.audio_path)
                if out_dir is not None:
                    dump = Path(out_dir) / engine.name / f"{item.item_id}.json"
                    save_notes(est_notes, dump)
                result.metrics[engine.name][item.item_id] = evaluate_notes(
                    est_notes, ref_notes, onset_tolerance_sec=onset_tolerance_sec
                )
            except Exception as e:  # 1アイテムの失敗で全体を止めない
                traceback.print_exc(file=sys.stderr)
                result.errors[engine.name][item.item_id] = f"{type(e).__name__}: {e}"
    return result


def format_table(result: BenchmarkResult) -> str:
    """エンジン別比較表（テキスト）を返す。"""
    lines: list[str] = []
    lines.append(
        f"note-level P/R/F1  (onset tolerance = "
        f"{result.onset_tolerance_sec * 1000:.0f}ms, pitch = same semitone)"
    )
    header = (
        f"{'engine':<14} {'item':<28} {'P':>7} {'R':>7} {'F1':>7} "
        f"{'ref':>5} {'est':>5}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for engine_name in result.metrics:
        per_item = result.metrics[engine_name]
        for item_id, m in sorted(per_item.items()):
            lines.append(
                f"{engine_name:<14} {item_id:<28} {m.precision:>7.3f} "
                f"{m.recall:>7.3f} {m.f1:>7.3f} {m.n_ref:>5d} {m.n_est:>5d}"
            )
        for item_id, err in sorted(result.errors[engine_name].items()):
            lines.append(f"{engine_name:<14} {item_id:<28}   ERROR: {err}")
        if per_item:
            mean_p = sum(m.precision for m in per_item.values()) / len(per_item)
            mean_r = sum(m.recall for m in per_item.values()) / len(per_item)
            mean_f1 = sum(m.f1 for m in per_item.values()) / len(per_item)
            lines.append(
                f"{engine_name:<14} {'** mean (' + str(len(per_item)) + ' items) **':<28} "
                f"{mean_p:>7.3f} {mean_r:>7.3f} {mean_f1:>7.3f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
