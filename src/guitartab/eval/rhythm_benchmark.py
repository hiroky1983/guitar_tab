"""リズム量子化ベンチ（M4）の一括評価。docs/DESIGN_M4_QUANTIZATION.md §4.4。

2 種のベンチアイテムに対応:

1. 実データ（GuitarSet 形式: annotations/*.jams + audio/）
   - 測るもの: TempoAcc1/2、Beat F-measure / CMLt / AMLt、量子化変位（参考統計）
   - 音符 onset は JAMS の GT ノートを使う（転写誤差と量子化誤差を混ぜない、
     量子化ステージの単独評価。設計 §4.3-4 の「真の onset 入力」モード相当）
   - per-note GPA は実データでは定義不能のため**報告しない**（設計 §4.2）

2. 合成リズムベンチ（eval_data/rhythm_synth/ 形式: <variant>/<track>/{audio.wav, score.json}）
   - 測るもの: GPA（per-note tick 一致率）、TempoAcc1/2、量子化変位
   - 既定は score.json の物理 onset を直接量子化ステージへ入れる単独評価モード。
     engine を渡すとフル転写経由（E2E）になり、GPA は mir_eval の
     ノートマッチ（onset 50ms・同一半音）で対応付いたノートに対して算出する。
"""

from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import mir_eval
import numpy as np

from guitartab.eval.loaders import (
    load_jams_beats,
    load_jams_note_midi,
    load_jams_tempo,
)
from guitartab.eval.rhythm_metrics import (
    displacement_stats,
    evaluate_beats,
    grid_position_accuracy,
    tempo_acc1,
    tempo_acc2,
)
from guitartab.rhythm.estimate import TempoEstimator
from guitartab.rhythm.quantize import quantize_notes
from guitartab.transcribe.base import NoteEvent, TranscriberEngine


@dataclass(frozen=True)
class RhythmBenchItem:
    item_id: str
    audio_path: Path
    gt_path: Path  # JAMS（実データ）または score.json（合成）
    kind: str  # "real" | "synth"


@dataclass
class RhythmItemResult:
    ref_bpm: float
    est_bpm: float
    acc1: bool
    acc2: bool
    beat_f: float | None = None  # 実データのみ
    cmlt: float | None = None
    amlt: float | None = None
    gpa: float | None = None  # 合成のみ
    n_notes: int = 0
    disp_median_sec: float = 0.0
    disp_p90_sec: float = 0.0


@dataclass
class RhythmBenchmarkResult:
    items: dict[str, RhythmItemResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


def discover_rhythm_items(eval_root: Path) -> list[RhythmBenchItem]:
    """eval_root 以下から実データ（GuitarSet 形式）と合成クリップを集める。"""
    eval_root = Path(eval_root)
    items: list[RhythmBenchItem] = []
    if not eval_root.exists():
        return items
    for directory in sorted(d for d in [eval_root, *eval_root.rglob("*")] if d.is_dir()):
        score = directory / "score.json"
        audio = directory / "audio.wav"
        if score.exists() and audio.exists():
            items.append(
                RhythmBenchItem(
                    item_id=str(directory.relative_to(eval_root))
                    if directory != eval_root
                    else directory.name,
                    audio_path=audio,
                    gt_path=score,
                    kind="synth",
                )
            )
            continue
        ann_dir = directory / "annotations"
        audio_dir = directory / "audio"
        if ann_dir.is_dir() and audio_dir.is_dir():
            audio_files = sorted(audio_dir.glob("*.wav"))
            for jams in sorted(ann_dir.glob("*.jams")):
                matches = [a for a in audio_files if a.stem.startswith(jams.stem)]
                if not matches:
                    continue
                rel = (
                    f"{directory.relative_to(eval_root)}/{jams.stem}"
                    if directory != eval_root
                    else jams.stem
                )
                items.append(
                    RhythmBenchItem(
                        item_id=rel,
                        audio_path=matches[0],
                        gt_path=jams,
                        kind="real",
                    )
                )
    return items


def _est_beat_times(bpm: float, grid_origin_sec: float, duration_sec: float) -> np.ndarray:
    """推定テンポ・位相から一定間隔の拍列を生成する（[0, duration]）。"""
    period = 60.0 / bpm
    start = grid_origin_sec % period
    n = max(0, int((duration_sec - start) / period) + 1)
    return start + np.arange(n) * period


def _eval_real_item(
    item: RhythmBenchItem, estimator: TempoEstimator, *, use_audio: bool
) -> RhythmItemResult:
    ref_bpm = load_jams_tempo(item.gt_path)
    ref_beats = [b["time_sec"] for b in load_jams_beats(item.gt_path)]
    notes = load_jams_note_midi(item.gt_path)
    estimate = estimator.estimate(
        [n.onset_sec for n in notes],
        audio_path=item.audio_path if use_audio else None,
    )
    rhythm = quantize_notes(notes, estimate)
    duration = max(ref_beats[-1], notes[-1].offset_sec if notes else 0.0)
    beats = _est_beat_times(estimate.bpm, estimate.grid_origin_sec, duration + 1.0)
    bm = evaluate_beats(beats, ref_beats)
    disp = displacement_stats([n.deviation_sec for n in rhythm.notes])
    return RhythmItemResult(
        ref_bpm=ref_bpm,
        est_bpm=estimate.bpm,
        acc1=tempo_acc1(estimate.bpm, ref_bpm),
        acc2=tempo_acc2(estimate.bpm, ref_bpm),
        beat_f=bm.f_measure,
        cmlt=bm.cmlt,
        amlt=bm.amlt,
        n_notes=len(notes),
        disp_median_sec=disp.median_sec,
        disp_p90_sec=disp.p90_sec,
    )


def _eval_synth_item(
    item: RhythmBenchItem,
    estimator: TempoEstimator,
    *,
    use_audio: bool,
    engine: TranscriberEngine | None = None,
) -> RhythmItemResult:
    score = json.loads(item.gt_path.read_text())
    ref_bpm = float(score["tempo_bpm"])
    ref_notes = score["notes"]  # (onset_sec, midi_pitch) 順

    if engine is None:
        # 単独評価モード（本命）: 真のスコアの物理 onset 列を直接量子化
        notes = [
            NoteEvent(
                onset_sec=n["onset_sec"],
                offset_sec=n["onset_sec"] + n["duration_sec"],
                midi_pitch=n["midi_pitch"],
            )
            for n in ref_notes
        ]
        ref_ticks = [n["onset_tick"] for n in ref_notes]
    else:
        # E2E モード: 転写ノートを量子化し、GT とマッチしたノートで GPA を出す
        notes = engine.transcribe(item.audio_path)
        ref_ticks = None  # 後段でマッチングして作る

    estimate = estimator.estimate(
        [n.onset_sec for n in notes],
        audio_path=item.audio_path if use_audio else None,
    )
    rhythm = quantize_notes(notes, estimate)
    est_ticks = [n.onset_tick for n in rhythm.notes]

    if engine is not None:
        ref_intervals = np.array(
            [[n["onset_sec"], n["onset_sec"] + n["duration_sec"]] for n in ref_notes]
        )
        ref_pitches = 440.0 * 2 ** ((np.array([n["midi_pitch"] for n in ref_notes]) - 69) / 12)
        est_intervals = np.array([[n.onset_sec, max(n.offset_sec, n.onset_sec + 1e-4)] for n in notes])
        est_pitches = 440.0 * 2 ** ((np.array([n.midi_pitch for n in notes]) - 69) / 12)
        matching = mir_eval.transcription.match_notes(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, offset_ratio=None,
        )
        ref_ticks = [ref_notes[i]["onset_tick"] for i, _ in matching]
        est_ticks = [est_ticks[j] for _, j in matching]

    disp = displacement_stats([n.deviation_sec for n in rhythm.notes])
    return RhythmItemResult(
        ref_bpm=ref_bpm,
        est_bpm=estimate.bpm,
        acc1=tempo_acc1(estimate.bpm, ref_bpm),
        acc2=tempo_acc2(estimate.bpm, ref_bpm),
        gpa=grid_position_accuracy(est_ticks, ref_ticks),
        n_notes=len(notes),
        disp_median_sec=disp.median_sec,
        disp_p90_sec=disp.p90_sec,
    )


def run_rhythm_benchmark(
    estimator: TempoEstimator,
    items: list[RhythmBenchItem],
    *,
    use_audio: bool = True,
    engine: TranscriberEngine | None = None,
) -> RhythmBenchmarkResult:
    """全アイテムを評価する。engine は合成ベンチの E2E モードのみで使う。"""
    result = RhythmBenchmarkResult()
    for item in items:
        try:
            if item.kind == "real":
                result.items[item.item_id] = _eval_real_item(
                    item, estimator, use_audio=use_audio
                )
            else:
                result.items[item.item_id] = _eval_synth_item(
                    item, estimator, use_audio=use_audio, engine=engine
                )
        except Exception as e:  # 1 アイテムの失敗で全体を止めない
            traceback.print_exc(file=sys.stderr)
            result.errors[item.item_id] = f"{type(e).__name__}: {e}"
    return result


def format_rhythm_table(result: RhythmBenchmarkResult) -> str:
    """トラック別 + mean の比較表（テキスト）を返す。"""
    lines: list[str] = []
    header = (
        f"{'item':<38} {'refBPM':>7} {'estBPM':>7} {'A1':>3} {'A2':>3} "
        f"{'beatF':>6} {'CMLt':>6} {'AMLt':>6} {'GPA':>6} {'disp50':>7} {'disp90':>7}"
    )
    lines.append("rhythm metrics (TempoAcc tol=4%, beat F tol=70ms, disp=sec)")
    lines.append(header)
    lines.append("-" * len(header))

    def fmt(v: float | None, width: int = 6) -> str:
        return f"{v:>{width}.3f}" if v is not None else " " * (width - 1) + "-"

    for item_id, r in sorted(result.items.items()):
        lines.append(
            f"{item_id:<38} {r.ref_bpm:>7.1f} {r.est_bpm:>7.1f} "
            f"{'o' if r.acc1 else 'x':>3} {'o' if r.acc2 else 'x':>3} "
            f"{fmt(r.beat_f)} {fmt(r.cmlt)} {fmt(r.amlt)} {fmt(r.gpa)} "
            f"{r.disp_median_sec:>7.3f} {r.disp_p90_sec:>7.3f}"
        )
    for item_id, err in sorted(result.errors.items()):
        lines.append(f"{item_id:<38}   ERROR: {err}")

    rs = list(result.items.values())
    if rs:
        n = len(rs)
        n_acc1 = sum(r.acc1 for r in rs)
        n_acc2 = sum(r.acc2 for r in rs)
        summary = (
            f"{'** summary (' + str(n) + ' items) **':<38} {'':>7} {'':>7} "
            f"{n_acc1:>3} {n_acc2:>3}"
        )
        beat_fs = [r.beat_f for r in rs if r.beat_f is not None]
        cmlts = [r.cmlt for r in rs if r.cmlt is not None]
        amlts = [r.amlt for r in rs if r.amlt is not None]
        gpas = [r.gpa for r in rs if r.gpa is not None]
        summary += f" {fmt(sum(beat_fs) / len(beat_fs) if beat_fs else None)}"
        summary += f" {fmt(sum(cmlts) / len(cmlts) if cmlts else None)}"
        summary += f" {fmt(sum(amlts) / len(amlts) if amlts else None)}"
        summary += f" {fmt(sum(gpas) / len(gpas) if gpas else None)}"
        lines.append(summary)
    return "\n".join(lines) + "\n"


__all__ = [
    "RhythmBenchItem",
    "RhythmBenchmarkResult",
    "RhythmItemResult",
    "discover_rhythm_items",
    "format_rhythm_table",
    "run_rhythm_benchmark",
]
