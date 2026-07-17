"""mir_eval.transcription ラッパー — note-level Precision/Recall/F1。

マッチ条件（mir_eval 標準）:
- onset が onset_tolerance（デフォルト 50ms）以内
- ピッチが pitch_tolerance_cents（デフォルト 50 cents = 同一半音）以内
- offset は評価しない（offset_ratio=None）。音価の評価は後段マイルストーンで検討。

精度の数値は本モジュール経由の実測値のみを正とする（docs/DESIGN.md 開発ルール2）。
"""

from __future__ import annotations

from dataclasses import dataclass

import mir_eval
import numpy as np

from guitartab.transcribe.base import NoteEvent

DEFAULT_ONSET_TOLERANCE_SEC = 0.05
DEFAULT_PITCH_TOLERANCE_CENTS = 50.0  # ±50 cents = 整数 MIDI ピッチなら同一半音のみ一致


@dataclass(frozen=True)
class NoteMetrics:
    precision: float
    recall: float
    f1: float
    n_ref: int
    n_est: int

    def __str__(self) -> str:
        return (
            f"P={self.precision:.3f} R={self.recall:.3f} F1={self.f1:.3f} "
            f"(ref={self.n_ref}, est={self.n_est})"
        )


def _midi_to_hz(midi: np.ndarray) -> np.ndarray:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def _to_mir_eval_arrays(notes: list[NoteEvent]) -> tuple[np.ndarray, np.ndarray]:
    """NoteEvent リストを mir_eval 用の (intervals, pitches_hz) に変換する。"""
    if not notes:
        return np.zeros((0, 2)), np.array([])
    intervals = np.array([[n.onset_sec, n.offset_sec] for n in notes], dtype=float)
    # mir_eval は正の音価を要求する。offset<=onset のデータは最小音価に丸める。
    min_dur = 1e-4
    intervals[:, 1] = np.maximum(intervals[:, 1], intervals[:, 0] + min_dur)
    pitches = _midi_to_hz(np.array([n.midi_pitch for n in notes], dtype=float))
    return intervals, pitches


def evaluate_notes(
    est_notes: list[NoteEvent],
    ref_notes: list[NoteEvent],
    *,
    onset_tolerance_sec: float = DEFAULT_ONSET_TOLERANCE_SEC,
    pitch_tolerance_cents: float = DEFAULT_PITCH_TOLERANCE_CENTS,
) -> NoteMetrics:
    """推定ノート列を ground truth と比較し note-level P/R/F1 を返す。

    Args:
        est_notes: 推定（エンジン出力）
        ref_notes: ground truth
    """
    n_ref, n_est = len(ref_notes), len(est_notes)
    if n_ref == 0 and n_est == 0:
        return NoteMetrics(1.0, 1.0, 1.0, 0, 0)
    if n_ref == 0 or n_est == 0:
        return NoteMetrics(0.0, 0.0, 0.0, n_ref, n_est)

    ref_intervals, ref_pitches = _to_mir_eval_arrays(ref_notes)
    est_intervals, est_pitches = _to_mir_eval_arrays(est_notes)

    precision, recall, f1, _overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=onset_tolerance_sec,
            pitch_tolerance=pitch_tolerance_cents,
            offset_ratio=None,  # offset は評価しない
        )
    )
    return NoteMetrics(float(precision), float(recall), float(f1), n_ref, n_est)
