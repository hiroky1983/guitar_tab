"""リズム量子化メトリクス（docs/DESIGN_M4_QUANTIZATION.md §4.1、mir_eval ベース）。

- TempoAcc1: 推定グローバルテンポが GT の ±4% 以内（オクターブも正しい）
- TempoAcc2: GT × {1/3, 1/2, 1, 2, 3} のいずれかの ±4% 以内（標準 Accuracy2）
- Beat F-measure / CMLt / AMLt: mir_eval.beat（±70ms 標準、先頭 5 秒トリム標準）
- GPA (Grid Position Accuracy): onset_tick の一致率。**合成ベンチ限定**
  （実データには per-note の格子 GT が存在しない — 設計 §1.2/§4.2）
- 量子化変位分布: |量子化後時刻 − 元 onset| の中央値 / p90（診断用の参考統計。
  小さいほど良いわけではなく、最適化目標にしない）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mir_eval
import numpy as np

DEFAULT_TEMPO_TOLERANCE = 0.04
TEMPO_OCTAVE_FACTORS = (1 / 3, 1 / 2, 1.0, 2.0, 3.0)


def tempo_acc1(
    est_bpm: float, ref_bpm: float, *, tolerance: float = DEFAULT_TEMPO_TOLERANCE
) -> bool:
    """推定テンポが GT の ±tolerance（比率）以内か。"""
    if ref_bpm <= 0:
        raise ValueError(f"ref_bpm must be positive: {ref_bpm}")
    return abs(est_bpm - ref_bpm) / ref_bpm <= tolerance


def tempo_acc2(
    est_bpm: float, ref_bpm: float, *, tolerance: float = DEFAULT_TEMPO_TOLERANCE
) -> bool:
    """GT のオクターブ族（×{1/3, 1/2, 1, 2, 3}）のいずれかに ±tolerance 以内か。"""
    return any(
        tempo_acc1(est_bpm, ref_bpm * f, tolerance=tolerance)
        for f in TEMPO_OCTAVE_FACTORS
    )


@dataclass(frozen=True)
class BeatMetrics:
    f_measure: float
    cmlt: float
    amlt: float


def evaluate_beats(
    est_beats_sec: Sequence[float], ref_beats_sec: Sequence[float]
) -> BeatMetrics:
    """推定拍列を GT 拍列と比較する（mir_eval 標準: ±70ms、先頭 5 秒トリム）。"""
    ref = mir_eval.beat.trim_beats(np.asarray(ref_beats_sec, dtype=float))
    est = mir_eval.beat.trim_beats(np.asarray(est_beats_sec, dtype=float))
    f = mir_eval.beat.f_measure(ref, est)
    _cmlc, cmlt, _amlc, amlt = mir_eval.beat.continuity(ref, est)
    return BeatMetrics(float(f), float(cmlt), float(amlt))


def grid_position_accuracy(
    est_ticks: Sequence[int], ref_ticks: Sequence[int]
) -> float:
    """onset_tick が GT スコアの tick と一致するノートの割合（合成ベンチ限定）。

    est_ticks と ref_ticks は同数・同順（インデックス対応）であること。
    """
    if len(est_ticks) != len(ref_ticks):
        raise ValueError(
            f"est/ref tick count mismatch: {len(est_ticks)} vs {len(ref_ticks)}"
        )
    if not ref_ticks:
        return 1.0
    matches = sum(1 for e, r in zip(est_ticks, ref_ticks) if int(e) == int(r))
    return matches / len(ref_ticks)


@dataclass(frozen=True)
class DisplacementStats:
    median_sec: float
    p90_sec: float


def displacement_stats(deviations_sec: Sequence[float]) -> DisplacementStats:
    """量子化変位 |deviation_sec| の中央値と p90（診断用の参考統計）。"""
    if not deviations_sec:
        return DisplacementStats(0.0, 0.0)
    magnitudes = np.abs(np.asarray(deviations_sec, dtype=float))
    return DisplacementStats(
        float(np.median(magnitudes)), float(np.percentile(magnitudes, 90))
    )


__all__ = [
    "BeatMetrics",
    "DEFAULT_TEMPO_TOLERANCE",
    "DisplacementStats",
    "TEMPO_OCTAVE_FACTORS",
    "displacement_stats",
    "evaluate_beats",
    "grid_position_accuracy",
    "tempo_acc1",
    "tempo_acc2",
]
