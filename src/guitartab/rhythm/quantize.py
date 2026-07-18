"""最近傍格子スナップ（M4a: §3.1 方式 A、分岐なしの最小規則）。

TempoEstimate（一定テンポ + 位相）の格子（16 分 + 3 連、quarter=12 tick）へ
各音符の onset を最近傍スナップし、rhythm.json（Rhythm）を組み立てる。

- スナップ規則に分岐を作らない（v1 の轍回避。設計 §3.1）。
  各 onset は拍内の許容剰余 {0,3,4,6,8,9}（+ 次拍の 0）の最近傍へ落とすだけ。
- duration は tick 幅で丸め、最小 1 tick（音価の縮退整形はレンダラ側の責務）。
- 格子原点（tick 0）は「推定位相の拍のうち、全音符が非負 tick になる最も遅い
  小節境界」= 位相から小節単位で繰り上げた時刻。tempo_map[0].time_sec に記録する。
- 小節番号・拍位置は格子原点 = 小節 1 拍 1 の機械的割当（ダウンビートは M4c）。
"""

from __future__ import annotations

from typing import Sequence

from guitartab.rhythm.estimate import TempoEstimate
from guitartab.rhythm.schema import (
    ALLOWED_TICK_RESIDUES,
    DIVISIONS_PER_QUARTER,
    Beat,
    Rhythm,
    RhythmNote,
    TempoPoint,
)
from guitartab.transcribe.base import NoteEvent

# 拍内の許容剰余 + 次拍の頭（12）。スナップの最近傍候補。
_SNAP_TICKS = sorted(ALLOWED_TICK_RESIDUES) + [DIVISIONS_PER_QUARTER]


def snap_to_grid(onset_sec: float, estimate: TempoEstimate) -> int:
    """onset を 16 分+3 連格子の最近傍 tick（位相基準の絶対 tick、負もあり得る）へ。

    同距離のタイ（例: 3.5 tick は 16 分 3 と 3 連 4 の中点）は小さい側
    （先行する格子点）に決定的に倒す。
    """
    period = estimate.beat_period_sec
    beats_from_origin = (onset_sec - estimate.grid_origin_sec) / period
    beat_index = int(beats_from_origin // 1)
    frac_ticks = (beats_from_origin - beat_index) * DIVISIONS_PER_QUARTER
    best = min(_SNAP_TICKS, key=lambda r: (abs(frac_ticks - r), r))
    return beat_index * DIVISIONS_PER_QUARTER + best


def quantize_notes(
    notes: Sequence[NoteEvent],
    estimate: TempoEstimate,
    *,
    audio_source: str | None = None,
    beats_per_measure: int = 4,
) -> Rhythm:
    """NoteEvent 列を格子へスナップして Rhythm を返す（入力は変更しない）。

    notes は notes.json の順（onset, pitch ソート済み）を想定し、
    Rhythm.notes は同数・同順で対応する。
    """
    period = estimate.beat_period_sec
    tick_sec = period / DIVISIONS_PER_QUARTER
    raw_ticks = [snap_to_grid(n.onset_sec, estimate) for n in notes]

    # 全音符が非負 tick になるよう、格子原点を小節単位で前へずらす
    ticks_per_measure = beats_per_measure * DIVISIONS_PER_QUARTER
    min_tick = min(raw_ticks, default=0)
    shift_measures = max(0, -(min_tick // ticks_per_measure))  # ceil(-min/48)
    shift_ticks = shift_measures * ticks_per_measure
    origin_sec = estimate.grid_origin_sec - shift_ticks * tick_sec

    rhythm_notes: list[RhythmNote] = []
    for note, raw_tick in zip(notes, raw_ticks):
        tick = raw_tick + shift_ticks
        grid_time = origin_sec + tick * tick_sec
        duration_ticks = max(1, round((note.offset_sec - note.onset_sec) / tick_sec))
        rhythm_notes.append(
            RhythmNote(
                onset_tick=tick,
                duration_ticks=duration_ticks,
                deviation_sec=note.onset_sec - grid_time,
            )
        )

    # 拍列（評価・デバッグ用）: 格子原点から最終音符の tick まで
    last_tick = max((n.onset_tick for n in rhythm_notes), default=0)
    n_beats = last_tick // DIVISIONS_PER_QUARTER + 1
    beats = [
        Beat(
            time_sec=origin_sec + i * period,
            measure=i // beats_per_measure + 1,
            position=i % beats_per_measure + 1,
        )
        for i in range(n_beats)
    ]

    return Rhythm(
        tempo_bpm=estimate.bpm,
        tempo_map=[TempoPoint(time_sec=origin_sec, bpm=estimate.bpm)],
        notes=rhythm_notes,
        beats=beats,
        estimator=estimate.estimator,
        estimator_params=dict(estimate.params),
        audio_source=audio_source,
        time_signature=(beats_per_measure, 4),
        anacrusis_ticks=0,
    )


__all__ = ["quantize_notes", "snap_to_grid"]
