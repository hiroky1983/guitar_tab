"""リズム量子化ステージ（M4）。

notes.json（物理時刻の音符列）+ 任意の音声から、一定テンポ + 位相の格子
（16 分 + 3 連、quarter = 12 divisions）を推定し、各音符を格子位置（tick）に
割り当てて rhythm.json を出力する。設計は docs/DESIGN_M4_QUANTIZATION.md。

- estimate: テンポ・位相の推定（差し替え可能な TempoEstimator Protocol）
- quantize: 最近傍格子スナップ（M4a: 分岐なしの最小規則）
- schema:   rhythm.json の入出力
"""

from guitartab.rhythm.estimate import (
    LibrosaConstantTempoEstimator,
    TempoEstimate,
    TempoEstimator,
)
from guitartab.rhythm.quantize import quantize_notes
from guitartab.rhythm.schema import (
    ALLOWED_TICK_RESIDUES,
    DIVISIONS_PER_QUARTER,
    Rhythm,
    RhythmNote,
    load_rhythm,
    lookup_note_by_onset,
    save_rhythm,
)

__all__ = [
    "ALLOWED_TICK_RESIDUES",
    "DIVISIONS_PER_QUARTER",
    "LibrosaConstantTempoEstimator",
    "Rhythm",
    "RhythmNote",
    "TempoEstimate",
    "TempoEstimator",
    "load_rhythm",
    "lookup_note_by_onset",
    "quantize_notes",
    "save_rhythm",
]
