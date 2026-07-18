"""rhythm.json スキーマ（docs/DESIGN_M4_QUANTIZATION.md §5.2）。

    {
      "schema_version": 1,
      "estimator": "librosa_constant",
      "estimator_params": {...},
      "audio_source": "stems/guitar.wav",     // null = ノートのみで推定
      "tempo_bpm": 119.0,                     // グローバル代表値
      "tempo_map": [{"time_sec": 0.0, "bpm": 119.0}],  // M4a は 1 要素
      "time_signature": {"beats": 4, "beat_unit": 4},  // M4b までは 4/4 固定
      "anacrusis_ticks": 0,
      "divisions_per_quarter": 12,            // 16分=3 tick, 3連8分=4 tick
      "beats": [{"time_sec": 0.0, "measure": 1, "position": 1}],
      "notes": [{"onset_tick": 0, "duration_ticks": 6, "deviation_sec": -0.012}]
    }

- tick 0 の物理時刻は tempo_map[0].time_sec（格子原点）。
  tick_to_sec(t) = time_sec + t / divisions_per_quarter * 60 / bpm。
- notes[] は notes.json と同数・同順（インデックス対応）。
- deviation_sec = 元 onset_sec − 量子化後の格子時刻（演奏が格子より遅い側が正）。
  元の onset_sec は 格子時刻 + deviation_sec で復元でき、tab.json（fingering の
  除外で音符数が減り得る）との照合は onset_sec 経由で行える。
- M4a の小節番号・拍位置は「格子原点 = 小節 1 拍 1」の機械的割当
  （ダウンビート推定は M4c スコープ）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

RHYTHM_SCHEMA_VERSION = 1

# quarter = 12 divisions（16 分 = 3 tick、3 連 8 分 = 4 tick。4 と 3 の LCM）
DIVISIONS_PER_QUARTER = 12
# 拍内で許容する格子位置（tick 剰余）: 16 分 {0,3,6,9} ∪ 3 連 {0,4,8}
ALLOWED_TICK_RESIDUES = (0, 3, 4, 6, 8, 9)


@dataclass(frozen=True)
class TempoPoint:
    time_sec: float
    bpm: float


@dataclass(frozen=True)
class Beat:
    time_sec: float
    measure: int
    position: int


@dataclass(frozen=True)
class RhythmNote:
    onset_tick: int
    duration_ticks: int
    deviation_sec: float


@dataclass
class Rhythm:
    tempo_bpm: float
    tempo_map: list[TempoPoint]
    notes: list[RhythmNote]
    beats: list[Beat] = field(default_factory=list)
    estimator: str = ""
    estimator_params: dict = field(default_factory=dict)
    audio_source: str | None = None
    time_signature: tuple[int, int] = (4, 4)  # (beats, beat_unit)
    anacrusis_ticks: int = 0
    divisions_per_quarter: int = DIVISIONS_PER_QUARTER
    schema_version: int = RHYTHM_SCHEMA_VERSION

    def tick_to_sec(self, tick: float) -> float:
        """tick → 物理時刻（M4a: 一定テンポ = tempo_map 先頭のみ使用）。"""
        origin = self.tempo_map[0]
        return origin.time_sec + tick / self.divisions_per_quarter * 60.0 / origin.bpm

    def onset_sec_of(self, index: int) -> float:
        """notes[index] の元 onset_sec を復元する（格子時刻 + deviation）。"""
        note = self.notes[index]
        return self.tick_to_sec(note.onset_tick) + note.deviation_sec


def lookup_note_by_onset(
    rhythm: Rhythm, onset_sec: float, *, tolerance_sec: float = 0.005
) -> RhythmNote | None:
    """元 onset_sec が一致する RhythmNote を返す（なければ None）。

    tab.json は fingering の除外で notes.json より音符数が減り得るため、
    インデックスではなく onset_sec（格子時刻 + deviation で復元）で引く。
    同一 onset の和音は同じ onset_tick を持つので、どれに当たっても等価。
    """
    best: RhythmNote | None = None
    best_dist = tolerance_sec
    for i, note in enumerate(rhythm.notes):
        dist = abs(rhythm.onset_sec_of(i) - onset_sec)
        if dist <= best_dist:
            best, best_dist = note, dist
    return best


def save_rhythm(rhythm: Rhythm, path: Path) -> Path:
    """Rhythm を rhythm.json 形式で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": rhythm.schema_version,
        "estimator": rhythm.estimator,
        "estimator_params": rhythm.estimator_params,
        "audio_source": rhythm.audio_source,
        "tempo_bpm": rhythm.tempo_bpm,
        "tempo_map": [
            {"time_sec": p.time_sec, "bpm": p.bpm} for p in rhythm.tempo_map
        ],
        "time_signature": {
            "beats": rhythm.time_signature[0],
            "beat_unit": rhythm.time_signature[1],
        },
        "anacrusis_ticks": rhythm.anacrusis_ticks,
        "divisions_per_quarter": rhythm.divisions_per_quarter,
        "beats": [
            {"time_sec": b.time_sec, "measure": b.measure, "position": b.position}
            for b in rhythm.beats
        ],
        "notes": [
            {
                "onset_tick": n.onset_tick,
                "duration_ticks": n.duration_ticks,
                "deviation_sec": n.deviation_sec,
            }
            for n in rhythm.notes
        ],
    }
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return path


def load_rhythm(path: Path) -> Rhythm:
    """rhythm.json を読む。"""
    data = json.loads(Path(path).read_text())
    schema = data.get("schema_version")
    if schema != RHYTHM_SCHEMA_VERSION:
        raise ValueError(f"unsupported rhythm.json schema_version: {schema} ({path})")
    ts = data.get("time_signature", {"beats": 4, "beat_unit": 4})
    return Rhythm(
        tempo_bpm=float(data["tempo_bpm"]),
        tempo_map=[
            TempoPoint(time_sec=float(p["time_sec"]), bpm=float(p["bpm"]))
            for p in data["tempo_map"]
        ],
        notes=[
            RhythmNote(
                onset_tick=int(n["onset_tick"]),
                duration_ticks=int(n["duration_ticks"]),
                deviation_sec=float(n["deviation_sec"]),
            )
            for n in data["notes"]
        ],
        beats=[
            Beat(
                time_sec=float(b["time_sec"]),
                measure=int(b["measure"]),
                position=int(b["position"]),
            )
            for b in data.get("beats", [])
        ],
        estimator=str(data.get("estimator", "")),
        estimator_params=dict(data.get("estimator_params", {})),
        audio_source=data.get("audio_source"),
        time_signature=(int(ts["beats"]), int(ts["beat_unit"])),
        anacrusis_ticks=int(data.get("anacrusis_ticks", 0)),
        divisions_per_quarter=int(
            data.get("divisions_per_quarter", DIVISIONS_PER_QUARTER)
        ),
    )


__all__ = [
    "ALLOWED_TICK_RESIDUES",
    "Beat",
    "DIVISIONS_PER_QUARTER",
    "RHYTHM_SCHEMA_VERSION",
    "Rhythm",
    "RhythmNote",
    "TempoPoint",
    "load_rhythm",
    "lookup_note_by_onset",
    "save_rhythm",
]
