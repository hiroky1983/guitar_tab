"""転写エンジンの共通インターフェースと notes.json スキーマ。

NoteEvent は audio→notes ステージの共通出力形式（docs/DESIGN.md 参照）。
notes.json のスキーマ:

    {"schema": 1, "notes": [{"onset_sec": ..., "offset_sec": ...,
                             "midi_pitch": ..., "velocity": ..., "confidence": ...}]}

注意: transcribe/_basicpitch_runner.py は別 venv で単独実行されるため
このモジュールを import できない。スキーマを変更する場合はランナー側も同期すること。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

NOTES_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class NoteEvent:
    """1音符イベント。時刻は音源先頭からの秒、ピッチは MIDI ノート番号。

    velocity / confidence は 0.0-1.0。情報がないソース（人間製 GT 等）では 1.0。
    """

    onset_sec: float
    offset_sec: float
    midi_pitch: int
    velocity: float = 1.0
    confidence: float = 1.0


@runtime_checkable
class TranscriberEngine(Protocol):
    """audio→notes エンジンの差し替え可能インターフェース。"""

    name: str

    def transcribe(self, audio_path: Path) -> list[NoteEvent]: ...


def sort_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    """onset → pitch の順で安定ソートしたコピーを返す。"""
    return sorted(notes, key=lambda n: (n.onset_sec, n.midi_pitch))


def notes_to_dicts(notes: list[NoteEvent]) -> list[dict]:
    return [asdict(n) for n in sort_notes(notes)]


def notes_from_dicts(dicts: list[dict]) -> list[NoteEvent]:
    return [
        NoteEvent(
            onset_sec=float(d["onset_sec"]),
            offset_sec=float(d["offset_sec"]),
            midi_pitch=int(d["midi_pitch"]),
            velocity=float(d.get("velocity", 1.0)),
            confidence=float(d.get("confidence", 1.0)),
        )
        for d in dicts
    ]


def save_notes(notes: list[NoteEvent], path: Path) -> Path:
    """NoteEvent リストを notes.json 形式で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": NOTES_SCHEMA_VERSION, "notes": notes_to_dicts(notes)}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return path


def load_notes(path: Path) -> list[NoteEvent]:
    """notes.json を読む。生のリスト形式（schema ラッパーなし）も許容する。"""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        schema = data.get("schema", NOTES_SCHEMA_VERSION)
        if schema != NOTES_SCHEMA_VERSION:
            raise ValueError(f"unsupported notes.json schema: {schema} ({path})")
        dicts = data["notes"]
    elif isinstance(data, list):
        dicts = data
    else:
        raise ValueError(f"unrecognized notes.json structure: {path}")
    return notes_from_dicts(dicts)
