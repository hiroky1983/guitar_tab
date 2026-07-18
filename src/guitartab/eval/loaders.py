"""ground truth ローダー群。eval_data/gt/ 配下は凍結（読み取り専用）。

対応形式:
1. v1 人間製 GT（eval_data/gt/ground_truth.json）:
   [{"time": sec, "string": 1-6, "fret": int, "duration": sec}, ...]
   string は 1=1弦(高音E) 〜 6=6弦(低音E)、標準チューニング前提で MIDI に変換する。
2. notes.json（guitartab.transcribe.base のスキーマ）
3. GuitarSet JAMS（namespace "note_midi"）— JAMS は素の JSON なので jams
   パッケージには依存せずに直接パースする。
"""

from __future__ import annotations

import json
from pathlib import Path

from guitartab.transcribe.base import NoteEvent, load_notes, sort_notes

# 標準チューニング: 弦番号(1=高音E) → 開放弦の MIDI ノート番号
STANDARD_TUNING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

JAMS_NOTE_NAMESPACE = "note_midi"


def string_fret_to_midi(string: int, fret: int) -> int:
    """弦番号（1-6）とフレットから標準チューニングの MIDI ピッチを返す。"""
    if string not in STANDARD_TUNING_MIDI:
        raise ValueError(f"invalid string number: {string} (expected 1-6)")
    if fret < 0:
        raise ValueError(f"invalid fret: {fret}")
    return STANDARD_TUNING_MIDI[string] + fret


def load_v1_ground_truth(path: Path) -> list[NoteEvent]:
    """v1 形式（time/string/fret/duration のリスト）を読む。"""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError(f"v1 ground truth must be a JSON list: {path}")
    notes = [
        NoteEvent(
            onset_sec=float(d["time"]),
            offset_sec=float(d["time"]) + float(d["duration"]),
            midi_pitch=string_fret_to_midi(int(d["string"]), int(d["fret"])),
        )
        for d in data
    ]
    return sort_notes(notes)


def load_jams_note_midi(path: Path) -> list[NoteEvent]:
    """GuitarSet 等の JAMS ファイルから note_midi アノテーションを全て読み込む。

    GuitarSet は弦ごとに 6 個の note_midi アノテーションを持つため、
    全アノテーションをマージして時刻順に返す。
    """
    data = json.loads(Path(path).read_text())
    annotations = [
        ann
        for ann in data.get("annotations", [])
        if ann.get("namespace") == JAMS_NOTE_NAMESPACE
    ]
    if not annotations:
        raise ValueError(f"no '{JAMS_NOTE_NAMESPACE}' annotations found in {path}")

    notes: list[NoteEvent] = []
    for ann in annotations:
        for obs in ann.get("data", []):
            onset = float(obs["time"])
            duration = float(obs["duration"])
            confidence = obs.get("confidence")
            notes.append(
                NoteEvent(
                    onset_sec=onset,
                    offset_sec=onset + duration,
                    midi_pitch=round(float(obs["value"])),
                    confidence=1.0 if confidence is None else float(confidence),
                )
            )
    return sort_notes(notes)


def load_jams_tempo(path: Path) -> float:
    """JAMS の tempo アノテーション（GuitarSet: 全曲グローバル 1 値）を読む。"""
    data = json.loads(Path(path).read_text())
    for ann in data.get("annotations", []):
        if ann.get("namespace") == "tempo":
            observations = ann.get("data", [])
            if observations:
                return float(observations[0]["value"])
    raise ValueError(f"no 'tempo' annotation found in {path}")


def load_jams_beats(path: Path) -> list[dict]:
    """JAMS の beat_position アノテーションを読む。

    返り値: [{"time_sec": float, "measure": int, "position": int}, ...]（時刻順）。
    """
    data = json.loads(Path(path).read_text())
    for ann in data.get("annotations", []):
        if ann.get("namespace") == "beat_position":
            beats = [
                {
                    "time_sec": float(obs["time"]),
                    "measure": int(obs["value"]["measure"]),
                    "position": int(obs["value"]["position"]),
                }
                for obs in ann.get("data", [])
            ]
            if beats:
                return sorted(beats, key=lambda b: b["time_sec"])
    raise ValueError(f"no 'beat_position' annotation found in {path}")


def load_ground_truth(path: Path) -> list[NoteEvent]:
    """GT ファイルを拡張子と中身から判別して NoteEvent リストとして読む。"""
    path = Path(path)
    if path.suffix == ".jams":
        return load_jams_note_midi(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list) and data and "string" in data[0]:
            return load_v1_ground_truth(path)
        return sort_notes(load_notes(path))
    raise ValueError(f"unsupported ground truth format: {path}")
