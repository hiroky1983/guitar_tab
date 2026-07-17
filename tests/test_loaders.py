"""GT ローダーのテスト。eval_data/gt/ は読み取り専用（書き込み厳禁）。"""

import json
from pathlib import Path

import pytest

from guitartab.eval.loaders import (
    load_ground_truth,
    load_jams_note_midi,
    load_v1_ground_truth,
    string_fret_to_midi,
)
from guitartab.transcribe.base import NoteEvent, save_notes

REPO_ROOT = Path(__file__).resolve().parents[1]
FROZEN_GT = REPO_ROOT / "eval_data" / "gt" / "ground_truth.json"


def test_string_fret_to_midi_standard_tuning():
    assert string_fret_to_midi(1, 0) == 64  # 1弦開放 E4
    assert string_fret_to_midi(3, 0) == 55  # 3弦開放 G3（v1 が検出できなかった音）
    assert string_fret_to_midi(5, 0) == 45  # 5弦開放 A2
    assert string_fret_to_midi(6, 0) == 40  # 6弦開放 E2
    assert string_fret_to_midi(2, 11) == 70
    with pytest.raises(ValueError):
        string_fret_to_midi(7, 0)
    with pytest.raises(ValueError):
        string_fret_to_midi(1, -1)


def test_load_frozen_v1_ground_truth():
    notes = load_v1_ground_truth(FROZEN_GT)
    assert len(notes) == 17
    # 先頭: time=0.0, string=5, fret=0 → A2 (MIDI 45)
    assert notes[0] == NoteEvent(onset_sec=0.0, offset_sec=0.256, midi_pitch=45)
    # 末尾: time=4.352, string=2, fret=13 → MIDI 72
    assert notes[-1].onset_sec == pytest.approx(4.352)
    assert notes[-1].offset_sec == pytest.approx(4.608)
    assert notes[-1].midi_pitch == 72
    # 時刻順ソート済み
    onsets = [n.onset_sec for n in notes]
    assert onsets == sorted(onsets)


def test_load_ground_truth_dispatches_v1_format():
    assert load_ground_truth(FROZEN_GT) == load_v1_ground_truth(FROZEN_GT)


def test_load_ground_truth_dispatches_notes_json(tmp_path):
    notes = [NoteEvent(0.0, 0.5, 52), NoteEvent(0.5, 1.0, 57)]
    path = save_notes(notes, tmp_path / "gt.json")
    assert load_ground_truth(path) == notes


def test_load_jams_note_midi(tmp_path):
    # GuitarSet 形式の最小 JAMS: 弦ごとの note_midi アノテーション（抜粋2本）
    jams_doc = {
        "annotations": [
            {
                "namespace": "note_midi",
                "data": [
                    {"time": 0.5, "duration": 0.25, "value": 55.1, "confidence": None},
                ],
            },
            {
                "namespace": "beat",  # 無関係な namespace は無視される
                "data": [{"time": 0.0, "duration": 0.0, "value": 1, "confidence": None}],
            },
            {
                "namespace": "note_midi",
                "data": [
                    {"time": 0.0, "duration": 0.5, "value": 40.0, "confidence": 0.9},
                ],
            },
        ],
        "file_metadata": {"title": "synthetic"},
    }
    path = tmp_path / "sample.jams"
    path.write_text(json.dumps(jams_doc))

    notes = load_jams_note_midi(path)
    assert notes == [
        NoteEvent(0.0, 0.5, 40, confidence=0.9),
        NoteEvent(0.5, 0.75, 55),  # 55.1 は最近傍の半音へ丸め
    ]
    # 拡張子ディスパッチでも同じ結果
    assert load_ground_truth(path) == notes


def test_load_jams_without_note_midi_raises(tmp_path):
    path = tmp_path / "empty.jams"
    path.write_text(json.dumps({"annotations": []}))
    with pytest.raises(ValueError):
        load_jams_note_midi(path)
