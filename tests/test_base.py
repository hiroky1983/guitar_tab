"""NoteEvent / notes.json シリアライズの往復テスト。"""

import json

from guitartab.transcribe.base import (
    NOTES_SCHEMA_VERSION,
    NoteEvent,
    load_notes,
    notes_from_dicts,
    notes_to_dicts,
    save_notes,
)


def test_roundtrip_via_file(tmp_path):
    notes = [
        NoteEvent(onset_sec=1.5, offset_sec=1.75, midi_pitch=55, velocity=0.8, confidence=0.9),
        NoteEvent(onset_sec=0.0, offset_sec=0.25, midi_pitch=45),
    ]
    path = tmp_path / "notes.json"
    save_notes(notes, path)
    loaded = load_notes(path)

    # save_notes は onset→pitch 順にソートして書く
    assert loaded == [notes[1], notes[0]]


def test_roundtrip_via_dicts():
    notes = [NoteEvent(0.1, 0.2, 60, 0.5, 0.7)]
    assert notes_from_dicts(notes_to_dicts(notes)) == notes


def test_saved_file_has_schema_version(tmp_path):
    path = save_notes([NoteEvent(0.0, 0.1, 40)], tmp_path / "notes.json")
    data = json.loads(path.read_text())
    assert data["schema"] == NOTES_SCHEMA_VERSION
    assert data["notes"][0]["midi_pitch"] == 40


def test_defaults_velocity_confidence():
    n = NoteEvent(0.0, 0.1, 64)
    assert n.velocity == 1.0
    assert n.confidence == 1.0
    # velocity/confidence 欠落の JSON も読める
    loaded = notes_from_dicts([{"onset_sec": 0, "offset_sec": 0.1, "midi_pitch": 64}])
    assert loaded == [n]


def test_load_notes_accepts_bare_list(tmp_path):
    path = tmp_path / "bare.json"
    path.write_text(json.dumps([{"onset_sec": 0.0, "offset_sec": 0.5, "midi_pitch": 50}]))
    assert load_notes(path) == [NoteEvent(0.0, 0.5, 50)]
