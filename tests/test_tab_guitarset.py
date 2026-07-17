"""GuitarSet 実 GT に対する運指割当の演奏可能性テスト（読み取りのみ）と
CLI tab サブコマンドの統合テスト。

eval_data/ 配下には一切書き込まない。
"""

from pathlib import Path

import pytest

from guitartab.cli import main
from guitartab.eval.loaders import load_jams_note_midi
from guitartab.tab.fingering import (
    MAX_FRET,
    STANDARD_TUNING_MIDI,
    assign_fingering,
    load_tab,
)
from guitartab.transcribe.base import NoteEvent, save_notes

REPO_ROOT = Path(__file__).resolve().parents[1]
GUITARSET_JAMS = (
    REPO_ROOT / "eval_data" / "guitarset" / "annotations" / "00_BN3-119-G_solo.jams"
)


@pytest.mark.skipif(not GUITARSET_JAMS.exists(), reason="GuitarSet annotations not present")
def test_guitarset_track_fully_playable():
    notes = load_jams_note_midi(GUITARSET_JAMS)
    assert notes, "GT が空"

    tab = assign_fingering(notes)

    # 全ノートが割当され、演奏可能範囲に収まる
    assert len(tab) == len(notes)
    for t in tab:
        assert 1 <= t.string <= 6
        assert 0 <= t.fret <= MAX_FRET
        assert t.midi_pitch == STANDARD_TUNING_MIDI[t.string] + t.fret

    # GT のピッチ集合が保存されている（GuitarSet は標準チューニング内なのでクランプなし）
    assert sorted(t.midi_pitch for t in tab) == sorted(n.midi_pitch for n in notes)

    # 同時発音（同一 onset グループ）内で同一弦の競合がない
    by_onset: dict[float, list[int]] = {}
    for t in tab:
        by_onset.setdefault(round(t.onset_sec, 3), []).append(t.string)
    for strings in by_onset.values():
        assert len(strings) == len(set(strings))


# --- CLI 統合: python -m guitartab tab <notes.json> ---


def test_cli_tab_subcommand(tmp_path, capsys):
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=0.4, midi_pitch=48),
        NoteEvent(onset_sec=0.0, offset_sec=0.4, midi_pitch=55),
        NoteEvent(onset_sec=0.5, offset_sec=0.9, midi_pitch=64),
    ]
    notes_path = tmp_path / "notes.json"
    save_notes(notes, notes_path)

    assert main(["tab", str(notes_path)]) == 0

    tab_path = tmp_path / "tab.json"
    txt_path = tmp_path / "tab.txt"
    assert tab_path.exists() and txt_path.exists()
    assert len(load_tab(tab_path)) == 3
    text = txt_path.read_text()
    assert text.startswith("e|")
    assert len(text.splitlines()) >= 6

    out = capsys.readouterr().out
    assert str(tab_path) in out and str(txt_path) in out


def test_cli_tab_refuses_eval_data_output(tmp_path):
    notes_path = tmp_path / "notes.json"
    save_notes([NoteEvent(0.0, 0.1, 60)], notes_path)
    fake_eval = tmp_path / "eval_data" / "gt"
    with pytest.raises(SystemExit, match="eval_data"):
        main(["tab", str(notes_path), "--out-dir", str(fake_eval)])
