"""運指割当 DP（tab/fingering.py）のテスト。"""

import pytest

from guitartab.tab.fingering import (
    MAX_FRET,
    MAX_PITCH,
    MIN_PITCH,
    STANDARD_TUNING_MIDI,
    TabNote,
    assign_fingering,
    load_tab,
    save_tab,
)
from guitartab.transcribe.base import NoteEvent


def _note(onset: float, pitch: int, dur: float = 0.4) -> NoteEvent:
    return NoteEvent(onset_sec=onset, offset_sec=onset + dur, midi_pitch=pitch)


def _played_pitch(t: TabNote) -> int:
    return STANDARD_TUNING_MIDI[t.string] + t.fret


# --- (a) 単音列で開放弦優先 ---


def test_open_strings_preferred_for_open_pitches():
    # E2 A2 D3 G3 B3 E4 = 6 本の開放弦のピッチを順に弾く
    pitches = [40, 45, 50, 55, 59, 64]
    notes = [_note(i * 0.5, p) for i, p in enumerate(pitches)]
    tab = assign_fingering(notes)

    assert len(tab) == 6
    assert all(t.fret == 0 for t in tab), [(t.string, t.fret) for t in tab]
    assert [t.string for t in tab] == [6, 5, 4, 3, 2, 1]


# --- (b) 和音が演奏可能な割当になる ---


def test_c_major_chord_is_playable():
    # C メジャー (x32010): C3 E3 G3 C4 E4
    pitches = [48, 52, 55, 60, 64]
    notes = [_note(0.0, p) for p in pitches]
    tab = assign_fingering(notes)

    assert len(tab) == 5
    strings = [t.string for t in tab]
    assert len(set(strings)) == 5, "同一弦の競合がある"
    fretted = [t.fret for t in tab if t.fret > 0]
    assert fretted and max(fretted) - min(fretted) <= 4, "フレット幅が広すぎる"
    assert sorted(_played_pitch(t) for t in tab) == sorted(pitches)
    # 標準的な運指 x32010 が最小コストになるはず
    assert {(t.string, t.fret) for t in tab} == {(5, 3), (4, 2), (3, 0), (2, 1), (1, 0)}


def test_chord_no_string_conflict_even_with_unison():
    # 同一ピッチ E4 x2 でも別弦（1弦開放 / 2弦5F）に割当される
    notes = [_note(0.0, 64), _note(0.01, 64)]
    tab = assign_fingering(notes)
    assert len(tab) == 2
    assert tab[0].string != tab[1].string
    assert all(_played_pitch(t) == 64 for t in tab)


# --- (c) 高音ポジション連続でポジション移動が最小化される ---


def test_high_position_run_minimizes_position_shifts():
    # A5(81) は 1弦17F がほぼ唯一の現実解。E5(76) は 1弦12F / 2弦17F / 3弦21F。
    # ポジション移動最小化なら E5 は 2弦17F（手は 17F に留まる）を選ぶはず。
    notes = [
        _note(0.0, 81),
        _note(0.3, 76),
        _note(0.6, 81),
        _note(0.9, 76),
    ]
    tab = assign_fingering(notes)

    assert [t.midi_pitch for t in tab] == [81, 76, 81, 76]
    assert [(t.string, t.fret) for t in tab] == [(1, 17), (2, 17), (1, 17), (2, 17)]

    # 押弦ポジションの移動が起きていない
    positions = [t.fret for t in tab]
    shifts = [abs(a - b) for a, b in zip(positions, positions[1:])]
    assert max(shifts) == 0


# --- 音域外ノートのクランプ ---


def test_out_of_range_pitch_is_clamped_with_warning():
    with pytest.warns(UserWarning, match="clamped"):
        tab = assign_fingering([_note(0.0, 30)])  # E2 より低い
    assert len(tab) == 1
    assert MIN_PITCH <= tab[0].midi_pitch <= MAX_PITCH
    assert tab[0].midi_pitch % 12 == 30 % 12  # オクターブ移調

    with pytest.warns(UserWarning, match="clamped"):
        tab = assign_fingering([_note(0.0, 100)])  # 上限超
    assert MIN_PITCH <= tab[0].midi_pitch <= MAX_PITCH
    assert tab[0].midi_pitch % 12 == 100 % 12


def test_all_output_within_playable_range():
    notes = [_note(i * 0.2, p) for i, p in enumerate(range(40, 87))]
    tab = assign_fingering(notes)
    assert len(tab) == len(notes)
    for t in tab:
        assert 1 <= t.string <= 6
        assert 0 <= t.fret <= MAX_FRET
        assert _played_pitch(t) == t.midi_pitch


# --- (d) tab.json 往復 ---


def test_tab_json_roundtrip(tmp_path):
    tab = [
        TabNote(onset_sec=0.5, duration_sec=0.25, string=5, fret=3, midi_pitch=48),
        TabNote(onset_sec=0.0, duration_sec=0.5, string=1, fret=0, midi_pitch=64),
    ]
    path = tmp_path / "tab.json"
    save_tab(tab, path)
    loaded = load_tab(path)
    # save_tab は onset → 弦番号順にソートして書く
    assert loaded == [tab[1], tab[0]]


def test_load_tab_rejects_unknown_schema(tmp_path):
    path = tmp_path / "tab.json"
    path.write_text('{"schema": 99, "tab": []}')
    with pytest.raises(ValueError, match="schema"):
        load_tab(path)


def test_assign_fingering_empty():
    assert assign_fingering([]) == []


def test_unassignable_chord_note_dropped_with_warning():
    # MIDI 86 は 1弦22F のみ。同時に 2 音は割当不能 → 1 音を警告付きで除外
    notes = [_note(0.0, 86), _note(0.01, 86)]
    with pytest.warns(UserWarning, match="dropped unassignable"):
        tab = assign_fingering(notes)
    assert len(tab) == 1
    assert (tab[0].string, tab[0].fret) == (1, 22)
