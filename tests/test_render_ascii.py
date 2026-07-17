"""ASCII tab レンダラ（tab/render_ascii.py）のテスト。"""

import pytest

from guitartab.tab.fingering import TabNote
from guitartab.tab.render_ascii import render_ascii


def _tab(onset: float, string: int, fret: int) -> TabNote:
    from guitartab.tab.fingering import STANDARD_TUNING_MIDI

    return TabNote(
        onset_sec=onset,
        duration_sec=0.25,
        string=string,
        fret=fret,
        midi_pitch=STANDARD_TUNING_MIDI[string] + fret,
    )


# --- (e) ゴールデンテスト ---


def test_golden_small_input():
    tab = [
        _tab(0.0, 5, 3),
        _tab(0.5, 4, 2),
        _tab(1.0, 3, 0),
        _tab(1.5, 1, 10),  # 2 桁フレット
    ]
    expected = "\n".join(
        [
            "e|------10-|",
            "B|---------|",
            "G|----0----|",
            "D|--2------|",
            "A|3--------|",
            "E|---------|",
        ]
    )
    assert render_ascii(tab, time_step_sec=0.5) == expected


def test_golden_chord_in_same_column():
    # C メジャー x32010: 同時発音は同一カラム
    chord = [(5, 3), (4, 2), (3, 0), (2, 1), (1, 0)]
    tab = [_tab(0.0, s, f) for s, f in chord]
    expected = "\n".join(
        [
            "e|0-|",
            "B|1-|",
            "G|0-|",
            "D|2-|",
            "A|3-|",
            "E|--|",
        ]
    )
    assert render_ascii(tab, time_step_sec=0.5) == expected


# --- 桁ずれ・折り返し ---


def test_two_digit_frets_keep_lines_aligned():
    tab = [_tab(0.0, 1, 12), _tab(0.0, 2, 5), _tab(0.25, 1, 7)]
    lines = render_ascii(tab, time_step_sec=0.25).splitlines()
    assert len(lines) == 6
    assert len({len(line) for line in lines}) == 1, "行の長さが揃っていない"


def test_wraps_at_line_width():
    tab = [_tab(i * 0.125, 1, 0) for i in range(50)]  # 50 カラム x 2 桁 = 100 桁分
    out = render_ascii(tab, time_step_sec=0.125, line_width=80)
    blocks = out.split("\n\n")
    assert len(blocks) == 2
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) == 6
        for line in lines:
            assert len(line) <= 80


def test_same_string_collision_keeps_first():
    # 同一弦・同一カラムの衝突は先勝ち
    tab = [_tab(0.0, 1, 3), _tab(0.01, 1, 7)]
    out = render_ascii(tab, time_step_sec=0.5)
    assert "3" in out
    assert "7" not in out


def test_empty_tab():
    assert render_ascii([]) == "(no notes)"


def test_invalid_time_step():
    with pytest.raises(ValueError):
        render_ascii([_tab(0.0, 1, 0)], time_step_sec=0.0)
