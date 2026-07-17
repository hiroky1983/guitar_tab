"""MusicXML レンダラ（tab/render_musicxml.py）のテスト。

生成した MusicXML を xml.etree で解析し、well-formed であること・
必須要素（TAB クレフ・6 線・string/fret・chord マーク）・
小節の duration 合計が 4/4 ぴったりであることを検証する。
MuseScore 本体での開封確認は CI ではできないため、構造検証で代替する。
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from guitartab.cli import main
from guitartab.tab.fingering import STANDARD_TUNING_MIDI, TabNote, load_tab, save_tab
from guitartab.tab.render_musicxml import (
    GRID_SEC,
    UNITS_PER_MEASURE,
    render_musicxml,
    save_musicxml,
)

REAL_TAB_JSON = Path(__file__).parent.parent / "work" / "wr7xTGTG-Mo" / "tab.json"


def _tn(onset: float, dur: float, string: int, fret: int) -> TabNote:
    return TabNote(
        onset_sec=onset,
        duration_sec=dur,
        string=string,
        fret=fret,
        midi_pitch=STANDARD_TUNING_MIDI[string] + fret,
    )


def _parse(xml_text: str) -> ET.Element:
    """well-formed 検証を兼ねて score-partwise ルートを返す。"""
    root = ET.fromstring(xml_text)
    assert root.tag == "score-partwise"
    assert root.get("version") == "4.0"
    return root


def _measures(root: ET.Element) -> list[ET.Element]:
    return root.findall("./part/measure")


def _note_row(note_el: ET.Element) -> dict:
    """<note> 要素をテストしやすい dict にする。"""
    technical = note_el.find("./notations/technical")
    return {
        "chord": note_el.find("chord") is not None,
        "rest": note_el.find("rest") is not None,
        "duration": int(note_el.findtext("duration")),
        "type": note_el.findtext("type"),
        "string": None if technical is None else int(technical.findtext("string")),
        "fret": None if technical is None else int(technical.findtext("fret")),
    }


def _assert_measures_fill_4_4(root: ET.Element) -> None:
    """全小節で duration 合計（chord の 2 音目以降を除く）が 16 であること。"""
    for measure in _measures(root):
        total = sum(
            row["duration"]
            for row in map(_note_row, measure.findall("note"))
            if not row["chord"]
        )
        assert total == UNITS_PER_MEASURE, f"measure {measure.get('number')}: {total}"


# --- (a) 小さな固定入力のゴールデン構造 ---


def test_golden_structure():
    tab = [
        _tn(0.0, 0.5, 6, 3),  # unit 0, 4分
        _tn(0.5, 0.25, 5, 0),  # unit 4, 8分
        _tn(2.5, 0.5, 1, 5),  # unit 20 → 第2小節
    ]
    root = _parse(render_musicxml(tab))
    measures = _measures(root)
    assert len(measures) == 2
    assert [m.get("number") for m in measures] == ["1", "2"]
    _assert_measures_fill_4_4(root)

    rows1 = [_note_row(n) for n in measures[0].findall("note")]
    # 4分音符(6弦3F) + 8分音符(5弦0F) + 残り10単位の休符(2分+8分)
    assert [(r["rest"], r["duration"], r["type"]) for r in rows1] == [
        (False, 4, "quarter"),
        (False, 2, "eighth"),
        (True, 8, "half"),
        (True, 2, "eighth"),
    ]
    assert [(r["string"], r["fret"]) for r in rows1[:2]] == [(6, 3), (5, 0)]

    rows2 = [_note_row(n) for n in measures[1].findall("note")]
    # 4分休符 + 4分音符(1弦5F) + 2分休符
    assert [(r["rest"], r["duration"]) for r in rows2] == [
        (True, 4),
        (False, 4),
        (True, 8),
    ]
    assert (rows2[1]["string"], rows2[1]["fret"]) == (1, 5)


def test_pitch_elements():
    # 5弦3F = C3 (midi 48)、4弦1F = D#3 (midi 51, シャープ表記)
    root = _parse(render_musicxml([_tn(0.0, 0.5, 5, 3), _tn(0.5, 0.5, 4, 1)]))
    pitches = root.findall("./part/measure/note/pitch")
    assert [
        (p.findtext("step"), p.findtext("alter"), p.findtext("octave")) for p in pitches
    ] == [("C", None, "3"), ("D", "1", "3")]


def test_first_measure_attributes():
    root = _parse(render_musicxml([_tn(0.0, 0.5, 6, 0)]))
    attrs = root.find("./part/measure[@number='1']/attributes")
    assert attrs is not None
    assert attrs.findtext("divisions") == "4"
    assert attrs.findtext("time/beats") == "4"
    assert attrs.findtext("time/beat-type") == "4"
    assert attrs.findtext("clef/sign") == "TAB"
    details = attrs.find("staff-details")
    assert details.findtext("staff-lines") == "6"
    tunings = details.findall("staff-tuning")
    # line 1(最低音側)=E2 〜 line 6=E4 の EADGBE
    assert [
        (t.get("line"), t.findtext("tuning-step"), t.findtext("tuning-octave"))
        for t in tunings
    ] == [
        ("1", "E", "2"),
        ("2", "A", "2"),
        ("3", "D", "3"),
        ("4", "G", "3"),
        ("5", "B", "3"),
        ("6", "E", "4"),
    ]


# --- (b) 空入力 ---


def test_empty_input():
    root = _parse(render_musicxml([]))
    measures = _measures(root)
    assert len(measures) == 1
    rows = [_note_row(n) for n in measures[0].findall("note")]
    assert len(rows) == 1
    assert rows[0]["rest"] and rows[0]["duration"] == UNITS_PER_MEASURE
    # 全休符は <rest measure="yes"/>
    assert measures[0].find("note/rest").get("measure") == "yes"
    # 空でも attributes（TAB 譜表定義）は出力する
    assert measures[0].find("attributes/clef") is not None


# --- (c) 和音 ---


def test_chord_marks():
    # 同一グリッドの 3 音: 先頭に <chord/> なし、2音目以降にあり。弦順に並ぶ。
    tab = [_tn(1.0, 0.5, 3, 2), _tn(1.0, 0.5, 5, 3), _tn(1.02, 0.5, 4, 2)]
    root = _parse(render_musicxml(tab))
    _assert_measures_fill_4_4(root)
    rows = [r for r in map(_note_row, root.findall("./part/measure/note")) if not r["rest"]]
    assert [(r["chord"], r["string"], r["fret"]) for r in rows] == [
        (False, 3, 2),
        (True, 4, 2),
        (True, 5, 3),
    ]
    # 和音は duration を共有する
    assert {r["duration"] for r in rows} == {4}


def test_chord_exact_duplicate_is_deduped():
    tab = [_tn(0.0, 0.5, 6, 3), _tn(0.01, 0.5, 6, 3)]
    root = _parse(render_musicxml(tab))
    rows = [r for r in map(_note_row, root.findall("./part/measure/note")) if not r["rest"]]
    assert len(rows) == 1


# --- 音価の縮退規則 ---


def test_note_truncated_at_measure_boundary():
    # unit 14 開始・8 単位 → 小節末尾で 2 単位(8分)に切り詰め。第2小節は作らない
    root = _parse(render_musicxml([_tn(1.75, 1.0, 2, 1)]))
    measures = _measures(root)
    assert len(measures) == 1
    _assert_measures_fill_4_4(root)
    rows = [r for r in map(_note_row, measures[0].findall("note")) if not r["rest"]]
    assert [(r["duration"], r["type"]) for r in rows] == [(2, "eighth")]


def test_note_truncated_by_next_onset():
    # 8 単位の音の 2 単位後に次の音 → 2 単位に切り詰め（単声部・backup なし）
    tab = [_tn(0.0, 1.0, 6, 0), _tn(0.25, 0.25, 6, 2)]
    root = _parse(render_musicxml(tab))
    _assert_measures_fill_4_4(root)
    rows = [r for r in map(_note_row, root.findall("./part/measure/note")) if not r["rest"]]
    assert [(r["duration"], r["fret"]) for r in rows] == [(2, 0), (2, 2)]


def test_unrepresentable_duration_rounds_down():
    # 5 単位は単一音価で表せない → 4分(4単位)に切り詰め、残りは休符
    root = _parse(render_musicxml([_tn(0.0, 0.625, 6, 0)]))
    _assert_measures_fill_4_4(root)
    rows = [_note_row(n) for n in root.findall("./part/measure/note")]
    assert (rows[0]["duration"], rows[0]["type"]) == (4, "quarter")
    assert rows[1]["rest"]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        render_musicxml([TabNote(-0.1, 0.5, 6, 0, 40)])  # 負の onset
    with pytest.raises(ValueError):
        render_musicxml([TabNote(0.0, 0.5, 7, 0, 40)])  # 弦番号範囲外
    with pytest.raises(ValueError):
        render_musicxml([TabNote(0.0, 0.5, 6, -1, 40)])  # 負のフレット
    with pytest.raises(ValueError):
        render_musicxml([TabNote(0.0, 0.5, 6, 100, 140)])  # MIDI 範囲外


# --- save_musicxml / CLI 統合 ---


def test_save_musicxml_roundtrip(tmp_path):
    path = save_musicxml([_tn(0.0, 0.5, 6, 3)], tmp_path / "sub" / "out.musicxml")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<!DOCTYPE score-partwise" in text
    _parse(text)


def test_cli_musicxml_subcommand(tmp_path, capsys):
    tab_path = tmp_path / "tab.json"
    save_tab([_tn(0.0, 0.5, 6, 3), _tn(0.5, 0.5, 5, 0)], tab_path)

    assert main(["musicxml", str(tab_path)]) == 0

    out_path = tmp_path / "output.musicxml"
    assert out_path.exists()
    assert str(out_path) in capsys.readouterr().out
    root = _parse(out_path.read_text(encoding="utf-8"))
    _assert_measures_fill_4_4(root)


def test_cli_musicxml_refuses_eval_data_output(tmp_path):
    tab_path = tmp_path / "tab.json"
    save_tab([_tn(0.0, 0.5, 6, 3)], tab_path)
    fake_eval = tmp_path / "eval_data" / "out.musicxml"
    with pytest.raises(SystemExit, match="eval_data"):
        main(["musicxml", str(tab_path), "--out", str(fake_eval)])


# --- (d) 実データ ---


@pytest.mark.skipif(not REAL_TAB_JSON.exists(), reason="work/wr7xTGTG-Mo/tab.json がない")
def test_real_tab_json_renders_well_formed():
    tab = load_tab(REAL_TAB_JSON)
    root = _parse(render_musicxml(tab))
    _assert_measures_fill_4_4(root)
    rows = [r for r in map(_note_row, root.findall("./part/measure/note")) if not r["rest"]]
    assert rows, "実データから音符が 1 つも出力されていない"
    assert all(1 <= r["string"] <= 6 and r["fret"] >= 0 for r in rows)
    assert all(r["type"] is not None for r in rows)
