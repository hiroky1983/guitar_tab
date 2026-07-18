"""レンダラの rhythm.json 供給時モードとフォールバック不変のテスト。

- MusicXML: divisions=12・実テンポ表示・tick 配置・3 連の time-modification、
  rhythm 未供給時は従来の 120BPM 近似（divisions=4）のまま。
- MIDI: tempo meta とオンセット tick が rhythm に従う、未供給時は従来動作。
- pipeline: stage_quantize のキャッシュ、CLI quantize サブコマンド。
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from guitartab.cli import main
from guitartab.pipeline import stage_quantize
from guitartab.rhythm.schema import Rhythm, RhythmNote, TempoPoint, load_rhythm
from guitartab.tab.fingering import STANDARD_TUNING_MIDI, TabNote
from guitartab.tab.render_midi import render_midi
from guitartab.tab.render_musicxml import render_musicxml
from guitartab.transcribe.base import NoteEvent, save_notes

from test_render_midi import _parse_smf


def _tn(onset: float, dur: float, string: int, fret: int) -> TabNote:
    return TabNote(
        onset_sec=onset,
        duration_sec=dur,
        string=string,
        fret=fret,
        midi_pitch=STANDARD_TUNING_MIDI[string] + fret,
    )


def _rhythm(bpm: float, notes: list[RhythmNote], origin: float = 0.0) -> Rhythm:
    return Rhythm(
        tempo_bpm=bpm,
        tempo_map=[TempoPoint(time_sec=origin, bpm=bpm)],
        notes=notes,
        estimator="test",
    )


def _note_rows(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows = []
    for note_el in root.findall("./part/measure/note"):
        tm = note_el.find("time-modification")
        rows.append(
            {
                "rest": note_el.find("rest") is not None,
                "duration": int(note_el.findtext("duration")),
                "type": note_el.findtext("type"),
                "time_mod": None
                if tm is None
                else (
                    int(tm.findtext("actual-notes")),
                    int(tm.findtext("normal-notes")),
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# MusicXML
# ---------------------------------------------------------------------------


def test_musicxml_fallback_unchanged_without_rhythm():
    tab = [_tn(0.0, 0.25, 1, 0), _tn(0.5, 0.25, 2, 3)]
    xml_default = render_musicxml(tab)
    xml_none = render_musicxml(tab, rhythm=None)
    assert xml_default == xml_none
    root = ET.fromstring(xml_default)
    assert root.findtext("./part/measure/attributes/divisions") == "4"
    metronome = root.find("./part/measure/direction/direction-type/metronome")
    assert metronome.findtext("per-minute") == "120"


def test_musicxml_rhythm_mode_uses_estimated_tempo_and_ticks():
    # 96 BPM: 1 拍 = 0.625 秒、1 tick = 0.625/12 秒
    tick_sec = 0.625 / 12
    tab = [
        _tn(0 * tick_sec, 6 * tick_sec, 1, 0),   # tick 0、8 分
        _tn(9 * tick_sec, 3 * tick_sec, 2, 1),   # tick 9（付点 8 分の位置の 16 分）
    ]
    rhythm = _rhythm(
        96.0,
        [
            RhythmNote(onset_tick=0, duration_ticks=6, deviation_sec=0.0),
            RhythmNote(onset_tick=9, duration_ticks=3, deviation_sec=0.0),
        ],
    )
    xml = render_musicxml(tab, rhythm)
    root = ET.fromstring(xml)
    assert root.findtext("./part/measure/attributes/divisions") == "12"
    metronome = root.find("./part/measure/direction/direction-type/metronome")
    assert metronome.findtext("per-minute") == "96"
    rows = _note_rows(xml)
    # tick0 に 8 分（6 tick）→ 休符 3 tick（16 分）→ tick9 に 16 分（3 tick）
    assert [r["duration"] for r in rows[:3]] == [6, 3, 3]
    assert rows[0]["rest"] is False and rows[0]["type"] == "eighth"
    assert rows[1]["rest"] is True
    assert rows[2]["rest"] is False and rows[2]["type"] == "16th"
    # 小節合計 48 tick
    total = sum(r["duration"] for r in rows)
    assert total == 48


def test_musicxml_rhythm_mode_triplets_use_time_modification():
    tick_sec = 0.5 / 12  # 120 BPM
    tab = [
        _tn(0.0, 4 * tick_sec, 1, 0),
        _tn(4 * tick_sec, 4 * tick_sec, 1, 2),
        _tn(8 * tick_sec, 4 * tick_sec, 1, 3),
    ]
    rhythm = _rhythm(
        120.0,
        [
            RhythmNote(onset_tick=0, duration_ticks=4, deviation_sec=0.0),
            RhythmNote(onset_tick=4, duration_ticks=4, deviation_sec=0.0),
            RhythmNote(onset_tick=8, duration_ticks=4, deviation_sec=0.0),
        ],
    )
    rows = [r for r in _note_rows(render_musicxml(tab, rhythm)) if not r["rest"]]
    assert [r["duration"] for r in rows] == [4, 4, 4]
    assert all(r["type"] == "eighth" and r["time_mod"] == (3, 2) for r in rows)


def test_musicxml_rhythm_mode_fractional_tempo_display():
    rhythm = _rhythm(
        117.45, [RhythmNote(onset_tick=0, duration_ticks=12, deviation_sec=0.0)]
    )
    xml = render_musicxml([_tn(0.0, 0.5, 1, 0)], rhythm)
    root = ET.fromstring(xml)
    metronome = root.find("./part/measure/direction/direction-type/metronome")
    assert metronome.findtext("per-minute") == "117.5"  # 0.1 単位へ丸め


def test_musicxml_rhythm_mode_unmatched_note_falls_back_to_16th_grid():
    # rhythm に対応のない onset（照合失敗）でも例外にせず配置される
    rhythm = _rhythm(
        120.0, [RhythmNote(onset_tick=0, duration_ticks=3, deviation_sec=0.0)]
    )
    tab = [_tn(0.0, 0.125, 1, 0), _tn(1.0, 0.125, 2, 3)]  # 2 音目は照合不能
    rows = [r for r in _note_rows(render_musicxml(tab, rhythm)) if not r["rest"]]
    assert len(rows) == 2


def test_musicxml_rhythm_mode_rejects_wrong_divisions():
    rhythm = _rhythm(
        120.0, [RhythmNote(onset_tick=0, duration_ticks=3, deviation_sec=0.0)]
    )
    rhythm.divisions_per_quarter = 24
    with pytest.raises(ValueError, match="divisions_per_quarter"):
        render_musicxml([_tn(0.0, 0.125, 1, 0)], rhythm)


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------


def test_midi_rhythm_mode_tempo_and_ticks():
    # 96 BPM、tick 9 の音 → MIDI tick = 9 × (480/12) = 360
    tick_sec = 0.625 / 12
    notes = [
        NoteEvent(onset_sec=0.0, offset_sec=6 * tick_sec, midi_pitch=64),
        NoteEvent(onset_sec=9 * tick_sec, offset_sec=12 * tick_sec, midi_pitch=59),
    ]
    rhythm = _rhythm(
        96.0,
        [
            RhythmNote(onset_tick=0, duration_ticks=6, deviation_sec=0.0),
            RhythmNote(onset_tick=9, duration_ticks=3, deviation_sec=0.0),
        ],
    )
    parsed = _parse_smf(render_midi(notes, rhythm=rhythm))
    assert parsed["tempo"] == round(60_000_000 / 96.0)
    ons = [(t, p) for t, kind, p, _v in parsed["notes"] if kind == "on"]
    assert (0, 64) in ons
    assert (360, 59) in ons
    # duration 6 tick → off at 240
    offs = [(t, p) for t, kind, p, _v in parsed["notes"] if kind == "off"]
    assert (240, 64) in offs


def test_midi_fallback_unchanged_without_rhythm():
    notes = [NoteEvent(onset_sec=0.25, offset_sec=0.5, midi_pitch=60)]
    assert render_midi(notes) == render_midi(notes, rhythm=None)
    parsed = _parse_smf(render_midi(notes))
    assert parsed["tempo"] == 500000  # 120 BPM


# ---------------------------------------------------------------------------
# pipeline / CLI
# ---------------------------------------------------------------------------


def _write_notes(path: Path) -> None:
    beat = 0.6  # 100 BPM
    notes = []
    for k in range(16):
        for frac in (0.0, 0.25, 0.5, 0.75):
            onset = (k + frac) * beat
            notes.append(
                NoteEvent(onset_sec=onset, offset_sec=onset + 0.15, midi_pitch=60)
            )
    save_notes(notes, path)


def test_stage_quantize_writes_and_caches(tmp_path, capsys):
    notes_path = tmp_path / "notes.json"
    _write_notes(notes_path)
    rhythm_path = tmp_path / "rhythm.json"
    rhythm = stage_quantize(notes_path, rhythm_path)
    assert rhythm_path.exists()
    assert len(rhythm.notes) == 64
    mtime = rhythm_path.stat().st_mtime_ns
    cached = stage_quantize(notes_path, rhythm_path)
    assert rhythm_path.stat().st_mtime_ns == mtime  # 再生成されない
    assert cached == load_rhythm(rhythm_path)


def test_cli_quantize_subcommand(tmp_path, capsys):
    notes_path = tmp_path / "notes.json"
    _write_notes(notes_path)
    assert main(["quantize", str(notes_path)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.endswith("rhythm.json")
    rhythm = load_rhythm(Path(out))
    assert len(rhythm.notes) == 64


def test_cli_quantize_refuses_eval_data_output(tmp_path):
    notes_path = tmp_path / "notes.json"
    _write_notes(notes_path)
    repo_eval = Path(__file__).parent.parent / "eval_data" / "x" / "rhythm.json"
    with pytest.raises(SystemExit, match="eval_data"):
        main(["quantize", str(notes_path), "--out", str(repo_eval)])


def test_cli_midi_and_musicxml_accept_rhythm_flag(tmp_path, capsys):
    notes_path = tmp_path / "notes.json"
    _write_notes(notes_path)
    assert main(["quantize", str(notes_path)]) == 0
    capsys.readouterr()
    assert main(["midi", str(notes_path), "--rhythm", str(tmp_path / "rhythm.json")]) == 0
    capsys.readouterr()
    # tab.json を作って musicxml --rhythm
    assert main(["tab", str(notes_path)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "musicxml",
                str(tmp_path / "tab.json"),
                "--rhythm",
                str(tmp_path / "rhythm.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out.strip()
    xml = Path(out).read_text()
    assert "<divisions>12</divisions>" in xml
