"""MIDI レンダラ（tab/render_midi.py）のテスト。

生成した SMF バイナリをテスト内の自前パーサで解析し、note on/off の
pitch・絶対 tick・velocity が入力と一致することを検証する。
"""

import pytest

from guitartab.cli import main
from guitartab.tab.fingering import TabNote
from guitartab.tab.render_midi import (
    DEFAULT_TICKS_PER_BEAT,
    DEFAULT_VELOCITY,
    midi_velocity,
    render_midi,
    save_midi,
)
from guitartab.transcribe.base import NoteEvent, save_notes


def _note(onset: float, offset: float, pitch: int, velocity: float = 1.0) -> NoteEvent:
    return NoteEvent(
        onset_sec=onset, offset_sec=offset, midi_pitch=pitch, velocity=velocity
    )


# --- 検証用の最小 SMF パーサ ---


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    while True:
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos


def _parse_smf(data: bytes) -> dict:
    """SMF を解析して header 情報と note イベント列（絶対 tick）を返す。"""
    assert data[:4] == b"MThd"
    assert int.from_bytes(data[4:8], "big") == 6
    fmt = int.from_bytes(data[8:10], "big")
    ntrks = int.from_bytes(data[10:12], "big")
    division = int.from_bytes(data[12:14], "big")
    assert data[14:18] == b"MTrk"
    track_len = int.from_bytes(data[18:22], "big")
    track = data[22 : 22 + track_len]
    assert 22 + track_len == len(data), "トラック後に余分なバイトがある"

    notes = []  # (abs_tick, "on"|"off", pitch, velocity)
    tempo = None
    end_of_track = False
    pos, tick, running = 0, 0, None
    while pos < len(track):
        assert not end_of_track, "end of track 後にイベントがある"
        delta, pos = _read_vlq(track, pos)
        tick += delta
        status = track[pos]
        if status & 0x80:
            pos += 1
            if status < 0xF0:
                running = status
        else:
            assert running is not None
            status = running
        if status == 0xFF:  # meta
            meta_type = track[pos]
            length, pos = _read_vlq(track, pos + 1)
            payload = track[pos : pos + length]
            pos += length
            if meta_type == 0x51:
                tempo = int.from_bytes(payload, "big")
            elif meta_type == 0x2F:
                end_of_track = True
        elif status & 0xF0 == 0x90:
            pitch, vel = track[pos], track[pos + 1]
            pos += 2
            notes.append((tick, "on" if vel > 0 else "off", pitch, vel))
        elif status & 0xF0 == 0x80:
            pitch, vel = track[pos], track[pos + 1]
            pos += 2
            notes.append((tick, "off", pitch, vel))
        elif status & 0xF0 in (0xC0, 0xD0):  # program change / channel pressure
            pos += 1
        else:  # その他の 2 バイトメッセージ
            pos += 2
    assert end_of_track, "end of track がない"
    return {
        "format": fmt,
        "ntrks": ntrks,
        "division": division,
        "tempo": tempo,
        "notes": notes,
    }


def _sec_to_tick(sec: float, bpm: float = 120.0) -> int:
    return round(sec * DEFAULT_TICKS_PER_BEAT * bpm / 60.0)


# --- (a) note on/off の pitch・時刻・velocity が入力と一致 ---


def test_events_match_input():
    notes = [
        _note(0.0, 0.5, 60, velocity=1.0),
        _note(0.5, 1.0, 64, velocity=0.5),
        _note(1.25, 2.0, 43, velocity=0.25),
    ]
    parsed = _parse_smf(render_midi(notes))

    assert parsed["format"] == 0
    assert parsed["ntrks"] == 1
    assert parsed["division"] == DEFAULT_TICKS_PER_BEAT
    assert parsed["tempo"] == 500_000  # 120 BPM

    ons = [e for e in parsed["notes"] if e[1] == "on"]
    offs = [e for e in parsed["notes"] if e[1] == "off"]
    assert [(t, p, v) for t, _, p, v in ons] == [
        (_sec_to_tick(n.onset_sec), n.midi_pitch, midi_velocity(n.velocity))
        for n in notes
    ]
    assert [(t, p) for t, _, p, _ in offs] == [
        (_sec_to_tick(n.offset_sec), n.midi_pitch) for n in notes
    ]


def test_velocity_mapping():
    assert midi_velocity(1.0) == 127
    assert midi_velocity(0.5) == 64
    # 0 / None / 負値はデフォルト 100
    assert midi_velocity(0.0) == DEFAULT_VELOCITY
    assert midi_velocity(None) == DEFAULT_VELOCITY
    assert midi_velocity(-0.1) == DEFAULT_VELOCITY
    # 極小値でも 1 未満にならない
    assert midi_velocity(1e-6) == 1


def test_tempo_option_changes_tick_scale():
    notes = [_note(1.0, 2.0, 60)]
    parsed = _parse_smf(render_midi(notes, tempo_bpm=60.0))
    assert parsed["tempo"] == 1_000_000  # 60 BPM
    on = [e for e in parsed["notes"] if e[1] == "on"][0]
    assert on[0] == DEFAULT_TICKS_PER_BEAT  # 1 秒 = 1 拍 = 480 tick


# --- (b) 空入力 ---


def test_empty_input_is_valid_midi():
    parsed = _parse_smf(render_midi([]))
    assert parsed["format"] == 0
    assert parsed["ntrks"] == 1
    assert parsed["tempo"] == 500_000
    assert parsed["notes"] == []


# --- (c) 同時発音 ---


def test_chord_notes_share_tick():
    # C メジャートライアド: 同一 onset の 3 音が同一 tick の note on になる
    chord = [_note(0.25, 0.75, p) for p in (48, 52, 55)]
    parsed = _parse_smf(render_midi(chord))
    ons = [e for e in parsed["notes"] if e[1] == "on"]
    offs = [e for e in parsed["notes"] if e[1] == "off"]
    assert {t for t, _, _, _ in ons} == {_sec_to_tick(0.25)}
    assert {t for t, _, _, _ in offs} == {_sec_to_tick(0.75)}
    assert sorted(p for _, _, p, _ in ons) == [48, 52, 55]


def test_repeated_pitch_off_before_next_on():
    # 同音連打: 境界 tick では note off が note on より先に来る
    notes = [_note(0.0, 0.5, 60), _note(0.5, 1.0, 60)]
    parsed = _parse_smf(render_midi(notes))
    boundary = _sec_to_tick(0.5)
    kinds = [kind for t, kind, _, _ in parsed["notes"] if t == boundary]
    assert kinds == ["off", "on"]


# --- 縮退ケース・入力バリエーション ---


def test_zero_duration_gets_minimum_one_tick():
    parsed = _parse_smf(render_midi([_note(1.0, 1.0, 60)]))
    on = [e for e in parsed["notes"] if e[1] == "on"][0]
    off = [e for e in parsed["notes"] if e[1] == "off"][0]
    assert off[0] == on[0] + 1


def test_tabnote_input_uses_default_velocity():
    tab = [TabNote(onset_sec=0.5, duration_sec=0.5, string=5, fret=3, midi_pitch=48)]
    parsed = _parse_smf(render_midi(tab))
    ons = [e for e in parsed["notes"] if e[1] == "on"]
    offs = [e for e in parsed["notes"] if e[1] == "off"]
    assert ons == [(_sec_to_tick(0.5), "on", 48, DEFAULT_VELOCITY)]
    assert [(t, p) for t, _, p, _ in offs] == [(_sec_to_tick(1.0), 48)]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        render_midi([_note(0.0, 0.5, 128)])  # ピッチ範囲外
    with pytest.raises(ValueError):
        render_midi([_note(-0.1, 0.5, 60)])  # 負の onset
    with pytest.raises(ValueError):
        render_midi([], tempo_bpm=0.0)


# --- save_midi / CLI 統合 ---


def test_save_midi_roundtrip(tmp_path):
    notes = [_note(0.0, 0.5, 60), _note(0.5, 1.0, 64)]
    path = save_midi(notes, tmp_path / "sub" / "out.mid")
    assert path.exists()
    parsed = _parse_smf(path.read_bytes())
    assert len([e for e in parsed["notes"] if e[1] == "on"]) == 2


def test_cli_midi_subcommand(tmp_path, capsys):
    notes = [_note(0.0, 0.4, 48), _note(0.5, 0.9, 64, velocity=0.5)]
    notes_path = tmp_path / "notes.json"
    save_notes(notes, notes_path)

    assert main(["midi", str(notes_path)]) == 0

    midi_path = tmp_path / "output.mid"
    assert midi_path.exists()
    assert str(midi_path) in capsys.readouterr().out
    parsed = _parse_smf(midi_path.read_bytes())
    ons = [e for e in parsed["notes"] if e[1] == "on"]
    assert [(p, v) for _, _, p, v in ons] == [(48, 127), (64, 64)]


def test_cli_midi_refuses_eval_data_output(tmp_path):
    notes_path = tmp_path / "notes.json"
    save_notes([_note(0.0, 0.4, 48)], notes_path)
    fake_eval = tmp_path / "eval_data" / "out.mid"
    with pytest.raises(SystemExit, match="eval_data"):
        main(["midi", str(notes_path), "--out", str(fake_eval)])
