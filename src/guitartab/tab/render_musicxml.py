"""MusicXML レンダラ（score-partwise 4.0、TAB 単段）。

TabNote 列（tab.json）を MuseScore / Guitar Pro で開ける MusicXML に書き出す。
第三出力（docs/DESIGN.md「出力」参照）。stdlib の xml.etree のみで組み立てる。

- TAB 表記: ``<clef sign="TAB"/>`` + 6 線 ``<staff-details>``（EADGBE の
  staff-tuning）を持つ 1 パート構成で、各音符に
  ``<notations><technical><string>/<fret>`` を付ける。五線+TAB の 2 段構成では
  なく TAB 単段（MuseScore で開けて弦・フレットが正しく表示されることを最優先）。
- リズムの 2 モード:
  - **rhythm.json 供給時（M4a）**: quantize ステージが推定した実テンポと
    per-note tick（quarter = 12 divisions、16 分 + 3 連）をそのまま使う
    （レンダラ内では再量子化しない）。3 連系の音価は ``<time-modification>``
    3:2 で表現する。tab.json の音符は onset_sec（rhythm 側は格子時刻 +
    deviation で復元）で rhythm.json と照合するため、fingering の除外で
    音符数が一致しなくても頑健。照合できない音符は従来近似と同じ
    16 分丸め（ただし推定テンポの格子）で配置する。
  - **未供給時（従来どおりの表示用近似）**: 固定 120 BPM・4/4 とし、
    onset_sec を最も近い 16 分音符グリッド（0.125 秒）へ置き、duration も
    16 分単位に丸める。
  いずれも同一グリッドに落ちた複数音は和音（2 音目以降に ``<chord/>``）、
  隙間は休符で埋め、小節は 4/4 で機械的に区切る。
  **notes.json / tab.json のデータは一切変更しない。**
- 音価の縮退規則（すべて表示上の近似）:
  - 小節境界をまたぐ音は小節末尾で切り詰める（タイは張らない）。
  - 次の音の開始と重なる分は切り詰める（単声部で backup を使わないため）。
  - 単一音価で表現できない長さは表現できる最大の音価へ切り詰め、残りは休符になる。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from guitartab.rhythm.schema import Rhythm, lookup_note_by_onset
from guitartab.tab.fingering import TabNote

MUSICXML_VERSION = "4.0"
TEMPO_BPM = 120.0  # rhythm 未供給時の固定近似テンポ
BEATS_PER_MEASURE = 4  # 4/4 固定（拍子推定は M4c スコープ）
DIVISIONS_PER_QUARTER = 4  # 近似モード: 1 division = 16 分音符
UNITS_PER_MEASURE = BEATS_PER_MEASURE * DIVISIONS_PER_QUARTER  # 16
GRID_SEC = 60.0 / TEMPO_BPM / DIVISIONS_PER_QUARTER  # 0.125 秒 = 16 分音符

RHYTHM_DIVISIONS = 12  # rhythm モード: quarter = 12 tick（16 分 + 3 連）

# 単一の音価で表現できる長さ → (type, 付点数, time-modification or None)
# 近似モード（16 分単位、divisions=4）
_TYPE_BY_UNITS = {
    16: ("whole", 0, None),
    12: ("half", 1, None),
    8: ("half", 0, None),
    6: ("quarter", 1, None),
    4: ("quarter", 0, None),
    3: ("eighth", 1, None),
    2: ("eighth", 0, None),
    1: ("16th", 0, None),
}
# rhythm モード（tick 単位、divisions=12）。3 連系は 3:2 の time-modification。
_TYPE_BY_TICKS = {
    48: ("whole", 0, None),
    36: ("half", 1, None),
    24: ("half", 0, None),
    18: ("quarter", 1, None),
    12: ("quarter", 0, None),
    9: ("eighth", 1, None),
    6: ("eighth", 0, None),
    8: ("quarter", 0, (3, 2)),  # 3 連 4 分
    4: ("eighth", 0, (3, 2)),   # 3 連 8 分
    3: ("16th", 0, None),
    2: ("16th", 0, (3, 2)),     # 3 連 16 分
    1: ("32nd", 0, (3, 2)),     # 3 連 32 分（最小 tick）
}

# midi_pitch % 12 → (step, alter)。臨時記号はシャープ表記で統一。
_STEP_ALTER = [
    ("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0),
    ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0),
]

# staff-tuning: TAB の線番号（1 = 最低音側）→ 開放弦 (step, octave)。EADGBE。
_STAFF_TUNING = {1: ("E", 2), 2: ("A", 2), 3: ("D", 3), 4: ("G", 3), 5: ("B", 3), 6: ("E", 4)}

_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML '
    f'{MUSICXML_VERSION} Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
)


def _largest_representable(units: int, table: dict) -> int:
    """units 以下で単一音価として表現できる最大の長さを返す（units >= 1 前提）。"""
    for value in sorted(table, reverse=True):
        if value <= units:
            return value
    raise ValueError(f"units must be >= 1: {units}")


def _decompose(units: int, table: dict) -> list[int]:
    """休符用: units を表現できる音価の列（大きい順）に分解する。"""
    parts = []
    while units > 0:
        part = _largest_representable(units, table)
        parts.append(part)
        units -= part
    return parts


def _validate_note(note: TabNote) -> None:
    if note.onset_sec < 0:
        raise ValueError(f"onset_sec must be non-negative: {note.onset_sec}")
    if not 1 <= note.string <= 6:
        raise ValueError(f"string must be in [1, 6]: {note.string}")
    if note.fret < 0:
        raise ValueError(f"fret must be non-negative: {note.fret}")
    if not 0 <= note.midi_pitch <= 127:
        raise ValueError(f"midi_pitch out of MIDI range [0, 127]: {note.midi_pitch}")


def _group_events(
    positions: list[tuple[int, int, TabNote]],
) -> list[tuple[int, int, list[TabNote]]]:
    """(開始 unit, 長さ unit, note) 列を (開始, 長さ, 和音グループ) 列にする。

    - 同一開始 unit の音は和音グループ（長さは最長の構成音）。
    - 完全重複（同じ弦・同じフレット）は除去する。
    - 長さは「次のグループの開始」で切り詰める（単声部表現のため）。
    """
    groups: dict[int, list[TabNote]] = {}
    durations: dict[int, int] = {}
    for start, units, note in positions:
        group = groups.setdefault(start, [])
        if any(n.string == note.string and n.fret == note.fret for n in group):
            continue  # 量子化で潰れた完全重複
        group.append(note)
        durations[start] = max(durations.get(start, 0), units)

    events = []
    starts = sorted(groups)
    for i, start in enumerate(starts):
        units = durations[start]
        if i + 1 < len(starts):
            units = min(units, starts[i + 1] - start)
        notes = sorted(groups[start], key=lambda n: n.string)
        events.append((start, units, notes))
    return events


def _quantize(tab: Sequence[TabNote]) -> list[tuple[int, int, list[TabNote]]]:
    """近似モード: 固定 120BPM の 16 分グリッドへ量子化する。"""
    positions = []
    for note in tab:
        _validate_note(note)
        start = round(note.onset_sec / GRID_SEC)
        units = max(1, round(note.duration_sec / GRID_SEC))
        positions.append((start, units, note))
    return _group_events(positions)


def _quantize_with_rhythm(
    tab: Sequence[TabNote], rhythm: Rhythm
) -> list[tuple[int, int, list[TabNote]]]:
    """rhythm モード: rhythm.json の per-note tick をそのまま使う（再量子化しない）。

    onset_sec で照合できない音符（許容 5ms）のみ、推定テンポの 16 分格子への
    最近傍丸めでフォールバック配置する。
    """
    tick_sec = 60.0 / rhythm.tempo_bpm / rhythm.divisions_per_quarter
    origin = rhythm.tempo_map[0].time_sec
    positions = []
    for note in tab:
        _validate_note(note)
        matched = lookup_note_by_onset(rhythm, note.onset_sec)
        if matched is not None:
            start = matched.onset_tick
            units = matched.duration_ticks
        else:
            start = max(0, round((note.onset_sec - origin) / (tick_sec * 3)) * 3)
            units = max(1, round(note.duration_sec / tick_sec))
        positions.append((start, units, note))
    return _group_events(positions)


def _append_pitch(note_el: ET.Element, midi_pitch: int) -> None:
    step, alter = _STEP_ALTER[midi_pitch % 12]
    pitch_el = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch_el, "step").text = step
    if alter:
        ET.SubElement(pitch_el, "alter").text = str(alter)
    ET.SubElement(pitch_el, "octave").text = str(midi_pitch // 12 - 1)


def _append_duration_tail(
    note_el: ET.Element, units: int, table: dict
) -> None:
    """duration の後に置く type / dot / time-modification を追加する。"""
    note_type, dots, time_mod = table[units]
    ET.SubElement(note_el, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(note_el, "dot")
    if time_mod is not None:
        actual, normal = time_mod
        tm = ET.SubElement(note_el, "time-modification")
        ET.SubElement(tm, "actual-notes").text = str(actual)
        ET.SubElement(tm, "normal-notes").text = str(normal)


def _append_note(
    measure_el: ET.Element, note: TabNote, units: int, table: dict, *, chord: bool
) -> None:
    note_el = ET.SubElement(measure_el, "note")
    if chord:
        ET.SubElement(note_el, "chord")
    _append_pitch(note_el, note.midi_pitch)
    ET.SubElement(note_el, "duration").text = str(units)
    ET.SubElement(note_el, "voice").text = "1"
    _append_duration_tail(note_el, units, table)
    notations = ET.SubElement(note_el, "notations")
    technical = ET.SubElement(notations, "technical")
    ET.SubElement(technical, "string").text = str(note.string)
    ET.SubElement(technical, "fret").text = str(note.fret)


def _append_rest(
    measure_el: ET.Element, units: int, table: dict, *, full_measure: bool = False
) -> None:
    note_el = ET.SubElement(measure_el, "note")
    if full_measure:
        rest = ET.SubElement(note_el, "rest")
        rest.set("measure", "yes")
        ET.SubElement(note_el, "duration").text = str(units)
        ET.SubElement(note_el, "voice").text = "1"
        return
    ET.SubElement(note_el, "rest")
    ET.SubElement(note_el, "duration").text = str(units)
    ET.SubElement(note_el, "voice").text = "1"
    _append_duration_tail(note_el, units, table)


def _fill_rests(measure_el: ET.Element, units: int, table: dict) -> None:
    for part in _decompose(units, table):
        _append_rest(measure_el, part, table)


def _append_first_measure_header(
    measure_el: ET.Element, *, divisions: int, tempo_bpm: float
) -> None:
    """先頭小節の attributes（拍子・TAB 譜表定義）とテンポ表示を追加する。"""
    attributes = ET.SubElement(measure_el, "attributes")
    ET.SubElement(attributes, "divisions").text = str(divisions)
    key = ET.SubElement(attributes, "key")
    ET.SubElement(key, "fifths").text = "0"
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(BEATS_PER_MEASURE)
    ET.SubElement(time, "beat-type").text = "4"
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "TAB"
    ET.SubElement(clef, "line").text = "5"
    details = ET.SubElement(attributes, "staff-details")
    ET.SubElement(details, "staff-lines").text = "6"
    for line, (step, octave) in sorted(_STAFF_TUNING.items()):
        tuning = ET.SubElement(details, "staff-tuning")
        tuning.set("line", str(line))
        ET.SubElement(tuning, "tuning-step").text = step
        ET.SubElement(tuning, "tuning-octave").text = str(octave)

    direction = ET.SubElement(measure_el, "direction")
    direction.set("placement", "above")
    direction_type = ET.SubElement(direction, "direction-type")
    metronome = ET.SubElement(direction_type, "metronome")
    ET.SubElement(metronome, "beat-unit").text = "quarter"
    per_minute = round(tempo_bpm, 1)
    per_minute_text = str(int(per_minute)) if per_minute == int(per_minute) else str(per_minute)
    ET.SubElement(metronome, "per-minute").text = per_minute_text
    sound = ET.SubElement(direction, "sound")
    sound.set("tempo", per_minute_text)


def render_musicxml(tab: Sequence[TabNote], rhythm: Rhythm | None = None) -> str:
    """TabNote 列を MusicXML (score-partwise 4.0) の文字列にする。

    rhythm（rhythm.json）を渡すと推定テンポと per-note tick を使う。
    None のときはモジュール docstring の通り 120 BPM・4/4・16 分グリッドの
    表示用近似（従来動作）。入力データは変更しない。
    """
    if rhythm is None:
        events = _quantize(tab)
        divisions = DIVISIONS_PER_QUARTER
        table = _TYPE_BY_UNITS
        tempo_bpm = TEMPO_BPM
    else:
        events = _quantize_with_rhythm(tab, rhythm)
        divisions = rhythm.divisions_per_quarter
        if divisions != RHYTHM_DIVISIONS:
            raise ValueError(
                f"unsupported divisions_per_quarter: {divisions} "
                f"(expected {RHYTHM_DIVISIONS})"
            )
        table = _TYPE_BY_TICKS
        tempo_bpm = rhythm.tempo_bpm

    units_per_measure = BEATS_PER_MEASURE * divisions

    root = ET.Element("score-partwise")
    root.set("version", MUSICXML_VERSION)
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part")
    score_part.set("id", "P1")
    ET.SubElement(score_part, "part-name").text = "Guitar"
    part = ET.SubElement(root, "part")
    part.set("id", "P1")

    # 各音は小節末尾で切り詰めるので、終端も小節境界でクランプして数える
    last_end = max(
        (
            min(start + units, (start // units_per_measure + 1) * units_per_measure)
            for start, units, _ in events
        ),
        default=0,
    )
    n_measures = max(1, -(-last_end // units_per_measure))

    event_index = 0
    for m in range(n_measures):
        measure_el = ET.SubElement(part, "measure")
        measure_el.set("number", str(m + 1))
        if m == 0:
            _append_first_measure_header(
                measure_el, divisions=divisions, tempo_bpm=tempo_bpm
            )
        measure_start = m * units_per_measure
        measure_end = measure_start + units_per_measure

        cursor = measure_start
        has_content = False
        while event_index < len(events) and events[event_index][0] < measure_end:
            start, units, notes = events[event_index]
            if start > cursor:
                _fill_rests(measure_el, start - cursor, table)
            # 小節末尾で切り詰め → 単一音価に丸める（残りは休符で埋まる）
            units = _largest_representable(min(units, measure_end - start), table)
            for i, note in enumerate(notes):
                _append_note(measure_el, note, units, table, chord=i > 0)
            cursor = start + units
            has_content = True
            event_index += 1
        if not has_content:
            _append_rest(measure_el, units_per_measure, table, full_measure=True)
        elif cursor < measure_end:
            _fill_rests(measure_el, measure_end - cursor, table)

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{_DOCTYPE}\n{body}\n'


def save_musicxml(
    tab: Sequence[TabNote], path: Path, rhythm: Rhythm | None = None
) -> Path:
    """render_musicxml() の結果をファイルに書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_musicxml(tab, rhythm), encoding="utf-8")
    return path


__all__ = [
    "MUSICXML_VERSION",
    "TEMPO_BPM",
    "BEATS_PER_MEASURE",
    "DIVISIONS_PER_QUARTER",
    "RHYTHM_DIVISIONS",
    "UNITS_PER_MEASURE",
    "GRID_SEC",
    "render_musicxml",
    "save_musicxml",
]
