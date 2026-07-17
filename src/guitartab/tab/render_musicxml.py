"""MusicXML レンダラ（score-partwise 4.0、TAB 単段）。

TabNote 列（tab.json）を MuseScore / Guitar Pro で開ける MusicXML に書き出す。
第三出力（docs/DESIGN.md「出力」参照）。stdlib の xml.etree のみで組み立てる。

- TAB 表記: ``<clef sign="TAB"/>`` + 6 線 ``<staff-details>``（EADGBE の
  staff-tuning）を持つ 1 パート構成で、各音符に
  ``<notations><technical><string>/<fret>`` を付ける。五線+TAB の 2 段構成では
  なく TAB 単段（MuseScore で開けて弦・フレットが正しく表示されることを最優先）。
- リズムは**表示用の近似**: 固定 120 BPM・4/4 とし、onset_sec を最も近い
  16 分音符グリッド（0.125 秒）へ置き、duration も 16 分単位に丸める。
  同一グリッドに落ちた複数音は和音（2 音目以降に ``<chord/>``）、隙間は休符で
  埋め、小節は 4/4 で機械的に区切る。リズム量子化の本実装は M4 スコープで
  あり、**この丸めは MusicXML 出力だけの近似。notes.json / tab.json の
  データは一切変更しない。**
- 音価の縮退規則（すべて表示上の近似）:
  - 小節境界をまたぐ音は小節末尾で切り詰める（タイは張らない）。
  - 次の音の開始と重なる分は切り詰める（単声部で backup を使わないため）。
  - 単一音価で表現できない長さ（5 単位等）は表現できる最大の音価
    （全/付点2分/2分/付点4分/4分/付点8分/8分/16分）へ切り詰め、残りは休符になる。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from guitartab.tab.fingering import TabNote

MUSICXML_VERSION = "4.0"
TEMPO_BPM = 120.0  # 固定（量子化は M4 スコープ）
BEATS_PER_MEASURE = 4  # 4/4 固定
DIVISIONS_PER_QUARTER = 4  # 1 division = 16 分音符
UNITS_PER_MEASURE = BEATS_PER_MEASURE * DIVISIONS_PER_QUARTER  # 16
GRID_SEC = 60.0 / TEMPO_BPM / DIVISIONS_PER_QUARTER  # 0.125 秒 = 16 分音符

# 単一の音価で表現できる長さ（16 分音符単位）→ (type, 付点数)
_TYPE_BY_UNITS = {
    16: ("whole", 0),
    12: ("half", 1),
    8: ("half", 0),
    6: ("quarter", 1),
    4: ("quarter", 0),
    3: ("eighth", 1),
    2: ("eighth", 0),
    1: ("16th", 0),
}
_REPRESENTABLE_UNITS = sorted(_TYPE_BY_UNITS, reverse=True)

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


def _largest_representable(units: int) -> int:
    """units 以下で単一音価として表現できる最大の長さを返す（units >= 1 前提）。"""
    for value in _REPRESENTABLE_UNITS:
        if value <= units:
            return value
    raise ValueError(f"units must be >= 1: {units}")


def _decompose(units: int) -> list[int]:
    """休符用: units を表現できる音価の列（大きい順）に分解する。"""
    parts = []
    while units > 0:
        part = _largest_representable(units)
        parts.append(part)
        units -= part
    return parts


def _quantize(tab: Sequence[TabNote]) -> list[tuple[int, int, list[TabNote]]]:
    """16 分グリッドへ量子化し (開始 unit, 長さ unit, 構成音) 列を返す。

    - 同一グリッドの音は和音グループにまとめる（長さは最長の構成音）。
    - 完全重複（同じ弦・同じフレット）は除去する。
    - 長さは「次のグループの開始」で切り詰める（単声部表現のため）。
    """
    groups: dict[int, list[TabNote]] = {}
    durations: dict[int, int] = {}
    for note in tab:
        if note.onset_sec < 0:
            raise ValueError(f"onset_sec must be non-negative: {note.onset_sec}")
        if not 1 <= note.string <= 6:
            raise ValueError(f"string must be in [1, 6]: {note.string}")
        if note.fret < 0:
            raise ValueError(f"fret must be non-negative: {note.fret}")
        if not 0 <= note.midi_pitch <= 127:
            raise ValueError(f"midi_pitch out of MIDI range [0, 127]: {note.midi_pitch}")
        start = round(note.onset_sec / GRID_SEC)
        units = max(1, round(note.duration_sec / GRID_SEC))
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


def _append_pitch(note_el: ET.Element, midi_pitch: int) -> None:
    step, alter = _STEP_ALTER[midi_pitch % 12]
    pitch_el = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch_el, "step").text = step
    if alter:
        ET.SubElement(pitch_el, "alter").text = str(alter)
    ET.SubElement(pitch_el, "octave").text = str(midi_pitch // 12 - 1)


def _append_note(
    measure_el: ET.Element, note: TabNote, units: int, *, chord: bool
) -> None:
    note_type, dots = _TYPE_BY_UNITS[units]
    note_el = ET.SubElement(measure_el, "note")
    if chord:
        ET.SubElement(note_el, "chord")
    _append_pitch(note_el, note.midi_pitch)
    ET.SubElement(note_el, "duration").text = str(units)
    ET.SubElement(note_el, "voice").text = "1"
    ET.SubElement(note_el, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(note_el, "dot")
    notations = ET.SubElement(note_el, "notations")
    technical = ET.SubElement(notations, "technical")
    ET.SubElement(technical, "string").text = str(note.string)
    ET.SubElement(technical, "fret").text = str(note.fret)


def _append_rest(
    measure_el: ET.Element, units: int, *, full_measure: bool = False
) -> None:
    note_el = ET.SubElement(measure_el, "note")
    if full_measure:
        rest = ET.SubElement(note_el, "rest")
        rest.set("measure", "yes")
        ET.SubElement(note_el, "duration").text = str(units)
        ET.SubElement(note_el, "voice").text = "1"
        return
    note_type, dots = _TYPE_BY_UNITS[units]
    ET.SubElement(note_el, "rest")
    ET.SubElement(note_el, "duration").text = str(units)
    ET.SubElement(note_el, "voice").text = "1"
    ET.SubElement(note_el, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(note_el, "dot")


def _fill_rests(measure_el: ET.Element, units: int) -> None:
    for part in _decompose(units):
        _append_rest(measure_el, part)


def _append_first_measure_header(measure_el: ET.Element) -> None:
    """先頭小節の attributes（拍子・TAB 譜表定義）とテンポ表示を追加する。"""
    attributes = ET.SubElement(measure_el, "attributes")
    ET.SubElement(attributes, "divisions").text = str(DIVISIONS_PER_QUARTER)
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
    ET.SubElement(metronome, "per-minute").text = str(int(TEMPO_BPM))
    sound = ET.SubElement(direction, "sound")
    sound.set("tempo", str(int(TEMPO_BPM)))


def render_musicxml(tab: Sequence[TabNote]) -> str:
    """TabNote 列を MusicXML (score-partwise 4.0) の文字列にする。

    リズムはモジュール docstring の通り 120 BPM・4/4・16 分グリッドの
    表示用近似。入力データは変更しない。
    """
    events = _quantize(tab)

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
            min(start + units, (start // UNITS_PER_MEASURE + 1) * UNITS_PER_MEASURE)
            for start, units, _ in events
        ),
        default=0,
    )
    n_measures = max(1, -(-last_end // UNITS_PER_MEASURE))

    event_index = 0
    for m in range(n_measures):
        measure_el = ET.SubElement(part, "measure")
        measure_el.set("number", str(m + 1))
        if m == 0:
            _append_first_measure_header(measure_el)
        measure_start = m * UNITS_PER_MEASURE
        measure_end = measure_start + UNITS_PER_MEASURE

        cursor = measure_start
        has_content = False
        while event_index < len(events) and events[event_index][0] < measure_end:
            start, units, notes = events[event_index]
            if start > cursor:
                _fill_rests(measure_el, start - cursor)
            # 小節末尾で切り詰め → 単一音価に丸める（残りは休符で埋まる）
            units = _largest_representable(min(units, measure_end - start))
            for i, note in enumerate(notes):
                _append_note(measure_el, note, units, chord=i > 0)
            cursor = start + units
            has_content = True
            event_index += 1
        if not has_content:
            _append_rest(measure_el, UNITS_PER_MEASURE, full_measure=True)
        elif cursor < measure_end:
            _fill_rests(measure_el, measure_end - cursor)

    ET.indent(root)
    body = ET.tostring(root, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{_DOCTYPE}\n{body}\n'


def save_musicxml(tab: Sequence[TabNote], path: Path) -> Path:
    """render_musicxml() の結果をファイルに書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_musicxml(tab), encoding="utf-8")
    return path


__all__ = [
    "MUSICXML_VERSION",
    "TEMPO_BPM",
    "BEATS_PER_MEASURE",
    "DIVISIONS_PER_QUARTER",
    "UNITS_PER_MEASURE",
    "GRID_SEC",
    "render_musicxml",
    "save_musicxml",
]
