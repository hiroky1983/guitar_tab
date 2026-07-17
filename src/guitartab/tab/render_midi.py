"""MIDI レンダラ（SMF format 0）。

NoteEvent 列（notes.json）または TabNote 列（tab.json）を標準 MIDI ファイルに
書き出す。耳で検証するための第二出力（docs/DESIGN.md「出力」参照）。
イベント数が少ないため外部ライブラリは使わず stdlib のみで SMF を組み立てる。

- テンポは固定（デフォルト 120 BPM）。絶対秒 → tick の変換のみ行い、
  リズム量子化・小節割りは M4 スコープでここでは行わない。
- velocity: NoteEvent.velocity (0.0-1.0) を 1-127 に写像する。
  0 以下・情報なし（TabNote）は 100 とする。
- 同時発音は同一 tick の note on として並ぶ。同一 tick では note off を
  note on より先に置く（同音連打で音が消えないための SMF の慣例）。
- duration が 1 tick 未満の音は 1 tick に切り上げる（note on/off の順序保証）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Union

from guitartab.tab.fingering import TabNote
from guitartab.transcribe.base import NoteEvent

DEFAULT_TEMPO_BPM = 120.0
DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_VELOCITY = 100
DEFAULT_PROGRAM = 25  # GM (0-based): Acoustic Guitar (steel)
_CHANNEL = 0

RenderableNote = Union[NoteEvent, TabNote]


def _vlq(value: int) -> bytes:
    """SMF の可変長数値（variable-length quantity）にエンコードする。"""
    if value < 0:
        raise ValueError(f"delta time must be non-negative: {value}")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))


def midi_velocity(velocity: float | None) -> int:
    """0.0-1.0 の velocity を MIDI velocity 1-127 に写像する（0/None は 100）。"""
    if velocity is None or velocity <= 0.0:
        return DEFAULT_VELOCITY
    return max(1, min(127, round(velocity * 127)))


def _normalize(note: RenderableNote) -> tuple[float, float, int, int]:
    """(onset_sec, offset_sec, midi_pitch, midi_velocity) に正規化する。"""
    if isinstance(note, TabNote):
        return (
            note.onset_sec,
            note.onset_sec + note.duration_sec,
            note.midi_pitch,
            DEFAULT_VELOCITY,
        )
    return note.onset_sec, note.offset_sec, note.midi_pitch, midi_velocity(note.velocity)


def render_midi(
    notes: Sequence[RenderableNote],
    *,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
) -> bytes:
    """NoteEvent / TabNote 列を SMF format 0 のバイト列にする。"""
    if tempo_bpm <= 0:
        raise ValueError(f"tempo_bpm must be positive: {tempo_bpm}")
    if not 1 <= ticks_per_beat <= 0x7FFF:
        raise ValueError(f"ticks_per_beat must be in [1, 32767]: {ticks_per_beat}")

    ticks_per_sec = ticks_per_beat * tempo_bpm / 60.0

    # (tick, order, message) を集める。order: note off(0) < note on(1)。
    # 同順位は (pitch, velocity) で決定的に並べる。
    events: list[tuple[int, int, int, int, bytes]] = []
    for note in notes:
        onset_sec, offset_sec, pitch, velocity = _normalize(note)
        if not 0 <= pitch <= 127:
            raise ValueError(f"midi_pitch out of MIDI range [0, 127]: {pitch}")
        if onset_sec < 0:
            raise ValueError(f"onset_sec must be non-negative: {onset_sec}")
        on_tick = round(onset_sec * ticks_per_sec)
        off_tick = max(on_tick + 1, round(offset_sec * ticks_per_sec))
        events.append((on_tick, 1, pitch, velocity, bytes([0x90 | _CHANNEL, pitch, velocity])))
        events.append((off_tick, 0, pitch, velocity, bytes([0x80 | _CHANNEL, pitch, 0])))
    events.sort(key=lambda e: e[:4])

    track = bytearray()
    # tempo meta event (FF 51 03) + program change
    usec_per_beat = round(60_000_000 / tempo_bpm)
    track += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + usec_per_beat.to_bytes(3, "big")
    track += _vlq(0) + bytes([0xC0 | _CHANNEL, DEFAULT_PROGRAM])
    prev_tick = 0
    for tick, _, _, _, message in events:
        track += _vlq(tick - prev_tick) + message
        prev_tick = tick
    track += _vlq(0) + bytes([0xFF, 0x2F, 0x00])  # end of track

    header = b"MThd" + (6).to_bytes(4, "big")
    header += (0).to_bytes(2, "big")  # format 0
    header += (1).to_bytes(2, "big")  # 1 track
    header += ticks_per_beat.to_bytes(2, "big")
    return bytes(header + b"MTrk" + len(track).to_bytes(4, "big") + track)


def save_midi(
    notes: Sequence[RenderableNote],
    path: Path,
    *,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    ticks_per_beat: int = DEFAULT_TICKS_PER_BEAT,
) -> Path:
    """render_midi() の結果をファイルに書き出す。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_midi(notes, tempo_bpm=tempo_bpm, ticks_per_beat=ticks_per_beat))
    return path


__all__ = [
    "DEFAULT_TEMPO_BPM",
    "DEFAULT_TICKS_PER_BEAT",
    "DEFAULT_VELOCITY",
    "midi_velocity",
    "render_midi",
    "save_midi",
]
