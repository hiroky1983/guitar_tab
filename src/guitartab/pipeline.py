"""パイプラインのステージ実行とキャッシュ。

各ステージは work/{id}/ に中間成果物を残す:

    work/{id}/source.wav          # download
    work/{id}/stems/guitar.wav    # separate（他ステムも同ディレクトリ）
    work/{id}/notes.json          # transcribe
    work/{id}/tab.json            # tab（運指割当）
    work/{id}/tab.txt             # tab（ASCII レンダリング）

既存ファイルがあればステージをスキップし、force=True で再実行する。
"""

from __future__ import annotations

import sys
from pathlib import Path

from guitartab.download import download_audio
from guitartab.separate import separate_guitar
from guitartab.tab.fingering import TabNote, assign_fingering, load_tab, save_tab
from guitartab.tab.render_ascii import (
    DEFAULT_LINE_WIDTH,
    DEFAULT_TIME_STEP_SEC,
    render_ascii,
)
from guitartab.transcribe.base import NoteEvent, TranscriberEngine, load_notes, save_notes

DEFAULT_WORK_ROOT = Path("work")
NOTES_FILENAME = "notes.json"
STEMS_DIRNAME = "stems"
TAB_FILENAME = "tab.json"
TAB_TEXT_FILENAME = "tab.txt"


def stage_download(url: str, work_root: Path, *, force: bool = False) -> Path:
    """work/{id}/source.wav を返す（キャッシュあり）。"""
    return download_audio(url, work_root, force=force)


def stage_separate(source_wav: Path, *, force: bool = False) -> Path:
    """work/{id}/stems/guitar.wav を返す（キャッシュあり）。"""
    stems_dir = source_wav.parent / STEMS_DIRNAME
    return separate_guitar(source_wav, stems_dir, force=force)


def stage_transcribe(
    audio_path: Path,
    engine: TranscriberEngine,
    notes_path: Path,
    *,
    force: bool = False,
) -> list[NoteEvent]:
    """notes.json を生成して NoteEvent リストを返す（キャッシュあり）。"""
    if notes_path.exists() and not force:
        print(f"cached: {notes_path}", file=sys.stderr)
        return load_notes(notes_path)
    notes = engine.transcribe(audio_path)
    save_notes(notes, notes_path)
    print(f"wrote {notes_path} ({len(notes)} notes, engine={engine.name})", file=sys.stderr)
    return notes


def stage_tab(
    notes_path: Path,
    tab_path: Path,
    tab_txt_path: Path,
    *,
    time_step_sec: float = DEFAULT_TIME_STEP_SEC,
    line_width: int = DEFAULT_LINE_WIDTH,
    force: bool = False,
) -> list[TabNote]:
    """notes.json から tab.json と ASCII tab.txt を生成する（キャッシュあり）。"""
    if tab_path.exists() and tab_txt_path.exists() and not force:
        print(f"cached: {tab_path}", file=sys.stderr)
        return load_tab(tab_path)
    tab = assign_fingering(load_notes(notes_path))
    save_tab(tab, tab_path)
    tab_txt_path.parent.mkdir(parents=True, exist_ok=True)
    tab_txt_path.write_text(
        render_ascii(tab, time_step_sec=time_step_sec, line_width=line_width) + "\n"
    )
    print(f"wrote {tab_path} / {tab_txt_path} ({len(tab)} notes)", file=sys.stderr)
    return tab


def run_transcribe_pipeline(
    url: str,
    engine: TranscriberEngine,
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    separate: bool = True,
    force: bool = False,
) -> Path:
    """download → separate → transcribe → tab を実行し notes.json のパスを返す。

    tab ステージの成果物は work/{id}/tab.json と work/{id}/tab.txt に残る。
    """
    source = stage_download(url, work_root, force=force)
    work_dir = source.parent
    audio = stage_separate(source, force=force) if separate else source
    notes_path = work_dir / NOTES_FILENAME
    stage_transcribe(audio, engine, notes_path, force=force)
    stage_tab(
        notes_path,
        work_dir / TAB_FILENAME,
        work_dir / TAB_TEXT_FILENAME,
        force=force,
    )
    return notes_path
