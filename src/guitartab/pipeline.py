"""パイプラインのステージ実行とキャッシュ。

各ステージは work/{id}/ に中間成果物を残す:

    work/{id}/source.wav          # download
    work/{id}/stems/guitar.wav    # separate（他ステムも同ディレクトリ）
    work/{id}/notes.json          # transcribe

既存ファイルがあればステージをスキップし、force=True で再実行する。
"""

from __future__ import annotations

import sys
from pathlib import Path

from guitartab.download import download_audio
from guitartab.separate import separate_guitar
from guitartab.transcribe.base import NoteEvent, TranscriberEngine, load_notes, save_notes

DEFAULT_WORK_ROOT = Path("work")
NOTES_FILENAME = "notes.json"
STEMS_DIRNAME = "stems"


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


def run_transcribe_pipeline(
    url: str,
    engine: TranscriberEngine,
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    separate: bool = True,
    force: bool = False,
) -> Path:
    """download → separate → transcribe を実行し notes.json のパスを返す。"""
    source = stage_download(url, work_root, force=force)
    work_dir = source.parent
    audio = stage_separate(source, force=force) if separate else source
    notes_path = work_dir / NOTES_FILENAME
    stage_transcribe(audio, engine, notes_path, force=force)
    return notes_path
