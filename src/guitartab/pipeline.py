"""パイプラインのステージ実行とキャッシュ。

各ステージは work/{id}/ に中間成果物を残す:

    work/{id}/source.wav          # download
    work/{id}/stems/guitar.wav    # separate（他ステムも同ディレクトリ）
    work/{id}/notes.json          # transcribe
    work/{id}/tab.json            # tab（運指割当）
    work/{id}/tab.txt             # tab（ASCII レンダリング）
    work/{id}/rhythm.json         # quantize（テンポ推定 + 格子スナップ、M4a）
    work/{id}/output.mid          # midi（耳で検証する用の MIDI レンダリング）
    work/{id}/output.musicxml     # musicxml（MuseScore / Guitar Pro 連携用）

既存ファイルがあればステージをスキップし、force=True で再実行する。
quantize は tab とは独立（notes.json / tab.json は不変）で、失敗しても
midi / musicxml が従来の固定 120BPM 近似へフォールバックするだけで
パイプラインは止めない。
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

from guitartab.download import download_audio
from guitartab.rhythm.estimate import LibrosaConstantTempoEstimator, TempoEstimator
from guitartab.rhythm.quantize import quantize_notes
from guitartab.rhythm.schema import Rhythm, load_rhythm, save_rhythm
from guitartab.separate import separate_guitar
from guitartab.tab.fingering import TabNote, assign_fingering, load_tab, save_tab
from guitartab.tab.render_ascii import (
    DEFAULT_LINE_WIDTH,
    DEFAULT_TIME_STEP_SEC,
    render_ascii,
)
from guitartab.tab.render_midi import DEFAULT_TEMPO_BPM, save_midi
from guitartab.tab.render_musicxml import save_musicxml
from guitartab.transcribe.base import NoteEvent, TranscriberEngine, load_notes, save_notes
from guitartab.transcribe.select import ENGINE_SELECTION_FILENAME, AutoEngineSelector

DEFAULT_WORK_ROOT = Path("work")
NOTES_FILENAME = "notes.json"
STEMS_DIRNAME = "stems"
TAB_FILENAME = "tab.json"
TAB_TEXT_FILENAME = "tab.txt"
RHYTHM_FILENAME = "rhythm.json"
MIDI_FILENAME = "output.mid"
MUSICXML_FILENAME = "output.musicxml"


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


def stage_quantize(
    notes_path: Path,
    rhythm_path: Path,
    *,
    audio_path: Path | None = None,
    estimator: TempoEstimator | None = None,
    force: bool = False,
) -> Rhythm:
    """notes.json（+ 任意の音声）から rhythm.json を生成する（キャッシュあり）。"""
    if rhythm_path.exists() and not force:
        print(f"cached: {rhythm_path}", file=sys.stderr)
        return load_rhythm(rhythm_path)
    if estimator is None:
        estimator = LibrosaConstantTempoEstimator()
    notes = load_notes(notes_path)
    estimate = estimator.estimate(
        [n.onset_sec for n in notes], audio_path=audio_path
    )
    audio_source = None
    if audio_path is not None:
        try:
            audio_source = str(Path(audio_path).relative_to(notes_path.parent))
        except ValueError:
            audio_source = str(audio_path)
    rhythm = quantize_notes(notes, estimate, audio_source=audio_source)
    save_rhythm(rhythm, rhythm_path)
    print(
        f"wrote {rhythm_path} (tempo={rhythm.tempo_bpm:.1f} BPM, "
        f"{len(rhythm.notes)} notes)",
        file=sys.stderr,
    )
    return rhythm


def _load_rhythm_or_none(rhythm_path: Path | None) -> Rhythm | None:
    if rhythm_path is None:
        return None
    rhythm_path = Path(rhythm_path)
    if not rhythm_path.exists():
        raise FileNotFoundError(f"rhythm.json not found: {rhythm_path}")
    return load_rhythm(rhythm_path)


def stage_midi(
    notes_path: Path,
    midi_path: Path,
    *,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    rhythm_path: Path | None = None,
    force: bool = False,
) -> Path:
    """notes.json から MIDI (output.mid) を生成する（キャッシュあり）。

    rhythm_path を渡すと推定テンポ・量子化 tick でレンダリングする
    （なければ従来の固定テンポ・非量子化）。
    """
    if midi_path.exists() and not force:
        print(f"cached: {midi_path}", file=sys.stderr)
        return midi_path
    notes = load_notes(notes_path)
    save_midi(
        notes, midi_path, tempo_bpm=tempo_bpm, rhythm=_load_rhythm_or_none(rhythm_path)
    )
    print(f"wrote {midi_path} ({len(notes)} notes)", file=sys.stderr)
    return midi_path


def stage_musicxml(
    tab_path: Path,
    musicxml_path: Path,
    *,
    rhythm_path: Path | None = None,
    force: bool = False,
) -> Path:
    """tab.json から MusicXML (output.musicxml) を生成する（キャッシュあり）。

    rhythm_path を渡すと推定テンポ・量子化 tick でレンダリングする
    （なければ従来の固定 120BPM・16 分グリッド近似）。
    """
    if musicxml_path.exists() and not force:
        print(f"cached: {musicxml_path}", file=sys.stderr)
        return musicxml_path
    tab = load_tab(tab_path)
    save_musicxml(tab, musicxml_path, rhythm=_load_rhythm_or_none(rhythm_path))
    print(f"wrote {musicxml_path} ({len(tab)} notes)", file=sys.stderr)
    return musicxml_path


def run_transcribe_pipeline(
    url: str,
    engine: TranscriberEngine | AutoEngineSelector,
    *,
    work_root: Path = DEFAULT_WORK_ROOT,
    separate: bool = True,
    quantize: bool = True,
    rhythm_source: str = "stem",
    rhythm_estimator: TempoEstimator | None = None,
    force: bool = False,
) -> Path:
    """download → separate → transcribe → tab → quantize → midi → musicxml を
    実行し notes.json のパスを返す。

    tab ステージの成果物は work/{id}/tab.json と work/{id}/tab.txt、
    quantize は work/{id}/rhythm.json、midi は work/{id}/output.mid、
    musicxml は work/{id}/output.musicxml に残る。quantize が失敗した場合は
    警告を出し、midi / musicxml は従来の固定 120BPM 近似で続行する。

    rhythm_source はテンポ・拍推定に使う音声の選択（M4b）:
    "stem" = 転写に使った音声そのもの（従来動作）、
    "mix" = 分離前の原曲ミックス work/{id}/source.wav
    （転写ノートは従来どおりステム由来のまま = リズムだけミックスから取る）。
    rhythm_estimator は TempoEstimator の差し替え（None = librosa 候補 A。
    このとき rhythm_source="mix" なら音声トラッカー信頼モードを自動で有効化
    する — ミックスでは候補選択層がテンポレベルを上書きして族外へ落とす
    実測があるため。明示的に estimator を渡した場合はその設定を尊重する）。
    """
    if rhythm_source not in ("stem", "mix"):
        raise ValueError(f"rhythm_source must be 'stem' or 'mix': {rhythm_source!r}")
    if rhythm_estimator is None:
        rhythm_estimator = LibrosaConstantTempoEstimator(
            trust_tracker=rhythm_source == "mix"
        )
    source = stage_download(url, work_root, force=force)
    work_dir = source.parent
    audio = stage_separate(source, force=force) if separate else source
    if isinstance(engine, AutoEngineSelector):
        # --engine auto: separate 後のステム（--no-separate 時は source.wav）を
        # クリーン/歪み判定して実エンジンへ差し替える。判定は毎回実行して
        # 記録を work/{id}/engine_selection.json に残す（軽量なのでキャッシュ不要）。
        engine = engine.resolve(
            audio, selection_path=work_dir / ENGINE_SELECTION_FILENAME
        )
    notes_path = work_dir / NOTES_FILENAME
    stage_transcribe(audio, engine, notes_path, force=force)
    stage_tab(
        notes_path,
        work_dir / TAB_FILENAME,
        work_dir / TAB_TEXT_FILENAME,
        force=force,
    )
    rhythm_path: Path | None = None
    if quantize:
        try:
            rhythm_path = work_dir / RHYTHM_FILENAME
            rhythm_audio = source if rhythm_source == "mix" else audio
            stage_quantize(
                notes_path,
                rhythm_path,
                audio_path=rhythm_audio,
                estimator=rhythm_estimator,
                force=force,
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            print(
                "quantize stage failed; falling back to fixed-tempo rendering",
                file=sys.stderr,
            )
            rhythm_path = None
    stage_midi(notes_path, work_dir / MIDI_FILENAME, rhythm_path=rhythm_path, force=force)
    stage_musicxml(
        work_dir / TAB_FILENAME,
        work_dir / MUSICXML_FILENAME,
        rhythm_path=rhythm_path,
        force=force,
    )
    return notes_path
