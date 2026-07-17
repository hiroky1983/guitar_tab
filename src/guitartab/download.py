"""YouTube 音声ダウンロード（yt-dlp）。v1 guitartab_transcriber/youtube.py を整理して流用。

出力: work/{video_id}/source.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

import yt_dlp

SOURCE_FILENAME = "source.wav"


def _print_progress(download_info: dict) -> None:
    """静音モードでも進捗が分かるように最小限のステータス行を stderr へ出す。"""
    status = download_info.get("status")
    if status == "downloading":
        percent = download_info.get("_percent_str")
        speed = download_info.get("_speed_str")
        eta = download_info.get("_eta_str")
        line = "Downloading audio" + (
            f" - {percent} @ {speed} ETA {eta}" if percent else ""
        )
        print(line, file=sys.stderr)
    elif status == "finished":
        print("Download finished, extracting WAV...", file=sys.stderr)


def _ydl_opts(out_tmpl: str | None = None) -> dict:
    opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        # ローカル JS ランタイム非依存のプレイヤークライアントを明示指定
        # （v1 で Node.js 不在環境の警告回避に必要だった）
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["dash", "hls"],
            }
        },
        "quiet": True,
        "progress_hooks": [_print_progress],
    }
    if out_tmpl is not None:
        opts["outtmpl"] = out_tmpl
    return opts


def fetch_video_id(url: str) -> str:
    """ダウンロードせずに動画 ID を取得する（メタデータ取得のみ、ネットワークに出る）。"""
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return info["id"]


def download_audio(url: str, work_root: Path, *, force: bool = False) -> Path:
    """YouTube 音声を WAV で work_root/{video_id}/source.wav に保存しパスを返す。

    既に source.wav が存在する場合はダウンロードをスキップする（force=True で再取得）。
    """
    work_root = Path(work_root)
    video_id = fetch_video_id(url)
    target = work_root / video_id / SOURCE_FILENAME
    if target.exists() and not force:
        print(f"cached: {target}", file=sys.stderr)
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(target.parent / "source.%(ext)s")
    with yt_dlp.YoutubeDL(_ydl_opts(out_tmpl)) as ydl:
        ydl.extract_info(url, download=True)

    if not target.exists():
        raise RuntimeError(f"yt-dlp finished but {target} was not created")
    return target
