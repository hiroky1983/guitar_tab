from pathlib import Path
import sys
import yt_dlp

def _print_progress(download_info: dict) -> None:
    """Minimal progress output so users know the download isn't "playing" audio.

    yt-dlp is already optimized to fetch audio as fast as the network allows, but
    the default quiet mode can look like the script is stalled. This hook prints
    a lightweight status line without enabling the full yt-dlp progress UI.
    """

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


def download_youtube_audio(url: str, out_dir: Path) -> Path:
    """
    YouTubeの音声をwavにして保存して、そのパスを返す
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(out_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_tmpl,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        # Avoid reliance on a local JavaScript runtime by opting into the default
        # player client. This suppresses yt-dlp's warning and ensures extraction
        # works even when Node.js is unavailable.
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["dash", "hls"],
            }
        },
        "quiet": True,
        "progress_hooks": [_print_progress],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    audio_path = out_dir / f"{info['id']}.wav"
    return audio_path
