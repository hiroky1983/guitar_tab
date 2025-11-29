from pathlib import Path
import yt_dlp

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
        "extractor_args": {"youtube": {"player_client": ["default"]}},
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    audio_path = out_dir / f"{info['id']}.wav"
    return audio_path
