import shutil

from guitartab_transcriber import Transcriber

# インスタンス生成
t = Transcriber()

# --- パターンA: YouTubeから生成 ---
# テスト用動画: "THIS RIFF IS CRAZY #shorts" (ID: 3BoF2TROWjs)
url = "https://www.youtube.com/watch?v=3BoF2TROWjs"
print(f"Transcribing from YouTube: {url}")

try:
    tab = t.transcribe_from_youtube(url)

    # LilyPond 記法（.ly）を書き出し（ここまでがライブラリの責務）
    ly_file = tab.to_lilypond("result.ly", title="Sample TAB")
    print(f"Exported LilyPond source to {ly_file}")

    lilypond_path = shutil.which("lilypond")
    if lilypond_path:
        svg_file = tab.to_lilypond(
            "result.ly", title="Sample TAB", compile_output="score.svg", lilypond_executable=lilypond_path
        )
        print(f"Generated engraved SVG via LilyPond: {svg_file}")
    else:
        print("LilyPond is not installed; skipped SVG generation. Install LilyPond to produce score.svg.")

except Exception as e:
    print(f"Error occurred: {e}")
