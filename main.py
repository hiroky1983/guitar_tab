import argparse
import sys
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe guitar tabs from a YouTube URL.")
    parser.add_argument(
        "--url",
        "-u",
        required=True,
        help="YouTube video URL to transcribe",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="result.png",
        help="Path to save the generated tab visualization",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Manually specify BPM (overrides estimation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from guitartab_transcriber import Transcriber

    # インスタンス生成
    t = Transcriber()

    # --- パターンA: YouTubeから生成 ---
    url = args.url
    print(f"Transcribing from YouTube: {url}")

    try:
        tab = t.transcribe_from_youtube(url, bpm=args.bpm)

        # 結果をコンソールに表示
        print("\n=== TAB ===")
        print(tab.to_text())

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
        sys.exit(1)


if __name__ == "__main__":
    main()
