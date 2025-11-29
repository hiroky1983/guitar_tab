import argparse
import sys


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
        tab = t.transcribe_from_youtube(url)

        # 結果をコンソールに表示
        print("\n=== TAB ===")
        print(tab.to_text())

        # 画像として保存
        tab.to_matplotlib(args.output)
        print(f"Saved visualization to {args.output}")

    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
