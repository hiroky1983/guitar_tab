from guitartab_transcriber import Transcriber

# インスタンス生成
t = Transcriber()

# --- パターンA: YouTubeから生成 ---
# テスト用動画: "THIS RIFF IS CRAZY #shorts" (ID: 3BoF2TROWjs)
url = "https://www.youtube.com/watch?v=3BoF2TROWjs"
print(f"Transcribing from YouTube: {url}")

try:
    tab = t.transcribe_from_youtube(url)

    # 結果をコンソールに表示
    print("\n=== TAB ===")
    print(tab.to_text())

    # PNG の代わりにSVG形式で保存
    tab.to_svg("result.svg")
    print("Saved visualization to result.svg")

except Exception as e:
    print(f"Error occurred: {e}")
