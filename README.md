# guitartab-transcriber

音声ファイル（wav/mp3 など）や YouTube URL からギター TAB 譜を生成する Python ライブラリです。
音声認識には [Basic Pitch](https://github.com/spotify/basic-pitch) を使用しています。

## インストール

```bash
pip install -e .
```

## 必要な依存ライブラリ

- basic-pitch
- librosa
- soundfile
- yt-dlp
- matplotlib

### 外部ツール（任意）

- LilyPond
  - 高品質な譜面（標準譜＋TAB）を SVG / PNG / PDF として出力する場合に使用します。

## 使い方

### 1. 実行用スクリプトの作成

プロジェクトルートに `main.py` などの名前でファイルを作成します。

```python
from guitartab_transcriber import Transcriber

# インスタンス生成
t = Transcriber()

# --- パターンA: YouTubeから生成 ---
url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
print(f"Transcribing from YouTube: {url}")

tab = t.transcribe_from_youtube(url)

# 結果をコンソールに表示
print("\n=== TAB ===")
print(tab.to_text())

# SVG 画像として保存（PNG の代替として軽量かつ可搬性の高い形式）
tab.to_svg("result.svg")
print("Saved visualization to result.svg")

# LilyPond ファイルを書き出し、SVG をビルド
ly_file = tab.to_lilypond("result.ly", title="Sample TAB")
print(f"Exported LilyPond source to {ly_file}")

# 環境に lilypond がインストールされていれば、そのままビルド可能
tab.to_lilypond("result.ly", compile_output="score.svg")
print("Built LilyPond score as score.svg")

# --- パターンB: ローカルの音声ファイルから生成 ---
# tab = t.transcribe("path/to/your/audio.wav")
# print(tab.to_text())
```

### 2. 実行

作成したスクリプトを実行します。

```bash
python main.py
```

LilyPond のビルドを行う場合は、`lilypond` コマンドがパスに通っている必要があります。

## 機能

- 音声ファイルからの TAB 生成
- YouTube URL からの TAB 生成
- テキスト形式（ASCII タブ）での出力
- JSON 形式での出力
- Matplotlib を使用した簡易グラフ出力（PNG / SVG など拡張子に追従）
- LilyPond 連携（.ly 出力＋オプションで SVG / PNG / PDF を自動ビルド）
- 簡易的な運指決定ロジック（低音弦優先）

## 今後の予定

- Drop D などの変則チューニング対応
- 運指最適化ロジックの改善
- TAB フォーマットの精度向上
