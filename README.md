# guitartab-transcriber

音声ファイル（wav/mp3 など）や YouTube URL からギター TAB 譜を生成する Python ライブラリです。
音声認識には [Basic Pitch](https://github.com/spotify/basic-pitch) を使用しています。

本ライブラリの出力責務は **LilyPond 記法（.ly）まで** で、PDF / SVG / PNG などの最終成果物は LilyPond 本体（外部ツール）が生成します。

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

# LilyPond 記法（.ly）を書き出し
ly_file = tab.to_lilypond("result.ly", title="Sample TAB")
print(f"Exported LilyPond source to {ly_file}")

# 生成された .ly を LilyPond CLI でビルド（外部ツールの責務）
# 例: lilypond --svg -o score result.ly

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

### LilyPond 連携の考え方

1. **ライブラリの責務**: `tab.to_lilypond()` で TAB 情報から LilyPond 記法（`.ly`）を生成する。
2. **外部ツールの責務**: 生成した `.ly` を LilyPond コマンド（例: `lilypond --svg -o score result.ly`）でビルドし、PDF / SVG / PNG を得る。

`.ly` は HTML のような「楽譜の設計図」に相当し、Web フロントエンドに埋め込む際は LilyPond が生成した `score.svg` / `score.png` / `score.pdf` を利用する想定です。

## 機能

- 音声ファイルからの TAB 生成
- YouTube URL からの TAB 生成
- テキスト形式（ASCII タブ）での出力
- JSON 形式での出力
- Matplotlib を使用した簡易グラフ出力（PNG / SVG など拡張子に追従）
- LilyPond 連携（TAB から LilyPond 記法 .ly を生成）
- 簡易的な運指決定ロジック（低音弦優先）

## 今後の予定

- Drop D などの変則チューニング対応
- 運指最適化ロジックの改善
- TAB フォーマットの精度向上
