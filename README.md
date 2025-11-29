# guitartab-transcriber

音声ファイル（wav/mp3 など）や YouTube URL からギター TAB 譜を生成する Python ライブラリです。
音声認識には [Basic Pitch](https://github.com/spotify/basic-pitch) を使用しています。

生成した TAB は

- コンソールでの確認用 **ASCII TAB**
- プログラムから扱うための **JSON**
- 簡易な可視化用の **画像（Matplotlib）**
- LilyPond を用いた **高品質な譜面画像（SVG/PNG）や PDF**（Web フロントエンドに `<img>` / `<object>` などで埋め込み可能）

として利用できます。（パターン A）

---

## インストール

```bash
pip install -e .
```

---

## 必要な依存ライブラリ

- basic-pitch
- librosa
- soundfile
- yt-dlp
- matplotlib

### 外部ツール（任意）

- LilyPond

  - 高品質な楽譜（標準譜＋ TAB）を SVG / PNG / PDF として出力する場合に使用します。
  - `tab` オブジェクトから LilyPond 記法（`.ly`）を生成し、`lilypond` コマンドでビルドする想定です。

---

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

# 画像として保存
tab.to_matplotlib("result.png")
print("Saved visualization to result.png")

# --- パターンB: ローカルの音声ファイルから生成 ---
# tab = t.transcribe("path/to/your/audio.wav")
# print(tab.to_text())
```

LilyPond を使った高品質な譜面出力を行いたい場合は、
`tab` オブジェクトから LilyPond 記法（`.ly`）を生成するメソッド（例: `to_lilypond()`）を用意し、
生成された `.ly` ファイルを `lilypond` コマンドでビルドして SVG / PNG / PDF を作成します。
その出力ファイルを Web フロントエンド側で `<img>` / `<object>` / `<embed>` などを使って表示する想定です。

### 2. 実行

作成したスクリプトを実行します。

```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
# 結果の画像を別ファイル名で保存したい場合
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --output my_tab.png
```

- 音声は yt-dlp で一括ダウンロード（可能な限り高速）してから解析します。再生速度でストリーミングしているわけではありません。
  ダウンロード中の進捗は標準エラー出力に簡易表示されます。

---

## 機能

### 入力形式

- **ローカル音声ファイル**: wav/mp3 などの音声ファイルから直接 TAB 譜を生成
- **YouTube URL**: YouTube の動画 URL を指定して、音声を自動ダウンロード・変換して TAB 譜を生成

  - yt-dlp による高速ダウンロード（ストリーミング再生ではなく一括ダウンロード）
  - ダウンロード進捗の簡易表示（標準エラー出力）
  - ffmpeg による WAV 変換

### AI 音声認識

- **Basic Pitch (Spotify 製)** を使用した高精度な音声 →MIDI 変換

  - ICASSP 2022 モデルを使用
  - ピッチ検出範囲: MIDI 40-88（ギター音域に最適化）
  - サンプルレート: 44100Hz

### TAB 譜生成ロジック

- **運指決定アルゴリズム**:

  - E 標準チューニング対応（6 弦 E2=40, 5 弦 A2=45, 4 弦 D3=50, 3 弦 G3=55, 2 弦 B3=59, 1 弦 E4=64）
  - 各音に対して弾ける弦を探索し、できるだけ太い弦（低音弦）を優先
  - フレット範囲: 0-20 フレット
  - ギター音域外の音は自動的にフィルタリング

### 出力形式

1. **テキスト形式（ASCII タブ）**

   - 6 本の弦を視覚的に表現
   - 時間順にフレット番号を配置
   - コンソール出力に最適

2. **JSON 形式**

   - 各イベントの詳細情報（弦番号、フレット、開始時刻、終了時刻）
   - プログラムでの後処理に最適
   - BPM 情報（オプション）

3. **画像形式（Matplotlib）**

   - 6 本の弦を横線で表現
   - 時間軸に沿ってフレット番号を配置
   - PNG/JPG などの画像ファイルとして保存可能
   - 視覚的な確認・共有に最適

4. **高品質な譜面出力（LilyPond 連携 / パターン A）**

   - TAB 情報から LilyPond 記法（`.ly`）を生成
   - `lilypond` コマンドで SVG / PNG / PDF を生成
   - 標準譜＋ TAB、拍子記号、レイアウトなどを含む高品質な譜面を出力
   - 生成された SVG / PNG / PDF を Web フロントエンドにそのまま埋め込んで表示可能

---

### 設定オプション

- **TranscriptionConfig**でカスタマイズ可能:

  - `tuning`: チューニング設定（現在は E_standard、Drop_D 対応予定）
  - `sample_rate`: サンプリングレート（デフォルト: 44100Hz）
  - `min_pitch` / `max_pitch`: 検出するピッチ範囲（MIDI 番号）

---

## 今後の予定

- Drop D などの変則チューニング対応
- 運指最適化ロジックの改善
- TAB フォーマットの精度向上
- LilyPond 出力の改善（ポジション・記号・レイアウト調整など）
- 生成された SVG / PDF を使った Web フロントエンド向け UI のサンプル追加
