# AIエージェント向け実行手順書

このドキュメントは、guitartab-transcriber を使ってギターTAB譜を生成する作業を、AIエージェントが繰り返し実行するための手順書です。

## 前提条件

- Python環境がセットアップ済み
- 必要なパッケージがインストール済み（`pip install -e .`）
- LilyPondがインストール済み（高品質な譜面出力が必要な場合）

## 標準的な実行フロー

### 1. YouTube URLからTAB譜を生成する場合

```python
from guitartab_transcriber import Transcriber
import shutil

# Transcriber インスタンスを作成
t = Transcriber()

# YouTube URLを指定
url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# TAB譜を生成
tab = t.transcribe_from_youtube(url)

# コンソールにASCII TABを出力
print(tab.to_text())

# 画像として保存（Matplotlib）
tab.to_matplotlib("result.png")

# LilyPond記法（.ly）を生成
ly_file = tab.to_lilypond("result.ly", title="Guitar TAB")

# LilyPondでSVG生成（LilyPondがインストール済みの場合）
lilypond_path = shutil.which("lilypond")
if lilypond_path:
    svg_file = tab.to_lilypond(
        "result.ly",
        title="Guitar TAB",
        compile_output="score.svg",
        lilypond_executable=lilypond_path
    )
    print(f"Generated: {svg_file}")
```

### 2. ローカル音声ファイルからTAB譜を生成する場合

```python
from guitartab_transcriber import Transcriber

t = Transcriber()

# ローカルファイルを指定
tab = t.transcribe("path/to/audio.wav")

# 以降の処理は上記と同じ
print(tab.to_text())
tab.to_matplotlib("result.png")
tab.to_lilypond("result.ly", title="Guitar TAB")
```

## 出力形式の選択ガイド

| 目的 | メソッド | 出力形式 |
|------|----------|----------|
| コンソールでの確認 | `tab.to_text()` | ASCII TAB |
| プログラムでの利用 | `tab.to_json()` | JSON |
| 簡易的な可視化 | `tab.to_matplotlib("file.png")` | PNG画像 |
| 高品質な譜面 | `tab.to_lilypond("file.ly", ...)` | .ly → SVG/PNG/PDF |

## エラーハンドリング

### よくあるエラーと対処法

1. **YouTube URLが無効**
   - エラー: `yt-dlp` が動画をダウンロードできない
   - 対処: URLの形式を確認、または動画の公開状態を確認

2. **音声ファイルが見つからない**
   - エラー: `FileNotFoundError`
   - 対処: ファイルパスが正しいか確認

3. **LilyPondが見つからない**
   - エラー: `shutil.which("lilypond")` が `None` を返す
   - 対処: LilyPondをインストール、または .ly ファイルの生成のみで終了

4. **音声認識に失敗**
   - エラー: Basic Pitch が MIDI を生成できない
   - 対処: 音声ファイルの品質を確認、またはギター音が含まれているか確認

## 自動化スクリプトのテンプレート

複数のYouTube URLを一括処理する場合:

```python
from guitartab_transcriber import Transcriber
import shutil
import os

def process_youtube_url(url, output_prefix):
    """YouTube URLからTAB譜を生成"""
    try:
        t = Transcriber()
        print(f"Processing: {url}")

        # TAB譜生成
        tab = t.transcribe_from_youtube(url)

        # 各種形式で出力
        print(tab.to_text())
        tab.to_matplotlib(f"{output_prefix}.png")

        ly_file = tab.to_lilypond(f"{output_prefix}.ly", title=output_prefix)

        # LilyPondでSVG生成
        lilypond_path = shutil.which("lilypond")
        if lilypond_path:
            tab.to_lilypond(
                f"{output_prefix}.ly",
                title=output_prefix,
                compile_output=f"{output_prefix}.svg",
                lilypond_executable=lilypond_path
            )
            print(f"✓ Generated: {output_prefix}.svg")
        else:
            print(f"✓ Generated: {output_prefix}.ly (LilyPond not found)")

        return True

    except Exception as e:
        print(f"✗ Error processing {url}: {e}")
        return False

# 実行例
urls = [
    ("https://www.youtube.com/watch?v=VIDEO_ID_1", "song1"),
    ("https://www.youtube.com/watch?v=VIDEO_ID_2", "song2"),
]

for url, prefix in urls:
    process_youtube_url(url, prefix)
```

## カスタマイズオプション

### TranscriptionConfig を使った設定

```python
from guitartab_transcriber import Transcriber, TranscriptionConfig

# カスタム設定
config = TranscriptionConfig(
    tuning="E_standard",  # チューニング
    sample_rate=44100,     # サンプリングレート
    min_pitch=40,          # 検出する最低ピッチ (MIDI番号)
    max_pitch=88,          # 検出する最高ピッチ (MIDI番号)
)

t = Transcriber(config=config)
tab = t.transcribe("audio.wav")
```

## チェックリスト

AIエージェントが実行する際の確認事項:

- [ ] Python環境が利用可能か
- [ ] 必要なパッケージがインストール済みか
- [ ] 入力ソース（URL or ファイルパス）が有効か
- [ ] 出力先ディレクトリが存在するか
- [ ] LilyPondが必要な場合、インストール済みか
- [ ] エラーハンドリングが実装されているか

## 技術的な注意点

### 音声認識の仕組み

- **Basic Pitch (Spotify製)** を使用
- MIDI番号 40-88 の範囲を検出（ギター音域）
- E標準チューニング対応（6弦: E2=40, 5弦: A2=45, 4弦: D3=50, 3弦: G3=55, 2弦: B3=59, 1弦: E4=64）
- フレット範囲: 0-20

### LilyPond連携の責務分離

1. **ライブラリの責務**: `tab.to_lilypond()` で .ly ファイルを生成
2. **外部ツールの責務**: LilyPondが .ly から SVG/PNG/PDF を生成

### パフォーマンス

- YouTube動画は yt-dlp で一括ダウンロード（ストリーミング再生ではない）
- ダウンロード進捗は標準エラー出力に表示
- 長い動画ほど処理時間が増加

## トラブルシューティング

### 出力が空の場合

- 音声にギター音が含まれているか確認
- MIDI番号 40-88 の範囲外の音は自動フィルタリングされる
- Basic Pitch が認識できる音質か確認

### 運指が不自然な場合

- 現在の運指決定アルゴリズムは「太い弦（低音弦）優先」
- 今後の改善予定あり
- 必要に応じて JSON 出力を加工して調整

### LilyPond出力がエラーになる場合

- LilyPondのバージョンを確認
- .ly ファイルの構文エラーがないか確認
- `lilypond` コマンドが PATH に通っているか確認
