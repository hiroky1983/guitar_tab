# AIエージェント向け 自動改善サイクル実行手順書

このドキュメントは、guitartab-transcriber を使ってギターTAB譜を生成し、正解データと比較しながら自動的に改善を繰り返すための手順書です。

## 目的

YouTube URLから生成したTAB譜を、正解データ（image.png）と比較して適合率を計算し、出力を正解データに近づけるように改善を繰り返す。

## 前提条件

- Python環境がセットアップ済み
- 必要なパッケージがインストール済み（`pip install -e .`）
- LilyPondがインストール済み
- 正解データ: `image.png` （「ギリギリchop」のTAB譜）
- 対象YouTube URL: `https://www.youtube.com/watch?v=wr7xTGTG-Mo&list=RDwr7xTGTG-Mo&start_radio=1`

## 自動改善サイクルの手順

### ステップ1: 現在の実装でTAB譜を生成

```bash
python main.py --url "https://www.youtube.com/watch?v=wr7xTGTG-Mo&list=RDwr7xTGTG-Mo&start_radio=1"
```

**出力ファイル:**
- `result.ly` - LilyPond記法
- `score.svg` - 生成されたTAB譜の画像（SVG形式）
- コンソール出力 - ASCII TAB

### ステップ2: 正解データの読み込みと理解

正解データ: `image.png`

**正解データから読み取るべき情報:**
- 曲名: ギリギリchop
- 拍子記号: 4/4
- テンポ（BPM）: 約117
- 使用する弦とフレット番号
- 音符のタイミングと長さ
- フレーズの構造
- パームミュート（P.M.）の位置
- アーティキュレーション記号

### ステップ3: 生成結果と正解データの比較

**比較する項目:**

1. **音符の正確性**
   - 正解: 各時刻における弦番号とフレット番号
   - 生成: 生成されたTAB譜の弦番号とフレット番号
   - 評価: 一致率（％）

2. **タイミングの正確性**
   - 正解: 各音符の開始時刻と終了時刻
   - 生成: 生成された各音符の開始時刻と終了時刻
   - 評価: 時間的な誤差（秒）

3. **運指の適切性**
   - 正解: 実際に演奏しやすい運指
   - 生成: アルゴリズムが選択した運指
   - 評価: 運指の妥当性（主観的だが重要）

4. **全体の適合率**
   - 計算式: `(一致した音符数) / (正解の総音符数) × 100`

### ステップ4: 適合率の計算方法

**基本的な適合率計算:**

```python
def calculate_accuracy(ground_truth, generated):
    """
    正解データと生成データを比較して適合率を計算

    Args:
        ground_truth: 正解のTABデータ（時刻、弦、フレットのリスト）
        generated: 生成されたTABデータ（時刻、弦、フレットのリスト）

    Returns:
        accuracy: 適合率（0.0 - 1.0）
    """
    matches = 0
    total = len(ground_truth)

    for gt_note in ground_truth:
        # 時刻の許容誤差: ±0.05秒
        time_tolerance = 0.05

        # 生成データから近い時刻の音符を探す
        for gen_note in generated:
            if abs(gt_note['time'] - gen_note['time']) < time_tolerance:
                # 弦とフレットが一致するかチェック
                if (gt_note['string'] == gen_note['string'] and
                    gt_note['fret'] == gen_note['fret']):
                    matches += 1
                    break

    accuracy = matches / total if total > 0 else 0.0
    return accuracy
```

### ステップ5: 改善すべきパラメータの特定

**調整可能なパラメータ:**

1. **音声認識パラメータ（Basic Pitch）**
   - `min_pitch`: 検出する最低ピッチ（MIDI番号）
   - `max_pitch`: 検出する最高ピッチ（MIDI番号）
   - `onset_threshold`: 音の立ち上がり検出の閾値
   - `frame_threshold`: フレーム検出の閾値

2. **運指決定アルゴリズム**
   - 弦の優先順位（現在: 太い弦優先）
   - フレット範囲の制限（現在: 0-20）
   - ポジション移動のペナルティ
   - 開放弦の優先度

3. **タイミング調整**
   - BPMの推定精度
   - 量子化の閾値
   - ノートの最小長さ

4. **フィルタリング**
   - ノイズ除去の閾値
   - 倍音除去のロジック
   - 音量の閾値

### ステップ6: 改善の実施

**改善アプローチ:**

1. **パラメータチューニング**
   - 適合率が最も向上するパラメータを探索
   - 例: onset_threshold を 0.3 から 0.5 に変更

2. **アルゴリズムの修正**
   - 運指決定ロジックの改善
   - 例: 連続する音符のポジション移動を最小化

3. **前処理の追加**
   - 音声のノイズ除去
   - ドラムや他の楽器の分離

4. **後処理の追加**
   - 明らかな誤りの修正
   - リズムの量子化

### ステップ7: 改善結果の検証

```bash
# 改善後に再度実行
python main.py --url "https://www.youtube.com/watch?v=wr7xTGTG-Mo&list=RDwr7xTGTG-Mo&start_radio=1"

# 新しい適合率を計算
# 前回の適合率と比較
```

**記録すべき情報:**
- 改善前の適合率: X%
- 改善後の適合率: Y%
- 改善幅: (Y - X)%
- 変更したパラメータ/ロジック
- 副作用（他の部分への影響）

### ステップ8: 繰り返し

適合率が目標値（例: 90%以上）に達するまで、ステップ1-7を繰り返す。

**終了条件:**
- 適合率が90%以上に到達
- または、10回連続で改善が見られない場合

## 実装例: 自動改善スクリプト

```python
#!/usr/bin/env python3
"""
自動改善サイクルを実行するスクリプト
"""

import subprocess
import json
from pathlib import Path

# 設定
YOUTUBE_URL = "https://www.youtube.com/watch?v=wr7xTGTG-Mo&list=RDwr7xTGTG-Mo&start_radio=1"
GROUND_TRUTH_FILE = "image.png"
MAX_ITERATIONS = 50
TARGET_ACCURACY = 0.90

def run_transcription():
    """TAB譜生成を実行"""
    result = subprocess.run(
        ["python", "main.py", "--url", YOUTUBE_URL],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def extract_ground_truth(image_path):
    """
    正解画像からTABデータを抽出（手動で作成するか、OCRを使用）

    Returns:
        List[Dict]: [{'time': 0.0, 'string': 6, 'fret': 0}, ...]
    """
    # TODO: 画像からTABデータを抽出
    # 現時点では手動で作成したJSONファイルから読み込む
    with open("ground_truth.json", "r") as f:
        return json.load(f)

def parse_generated_tab():
    """
    生成されたTABデータを解析

    Returns:
        List[Dict]: [{'time': 0.0, 'string': 6, 'fret': 0}, ...]
    """
    # result.lyまたはJSONから生成データを読み込む
    # TODO: 実装
    pass

def calculate_accuracy(ground_truth, generated):
    """適合率を計算"""
    matches = 0
    total = len(ground_truth)
    time_tolerance = 0.05

    for gt_note in ground_truth:
        for gen_note in generated:
            if abs(gt_note['time'] - gen_note['time']) < time_tolerance:
                if (gt_note['string'] == gen_note['string'] and
                    gt_note['fret'] == gen_note['fret']):
                    matches += 1
                    break

    return matches / total if total > 0 else 0.0

def suggest_improvements(accuracy, ground_truth, generated):
    """
    適合率と差分から改善案を提案

    Returns:
        Dict: 改善案
    """
    improvements = {
        "parameters": {},
        "algorithm_changes": [],
        "notes": []
    }

    # 適合率が低い場合の分析
    if accuracy < 0.5:
        improvements["notes"].append("適合率が非常に低い。Basic Pitchの閾値を調整する必要がある")
        improvements["parameters"]["onset_threshold"] = 0.5
    elif accuracy < 0.7:
        improvements["notes"].append("運指の選択に問題がある可能性")
        improvements["algorithm_changes"].append("ポジション移動ペナルティの導入")

    return improvements

def apply_improvements(improvements):
    """
    改善案を適用

    Args:
        improvements: suggest_improvements()の出力
    """
    # TODO: パラメータファイルまたはコードを更新
    print("改善案を適用:")
    print(json.dumps(improvements, indent=2, ensure_ascii=False))

def main():
    """自動改善サイクルのメイン処理"""
    print("=== 自動改善サイクル開始 ===")

    # 正解データの読み込み
    ground_truth = extract_ground_truth(GROUND_TRUTH_FILE)
    print(f"正解データ: {len(ground_truth)} 個の音符")

    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- イテレーション {iteration} ---")

        # TAB譜生成
        print("TAB譜を生成中...")
        if not run_transcription():
            print("エラー: TAB譜生成に失敗")
            continue

        # 生成データの解析
        generated = parse_generated_tab()

        # 適合率の計算
        accuracy = calculate_accuracy(ground_truth, generated)
        print(f"適合率: {accuracy * 100:.2f}%")

        # 履歴に記録
        history.append({
            "iteration": iteration,
            "accuracy": accuracy
        })

        # 目標達成チェック
        if accuracy >= TARGET_ACCURACY:
            print(f"\n目標達成！ 適合率: {accuracy * 100:.2f}%")
            break

        # 改善が停滞しているかチェック
        if len(history) >= 10:
            recent_accuracies = [h["accuracy"] for h in history[-10:]]
            if max(recent_accuracies) - min(recent_accuracies) < 0.01:
                print("\n改善が停滞しています。終了します。")
                break

        # 改善案の提案
        improvements = suggest_improvements(accuracy, ground_truth, generated)

        # 改善の適用
        apply_improvements(improvements)

    print("\n=== 自動改善サイクル終了 ===")
    print(f"最終適合率: {history[-1]['accuracy'] * 100:.2f}%")

if __name__ == "__main__":
    main()
```

## 重要な注意事項

### 正解データの準備

image.pngから手動でTABデータを抽出し、`ground_truth.json`として保存する必要があります。

```json
[
  {"time": 0.0, "string": 6, "fret": 0, "duration": 0.5},
  {"time": 0.5, "string": 5, "fret": 0, "duration": 0.5},
  {"time": 1.0, "string": 4, "fret": 0, "duration": 0.5}
]
```

### 改善の優先順位

1. **第1優先: 音符の検出精度**
   - Basic Pitchのパラメータ調整
   - 前処理（ノイズ除去）

2. **第2優先: タイミングの精度**
   - BPM推定の改善
   - 量子化の調整

3. **第3優先: 運指の適切性**
   - 運指決定アルゴリズムの改善
   - ポジション移動の最適化

### デバッグのヒント

- `score.svg`と`image.png`を並べて視覚的に比較
- 大きく外れている部分から優先的に修正
- 1つのパラメータずつ変更して影響を確認
- 各イテレーションの結果をログに記録

## チェックリスト

AIエージェントが実行する際の確認事項:

- [x] Python環境が利用可能
- [x] 必要なパッケージがインストール済み
- [x] LilyPondがインストール済み
- [x] 正解データ（image.png）が存在
- [x] ground_truth.jsonが準備済み（または作成方法が確立）
- [x] 改善結果を記録する仕組みがある
- [x] 目標適合率が設定されている

## 実行履歴

### Iteration 1 (2026-01-07 22:10)
- 初回実行
- YouTube サービスが利用不可のため、適合率測定不可
- 環境確認とセットアップ完了

### Iteration 2 (2026-01-07 22:15)
- Basic Pitch パラメータを最適化 (onset_threshold: 0.4→0.35, frame_threshold: 0.3→0.25)
- 倍音補正アルゴリズムを v2 から v3 にアップグレード
- ノイズフィルタリングを改善（duration閾値: 0.05→0.03, velocity閾値追加）
- 運指決定アルゴリズムに開放弦ボーナス追加
- 詳細: `IMPROVEMENT_REPORT.md` 参照

## トラブルシューティング

### 適合率が全く上がらない

- Basic Pitchが正しく音を検出できているか確認
- 音声ファイルの品質を確認
- 正解データが正しいか再確認

### 改善が特定の値で停滞

- 局所最適解に陥っている可能性
- パラメータの探索範囲を広げる
- アルゴリズム自体を見直す

### 生成速度が遅い

- YouTubeダウンロードは初回のみ（キャッシュを活用）
- Basic Pitchの推論はGPU利用を検討
- 並列処理の導入
