#!/usr/bin/env python3
"""
自動改善サイクルを実行するスクリプト
"""

import subprocess
import json
from pathlib import Path
import sys

# 設定
YOUTUBE_URL = "https://www.youtube.com/watch?v=wr7xTGTG-Mo&list=RDwr7xTGTG-Mo&start_radio=1"
GROUND_TRUTH_FILE = "ground_truth.json"
MAX_ITERATIONS = 50
TARGET_ACCURACY = 0.90
HISTORY_FILE = "improvement_history.json"


def run_transcription():
    """TAB譜生成を実行"""
    result = subprocess.run(
        ["python", "main.py", "--url", YOUTUBE_URL],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.returncode == 0


def load_ground_truth():
    """
    正解データをJSONファイルから読み込む
    """
    with open(GROUND_TRUTH_FILE, "r") as f:
        return json.load(f)


def parse_generated_tab():
    """
    生成されたTABデータを解析
    result.lyファイルから解析するか、TabResultから直接取得
    """
    # TODO: 実際にはresult.lyを解析するか、TabResultをJSON出力する機能が必要
    # 仮実装: 空のリストを返す
    return []


def calculate_accuracy(ground_truth, generated):
    """適合率を計算"""
    if not generated:
        return 0.0

    matches = 0
    total = len(ground_truth)
    time_tolerance = 0.05  # 50ms

    for gt_note in ground_truth:
        for gen_note in generated:
            if abs(gt_note['time'] - gen_note['time']) < time_tolerance:
                if (gt_note['string'] == gen_note['string'] and
                    gt_note['fret'] == gen_note['fret']):
                    matches += 1
                    break

    return matches / total if total > 0 else 0.0


def suggest_improvements(accuracy, iteration, history):
    """
    適合率と履歴から改善案を提案
    """
    improvements = {
        "parameters": {},
        "algorithm_changes": [],
        "notes": []
    }

    # 適合率に基づく分析
    if accuracy < 0.3:
        improvements["notes"].append("適合率が非常に低い。Basic Pitchの閾値を調整する必要がある")
        improvements["parameters"]["onset_threshold"] = 0.3
        improvements["parameters"]["frame_threshold"] = 0.2
    elif accuracy < 0.5:
        improvements["notes"].append("音符検出精度が低い。閾値とピッチ範囲を見直す")
        improvements["parameters"]["onset_threshold"] = 0.35
        improvements["parameters"]["minimum_frequency"] = 35.0
    elif accuracy < 0.7:
        improvements["notes"].append("運指の選択に問題がある可能性")
        improvements["algorithm_changes"].append("ポジション移動ペナルティの導入")
    else:
        improvements["notes"].append("細かい調整を行う")
        improvements["algorithm_changes"].append("タイミングの微調整")

    # 停滞チェック
    if len(history) >= 5:
        recent = [h["accuracy"] for h in history[-5:]]
        if max(recent) - min(recent) < 0.02:
            improvements["notes"].append("改善が停滞。アルゴリズム変更を推奨")
            improvements["algorithm_changes"].append("運指決定ロジックの根本的見直し")

    return improvements


def save_history(history):
    """履歴をファイルに保存"""
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, indent=2, ensure_ascii=False, fp=f)


def load_history():
    """履歴をファイルから読み込む"""
    if Path(HISTORY_FILE).exists():
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def main():
    """自動改善サイクルのメイン処理"""
    print("=== 自動改善サイクル開始 ===")
    print(f"目標適合率: {TARGET_ACCURACY * 100}%")
    print(f"最大イテレーション: {MAX_ITERATIONS}")

    # 正解データの読み込み
    try:
        ground_truth = load_ground_truth()
        print(f"正解データ: {len(ground_truth)} 個の音符")
    except FileNotFoundError:
        print(f"エラー: {GROUND_TRUTH_FILE} が見つかりません")
        sys.exit(1)

    # 履歴の読み込み
    history = load_history()
    start_iteration = len(history) + 1

    for iteration in range(start_iteration, MAX_ITERATIONS + 1):
        print(f"\n{'='*60}")
        print(f"イテレーション {iteration}")
        print(f"{'='*60}")

        # TAB譜生成
        print("TAB譜を生成中...")
        if not run_transcription():
            print("エラー: TAB譜生成に失敗しました")
            print("YouTube が利用できない可能性があります")
            print("スキップして次のステップに進みます")

            # 仮の低スコアを記録
            accuracy = 0.0
        else:
            # 生成データの解析
            generated = parse_generated_tab()

            # 適合率の計算
            accuracy = calculate_accuracy(ground_truth, generated)

        print(f"\n適合率: {accuracy * 100:.2f}%")

        # 履歴に記録
        history.append({
            "iteration": iteration,
            "accuracy": accuracy,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
        save_history(history)

        # 目標達成チェック
        if accuracy >= TARGET_ACCURACY:
            print(f"\n{'='*60}")
            print(f"🎉 目標達成！ 適合率: {accuracy * 100:.2f}%")
            print(f"{'='*60}")
            break

        # 改善が停滞しているかチェック
        if len(history) >= 10:
            recent_accuracies = [h["accuracy"] for h in history[-10:]]
            if max(recent_accuracies) - min(recent_accuracies) < 0.01:
                print(f"\n{'='*60}")
                print("改善が停滞しています。終了します。")
                print(f"{'='*60}")
                break

        # 改善案の提案
        print("\n--- 改善案 ---")
        improvements = suggest_improvements(accuracy, iteration, history)
        print(json.dumps(improvements, indent=2, ensure_ascii=False))

        print("\n次のイテレーションで改善を適用します...")

    # 最終レポート
    print(f"\n{'='*60}")
    print("=== 自動改善サイクル終了 ===")
    print(f"{'='*60}")
    if history:
        print(f"総イテレーション数: {len(history)}")
        print(f"初期適合率: {history[0]['accuracy'] * 100:.2f}%")
        print(f"最終適合率: {history[-1]['accuracy'] * 100:.2f}%")
        print(f"改善幅: {(history[-1]['accuracy'] - history[0]['accuracy']) * 100:.2f}%")

        print(f"\n履歴は {HISTORY_FILE} に保存されました")
    else:
        print("実行されたイテレーションがありません")


if __name__ == "__main__":
    main()
