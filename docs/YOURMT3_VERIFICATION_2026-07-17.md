# YourMT3+ Apple Silicon 実機検証結果（2026-07-17）

予備転写エンジンのフィージビリティ検証記録。検証環境: M2 / 24GB / macOS。

## 結論

**合格 — MuScriptor 不調・ブロック時の代替として現実的。**
MuScriptor と違い**ゲート承認・アカウント不要**で今すぐ使える。

## 検証結果サマリ

| 項目 | 結果 |
|---|---|
| チェックポイント | GitHub リポジトリにコードなし（README のみ）。実体は HF Space `mimbres/YourMT3`（gated: false）。5モデル配布、論文ベストは YPTF.MoE+Multi noPS（562MB） |
| セットアップ | 成功。`.venv-yourmt3`（Python 3.11、transformers==4.45.1 / numpy==1.26.4 固定）。torchaudio.load → soundfile 置換が必要だった1点のみ |
| Apple Silicon | MPS 完走。ただし自己回帰デコードがボトルネックで CPU とほぼ同速（32.3秒音源を約18-19秒、実時間比 ~1.7x） |
| 精度（概算） | GuitarSet 00_BN3 1トラックで **P 0.975 / R 0.987 / F1 0.981**（onset 50ms）。**ただし GuitarSet は YourMT3+ の学習データに含まれるため汚染の可能性大**。実力比較は GuitarSet 外（歪み音源等）で行うこと |
| NoteEvent 変換 | 容易。`Note{onset, offset, pitch, velocity, program}` → NoteEvent へ機械的変換（confidence=1.0 固定）。ギターは program 24 で出力された |
| ライセンス | **要注意**: GitHub は GPL-3.0、HF Space コードのヘッダは Apache-2.0 と混在。安全側 = GPL-3.0 前提で、コード同梱せず「別途DL + 別プロセス CLI」の疎結合に。本格採用前に著者確認を推奨 |

## 弱点

歪みエレキの明示サポートなし（学習データは MT3 系 + GuitarSet 中心）。歪み性能は M3 で MuScriptor / basic-pitch と実測比較が必要。

## 統合方針

`_basicpitch_runner.py` と同じ別 venv サブプロセス方式で `transcribe/yourmt3.py` を実装可能。
コード+チェックポイントはスクラッチパッド（セッション消滅で失われる）にあるため、統合時に gitignore 済みのプロジェクト内ディレクトリへ移すこと。
