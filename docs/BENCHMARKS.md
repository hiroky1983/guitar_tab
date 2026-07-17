# ベンチマーク実測記録

開発ルール（docs/DESIGN.md）に従い、本ファイルには `benchmark.py` による**実測値のみ**を記録する。
「期待精度」「見込み」は書かない。

---

## 2026-07-17: basic-pitch ベースライン（M0）

### 条件

| 項目 | 値 |
|---|---|
| エンジン | basic-pitch 0.4.0（CoreML バックエンド、`predict()` デフォルトパラメータ） |
| 実行環境 | 専用 venv `.venv-basicpitch`（CPython 3.10.12、numba 0.60 / llvmlite 0.43 / setuptools 80 にピン、README 参照） |
| データ | GuitarSet solo 10トラック（`eval_data/guitarset/`、audio_mono-mic 変種、JAMS アノテーション無改変） |
| 前処理 | なし（Demucs 分離なし、mic 録音 WAV を直接転写） |
| 評価 | `mir_eval.transcription`、onset tolerance = 50ms、pitch = 同一半音、offset 不評価（`offset_ratio=None`） |
| コマンド | `uv run python -m guitartab eval` |

### トラック別結果（note-level P/R/F1）

| track | style | P | R | F1 | ref | est |
|---|---|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | Bossa Nova | 0.535 | 0.962 | 0.688 | 79 | 142 |
| 00_Rock2-85-F_solo | Rock | 0.716 | 0.920 | 0.805 | 137 | 176 |
| 01_Funk1-97-C_solo | Funk | 0.777 | 0.837 | 0.806 | 104 | 112 |
| 02_Jazz2-110-Bb_solo | Jazz | 0.818 | 1.000 | 0.900 | 54 | 66 |
| 02_SS1-68-E_solo | Singer-Songwriter | 0.505 | 0.771 | 0.610 | 70 | 107 |
| 03_Jazz1-200-B_solo | Jazz | 0.800 | 0.837 | 0.818 | 43 | 45 |
| 04_Funk2-119-G_solo | Funk | 0.806 | 0.702 | 0.750 | 124 | 108 |
| 04_Rock3-148-C_solo | Rock | 0.747 | 0.873 | 0.805 | 71 | 83 |
| 05_BN2-166-Ab_solo | Bossa Nova | 0.884 | 0.910 | 0.897 | 67 | 69 |
| 05_SS2-107-Ab_solo | Singer-Songwriter | 0.897 | 0.897 | 0.897 | 87 | 87 |
| **mean (10)** | | **0.748** | **0.871** | **0.798** | | |

実行時間: 10トラックで約27秒（1トラック約2.7秒、トラックごとにサブプロセスでモデルロード込み。
初回のみ CoreML モデルコンパイルで +20秒程度）。

### エラーモード分析（同一出力に対する追加実測）

| track | F1@50ms | F1@100ms | F1@50ms chroma（オクターブ許容） |
|---|---:|---:|---:|
| 00_BN3-119-G_solo | 0.688 | 0.697 | 0.688 |
| 02_SS1-68-E_solo | 0.610 | 0.701 | 0.621 |
| 04_Funk2-119-G_solo | 0.750 | 0.750 | 0.784 |
| （他7トラック） | 0.805–0.900 | 変化 +0.00〜+0.04 | ほぼ変化なし |

観察（実測に基づく）:

- **Recall（0.871）> Precision（0.748）**: 支配的な誤りは取りこぼしではなく過検出。
  worst 2トラック（00_BN3: est 142 vs ref 79、02_SS1: est 107 vs ref 70）は est 数が ref の 1.5倍超。
- **オクターブ誤りは主因ではない**: chroma 折り畳みで F1 はほぼ不変（最大 +0.034、04_Funk2 のみ）。
- **02_SS1（遅いフィンガースタイル）は onset ずれ + 低音側スプリアスが複合**:
  tolerance を 100ms に緩めると F1 が 0.610→0.701 に回復。また MIDI 52（E3）未満の
  est ノートが 22 個あるのに対し ref は 1 個で、低音域の湧き出しが顕著。
  00_Rock2 も同様の傾向（est 16 / ref 5）。
- M1 ゲート（クリーンギター note F1 ≥ 0.80）に対し、basic-pitch ベースラインは
  mean F1 = 0.798 で未達。10トラック中6トラックは単体で 0.80 以上。
