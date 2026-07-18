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

---

## 2026-07-17: basic-pitch ネイティブパラメータスイープ（M1）

### 条件

ベースラインと同一条件（同一10トラック・onset tolerance 50ms・前処理なし）で、
`basic_pitch.inference.predict()` の**ネイティブ推論パラメータのみ**を変更
（自作の後処理フィルタは追加していない）。計 24 回のベンチ実行。
コマンド例:

```
uv run python -m guitartab eval \
  --bp-onset-threshold 0.75 --bp-frame-threshold 0.4 --bp-minimum-note-length 100
```

### 試行の要約（mean P/R/F1、各行が1回のベンチ実測）

デフォルト: onset_threshold 0.5 / frame_threshold 0.3 / minimum_note_length 127.7ms /
minimum_frequency None / melodia_trick True。

| 構成（デフォルトからの変更のみ） | P | R | F1 |
|---|---:|---:|---:|
| （デフォルト = ベースライン再現） | 0.748 | 0.871 | 0.798 |
| onset 0.6 | 0.781 | 0.871 | 0.817 |
| onset 0.65 | 0.799 | 0.871 | 0.828 |
| onset 0.7 | 0.808 | 0.867 | 0.831 |
| onset 0.75 | 0.812 | 0.865 | 0.833 |
| onset 0.8 | 0.810 | 0.851 | 0.826 |
| frame 0.4 | 0.793 | 0.841 | 0.807 |
| frame 0.5 | 0.821 | 0.780 | 0.790 |
| min_note_length 100ms | 0.731 | 0.902 | 0.801 |
| min_note_length 160ms | 0.759 | 0.779 | 0.756 |
| min_frequency 75Hz | 0.750 | 0.871 | 0.798 |
| melodia_trick False | 0.776 | 0.844 | 0.803 |
| onset 0.7 + frame 0.4 | 0.854 | 0.839 | 0.840 |
| onset 0.7 + min_len 100 | 0.787 | 0.894 | 0.833 |
| onset 0.8 + min_len 100 | 0.790 | 0.877 | 0.827 |
| onset 0.7 + frame 0.4 + min_len 100 | 0.847 | 0.880 | 0.859 |
| onset 0.65 + frame 0.4 + min_len 100 | 0.835 | 0.883 | 0.854 |
| **onset 0.75 + frame 0.4 + min_len 100** | **0.855** | **0.880** | **0.864** |
| onset 0.8 + frame 0.4 + min_len 100 | 0.855 | 0.868 | 0.858 |
| onset 0.7 + frame 0.45 + min_len 100 | 0.855 | 0.868 | 0.858 |
| onset 0.75 + frame 0.45 + min_len 100 | 0.863 | 0.869 | 0.862 |
| onset 0.7 + frame 0.35 + min_len 100 | 0.819 | 0.887 | 0.848 |
| onset 0.7 + frame 0.4 + min_len 80 | 0.822 | 0.899 | 0.855 |
| onset 0.7 + frame 0.4 + min_len 110 | 0.847 | 0.880 | 0.859 |

観察（実測に基づく）:

- 単因子では onset_threshold 引き上げ（過検出抑制 → P 改善）が最も効いた。
- frame_threshold 0.4 と min_note_length 100ms は単体では中立〜微減だが、
  onset 引き上げとの組合せで P と R が同時に改善した（0.831 → 0.859〜0.864）。
- minimum_frequency 75Hz は変化なし（E2 未満の推定ノートがほぼ存在しないため。
  ベースラインで観察した低音側スプリアスは E2〜E3 帯であり、実音域と重なるため
  ネイティブパラメータでは切れない）。

### 最良構成のトラック別結果（onset 0.75 / frame 0.4 / min_note_length 100ms）

| track | P | R | F1 | ref | est |
|---|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | 0.776 | 0.962 | 0.859 | 79 | 98 |
| 00_Rock2-85-F_solo | 0.824 | 0.920 | 0.869 | 137 | 153 |
| 01_Funk1-97-C_solo | 0.819 | 0.827 | 0.823 | 104 | 105 |
| 02_Jazz2-110-Bb_solo | 0.929 | 0.963 | 0.945 | 54 | 56 |
| 02_SS1-68-E_solo | 0.671 | 0.814 | 0.735 | 70 | 85 |
| 03_Jazz1-200-B_solo | 0.864 | 0.884 | 0.874 | 43 | 44 |
| 04_Funk2-119-G_solo | 0.970 | 0.774 | 0.861 | 124 | 99 |
| 04_Rock3-148-C_solo | 0.808 | 0.831 | 0.819 | 71 | 73 |
| 05_BN2-166-Ab_solo | 0.926 | 0.940 | 0.933 | 67 | 68 |
| 05_SS2-107-Ab_solo | 0.963 | 0.885 | 0.922 | 87 | 80 |
| **mean (10)** | **0.855** | **0.880** | **0.864** | | |

ベースライン比: mean F1 0.798 → 0.864（+0.066）、P 0.748 → 0.855（+0.107）、
R 0.871 → 0.880（+0.009）。単体 0.80 未満のトラックは 4 → 1（02_SS1 のみ、0.735）。

### 過適合リスク（重要）

このスイープは **10トラックでチューニングし、同じ10トラックで評価している**。
mean F1 0.864 ≥ 0.80 はこのセット上の値であり、閾値選択が本セットに過適合して
いる可能性がある。**M1 ゲート通過の最終判定は、チューニングに使っていない未見の
GuitarSet トラック（holdout）での検証が必要**。

---

## 2026-07-17: Holdout 検証 — M1 ゲート最終判定

### 条件

| 項目 | 値 |
|---|---|
| データ | GuitarSet solo 8トラック（`eval_data/guitarset_holdout/`、audio_mono-mic 変種、JAMS 無改変） |
| holdout 設計 | dev セットと**進行（lead sheet）レベルで完全非重複**: dev は BN2/BN3・Funk1/Funk2・Jazz1/Jazz2・Rock2/Rock3・SS1/SS2、holdout は **BN1・Funk3・Jazz3・Rock1・SS3** のみ。同一プレイヤー×同一曲の重複なし。スタイル5種カバー、プレイヤー 00〜05 全員を含む |
| 実行回数 | チューニング構成1回 + デフォルト構成1回の**計2回のみ**。holdout 上での閾値再調整は行っていない |
| その他 | dev ベンチと同一（basic-pitch 0.4.0 CoreML、前処理なし、onset tolerance 50ms、offset 不評価） |
| コマンド | `uv run python -m guitartab eval --eval-data eval_data/guitarset_holdout --bp-onset-threshold 0.75 --bp-frame-threshold 0.4 --bp-minimum-note-length 100` |

### チューニング構成（onset 0.75 / frame 0.4 / min_note_length 100ms）のトラック別結果

| track | style | P | R | F1 | ref | est |
|---|---|---:|---:|---:|---:|---:|
| 00_Jazz3-137-Eb_solo | Jazz | 0.762 | 0.968 | 0.853 | 63 | 80 |
| 01_Jazz3-150-C_solo | Jazz | 0.933 | 0.899 | 0.916 | 109 | 105 |
| 02_Funk3-98-A_solo | Funk | 0.826 | 0.962 | 0.889 | 79 | 92 |
| 03_BN1-129-Eb_solo | Bossa Nova | 0.900 | 0.931 | 0.915 | 58 | 60 |
| 03_Rock1-90-C#_solo | Rock | 0.905 | 0.893 | 0.899 | 75 | 74 |
| 04_Funk3-112-C#_solo | Funk | 0.882 | 0.833 | 0.857 | 90 | 85 |
| 05_BN1-147-Gb_solo | Bossa Nova | 0.767 | 0.852 | 0.807 | 27 | 30 |
| 05_SS3-98-C_solo | Singer-Songwriter | 0.957 | 0.957 | 0.957 | 93 | 93 |
| **mean (8)** | | **0.867** | **0.912** | **0.887** | | |

### デフォルト構成（参考、同一 holdout・1回実測）

| track | P | R | F1 | ref | est |
|---|---:|---:|---:|---:|---:|
| 00_Jazz3-137-Eb_solo | 0.527 | 0.937 | 0.674 | 63 | 112 |
| 01_Jazz3-150-C_solo | 0.847 | 0.862 | 0.855 | 109 | 111 |
| 02_Funk3-98-A_solo | 0.802 | 0.975 | 0.880 | 79 | 96 |
| 03_BN1-129-Eb_solo | 0.831 | 0.931 | 0.878 | 58 | 65 |
| 03_Rock1-90-C#_solo | 0.815 | 0.880 | 0.846 | 75 | 81 |
| 04_Funk3-112-C#_solo | 0.789 | 0.789 | 0.789 | 90 | 90 |
| 05_BN1-147-Gb_solo | 0.714 | 0.926 | 0.806 | 27 | 35 |
| 05_SS3-98-C_solo | 0.907 | 0.946 | 0.926 | 93 | 97 |
| **mean (8)** | **0.779** | **0.906** | **0.832** | | |

### 判定と観察（実測に基づく）

- **M1 ゲート判定: 通過**。未見 8 トラックでチューニング構成の mean F1 = **0.887 ≥ 0.80**。
  全 8 トラックが単体でも 0.80 以上（最低 0.807、05_BN1）。
- **過適合は軽微**: チューニングによる改善幅は dev で +0.066（0.798→0.864）、
  holdout で +0.055（0.832→0.887）とほぼ同等に保たれた。閾値が dev セット固有の
  癖に効いていたのではなく、汎化する改善であることを示す。
- holdout の mean がデフォルト構成でも dev より高い（0.832 vs 0.798）ことから、
  holdout セット自体がやや易しい可能性はある（dev の worst 02_SS1 のような
  遅いフィンガースタイル曲が holdout の抽選に含まれていない）。
  tuned − default の差分で見る限り、構成選択の妥当性は支持される。
- デフォルト構成の worst（00_Jazz3、F1 0.674、est 112 vs ref 63）はチューニング構成で
  0.853 に回復しており、dev で観察した「過検出が支配的エラー」という診断と一致する。

---

## 2026-07-17: E2E 実走検証（歪みエレキ・参考値）— M3 入口

v2 パイプラインの download / separate ステージ初回実走。対象は v1 の最終ボス曲
B'z「ギリギリchop」（https://www.youtube.com/watch?v=wr7xTGTG-Mo、86.2 秒のショート版 MV）。

### 条件

| 項目 | 値 |
|---|---|
| コマンド | `uv run python -m guitartab transcribe --url "https://www.youtube.com/watch?v=wr7xTGTG-Mo" --bp-onset-threshold 0.75 --bp-frame-threshold 0.4 --bp-minimum-note-length 100`（M1 採用構成） |
| demucs | 4.1.0（本体 venv `.venv` / Python 3.11、torch 2.13.0 + torchaudio 2.11.0 + torchcodec 0.15.0） |
| モデル | htdemucs_6s（HF Hub から safetensors 52MB を初回取得。他モデルは未取得） |
| デバイス | CPU（Apple Silicon。MPS は利用可能だが separate.py は device 指定なし = CPU 実行。86 秒曲で 34 秒と十分速く、現時点で MPS 対応の必要なし） |
| basic-pitch | 0.4.0 CoreML（`.venv-basicpitch` サブプロセス、M1 と同一） |

セットアップ時の非互換1件: torchaudio 2.11 は `torchaudio.load()` が torchcodec 必須に
変わっており、`uv pip install demucs`（+ torchaudio）だけでは separate ステージが
ImportError で落ちる。**torchcodec 0.15.0 の追加インストールで解消**（ffmpeg 8.0.1 で動作
確認。コード変更なし）。README の demucs インストール手順に torchcodec を含めること。

### ステージ実行時間と成果物（実測）

| ステージ | 実行時間 | 成果物 |
|---|---:|---|
| download (yt-dlp) | 4.7 s | `work/wr7xTGTG-Mo/source.wav` 15.2MB（86.2s / 44.1kHz / stereo / PCM_16） |
| separate (htdemucs_6s, CPU) | 34.2 s | `stems/` 6 ステム × 22.8MB（PCM_24）。guitar.wav を後段へ |
| transcribe (basic-pitch) + tab | 6.9 s | `notes.json`（210 ノート・36KB）、`tab.json` 28KB、`tab.txt` 8.7KB |

パイプライン一気通し（download キャッシュ済み・モデル DL 済み）: 37.5 秒。全ステージの
キャッシュ・スキップ動作も確認済み。pytest 48 件全通過（コード変更なし）。

付記: separate を2回実行すると notes 数が 209 → 210 と 1 個ぶれた（demucs CPU 実行の
数値非決定性がステム波形経由で basic-pitch の閾値近傍ノートに影響）。ベンチ比較の際は
ステムを固定して使うこと。

### 凍結 GT との参考比較（これは benchmark.py の正式ベンチではない）

- GT: `eval_data/gt/ground_truth.json`（人間製・凍結・17 ノート・4.6 秒のリフ。読み取りのみ）
- GT は曲中のリフ位置に対応するため、GT 先頭ノートを推定オンセットに対応付ける
  オフセットを 20〜40 秒で総当たり（推定オンセット対応 + 10ms 刻みスイープ）し、
  **ベスト整合のみ**を報告する。オフセットは評価スクリプト内で GT 側時刻に加算しただけで、
  GT・コードには一切焼き込んでいない。
- 今回の 86 秒版 MV ではベストオフセットは **23.62 秒**（v1 の 27.5 秒は別ソース由来の値）。

| 測定 | P | R | F1 |
|---|---:|---:|---:|
| 全推定 210 ノート vs GT 17（tol 50ms） | 0.014 | 0.176 | 0.026 |
| GT 窓内（23.4–28.4s、推定 13 ノート）vs GT 17（tol 50ms） | 0.231 | 0.176 | 0.200 |
| 同上・tol 200ms | 0.231 | 0.176 | 0.200 |
| 同上・chroma（オクターブ許容、tol 50ms） | 0.308 | 0.235 | 0.267 |

マッチは 3/17 で、内訳は開放弦系の低音（MIDI 45/50/64 付近）のみ。GT 後半の
上昇速弾き（MIDI 54→72）は推定側にほぼ存在せず（窓内推定の最高音は 64）、
tol を 200ms に緩めても不変 = タイミングずれではなく**検出自体の失敗**。
v1 実測の到達上限 29.41%（5/17）と整合し、「basic-pitch は歪みエレキの高フレット・
速弾きを検出できない」という v1 の結論を v2 パイプライン上で再確認した。

### tab.txt 冒頭（work/wr7xTGTG-Mo/tab.txt）

```
e|----------------------------------------------------------------------------|
B|----------------------------------------------------------------------------|
G|------------------------------------------------------0---------------------|
D|----------------------------------------------------------------------------|
A|------------------------0---------------------------------------------------|
E|--------------------3-------------------------------------------------------|

e|----------------------------------------------------------------------------|
B|3---------------------------------1-----------------------------------------|
G|----------0-------------------------------2-----------------------------2---|
D|------------0---------------------------0-----------2-----------------------|
A|--------------0---------------------------------------0---------------------|
```

### 観察（実測に基づく）

- パイプライン機構（DL → 分離 → 転写 → tab、キャッシュ、venv 分離）は歪みエレキ実曲でも
  E2E で動作する。実行時間も 86 秒曲で 1 分未満と実用域。
- 分離ステムには帯域外ノートの混入がある（fingering が midi_pitch 29 = F1 を
  ギター音域にクランプする警告 1 件 → ベース帯域のブリード示唆）。
- ボトルネックは M0/M1 の想定どおり転写エンジン。M3 の主作業は
  MuScriptor 等の歪み対応エンジンの実測比較であり、分離・前処理の調整で
  basic-pitch を救える水準（F1 0.2）ではない。

---

## 2026-07-17: エンジン比較 — basic-pitch vs YourMT3+（M3）

YourMT3+（YPTF.MoE+Multi noPS）を `--engine yourmt3` として統合
（`transcribe/yourmt3.py` + `_yourmt3_runner.py`、別 venv `.venv-yourmt3`
サブプロセス方式、コード+チェックポイントは gitignore 済み `third_party/yourmt3/`。
検証記録は docs/YOURMT3_VERIFICATION_2026-07-17.md）。

### 汚染の注記（重要）

**YourMT3+ の学習データは GuitarSet を含む**（MT3 系 + GuitarSet 中心）。したがって
以下の GuitarSet dev セットの数値は学習データ再現の可能性があり**参考値**。
basic-pitch との公平な実力比較にはならない。歪みエレキ（学習外ドメイン）の
参考測定が実力の下限を示す。

### dev セット比較（GuitarSet solo 10トラック・onset tolerance 50ms・1回実測）

| 項目 | 値 |
|---|---|
| コマンド | `uv run python -m guitartab eval --engine basicpitch --engine yourmt3 --bp-onset-threshold 0.75 --bp-frame-threshold 0.4 --bp-minimum-note-length 100` |
| basic-pitch | 0.4.0 CoreML、M1 採用構成（onset 0.75 / frame 0.4 / min_len 100ms） |
| YourMT3+ | YPTF.MoE+Multi noPS、CPU、推論パラメータはデフォルト（チューニングなし） |

| track | bp P | bp R | bp F1 | ymt3 P | ymt3 R | ymt3 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | 0.776 | 0.962 | 0.859 | 0.975 | 0.987 | 0.981 |
| 00_Rock2-85-F_solo | 0.824 | 0.920 | 0.869 | 0.955 | 0.927 | 0.941 |
| 01_Funk1-97-C_solo | 0.819 | 0.827 | 0.823 | 0.843 | 0.875 | 0.858 |
| 02_Jazz2-110-Bb_solo | 0.929 | 0.963 | 0.945 | 1.000 | 0.944 | 0.971 |
| 02_SS1-68-E_solo | 0.671 | 0.814 | 0.735 | 0.926 | 0.900 | 0.913 |
| 03_Jazz1-200-B_solo | 0.864 | 0.884 | 0.874 | 0.854 | 0.953 | 0.901 |
| 04_Funk2-119-G_solo | 0.970 | 0.774 | 0.861 | 0.982 | 0.895 | 0.937 |
| 04_Rock3-148-C_solo | 0.808 | 0.831 | 0.819 | 0.845 | 0.845 | 0.845 |
| 05_BN2-166-Ab_solo | 0.926 | 0.940 | 0.933 | 0.970 | 0.955 | 0.962 |
| 05_SS2-107-Ab_solo | 0.963 | 0.885 | 0.922 | 0.919 | 0.908 | 0.913 |
| **mean (10)** | **0.855** | **0.880** | **0.864** | **0.927** | **0.919** | **0.922** |

- YourMT3+ は 10/10 トラックで basic-pitch チューニング済み構成と同等以上
  （mean F1 0.922 vs 0.864）。ただし上記の汚染注記のとおり参考値。
- 実行時間: 2エンジン×10トラックの eval 一括で 4 分 20 秒（実測。うち basic-pitch は
  従来実測どおり約 27 秒 → yourmt3 は約 3 分 50 秒 = 1トラック約 23 秒、
  トラックごとのサブプロセスでモデルロード込み。歪みエレキ転写ジョブと
  並行実行だったため CPU 競合下の値）。

### 歪みエレキ参考測定（B'z「ギリギリchop」、凍結 GT 17ノート）

E2E セクション（上記）の basic-pitch 測定手順を踏襲: 分離済みステム
`work/wr7xTGTG-Mo/stems/guitar.wav`（同一ファイル）を転写し、凍結 GT
`eval_data/gt/ground_truth.json`（読み取りのみ）とオフセット総当たり
（推定オンセット対応 + 10ms 刻みスイープ、20〜40s）でベスト整合のみを報告。
オフセットは評価スクリプト内で GT 側時刻に加算しただけで、GT・コードには
一切焼き込んでいない。YourMT3+ のベストオフセットは 22.65 秒
（basic-pitch は 23.62 秒。エンジンごとに最良整合が異なる）。

| 測定（GT 窓内 vs GT 17、tol 50ms） | P | R | F1 |
|---|---:|---:|---:|
| basic-pitch（M1 構成、E2E セクション再掲） | 0.231 | 0.176 | 0.200 |
| YourMT3+（窓内推定 13 ノート） | 0.231 | 0.176 | 0.200 |
| YourMT3+ tol 200ms | 0.231 | 0.176 | 0.200 |
| YourMT3+ chroma（オクターブ許容、tol 50ms） | 0.308 | 0.235 | 0.267 |

観察（実測に基づく）:

- **YourMT3+ も歪みエレキでは basic-pitch と同水準（F1 0.200、マッチ 3/17）**。
  エラーモードも同型: GT 後半の上昇速弾き（MIDI 54→72）が推定側にほぼ存在せず
  （窓内推定の最高音 64）、tol 200ms でも不変 = タイミングずれではなく検出自体の失敗。
- 全曲では 333 ノートを検出しており（basic-pitch は 210）、無音ではなく
  「何かは聞こえているが GT のリフとして正しく取れていない」状態。
- 検証記録の弱点欄（「歪みエレキの明示サポートなし」）が実測で確認された。
  クリーンギターの優位（dev 参考値 +0.058）は歪みドメインに転移していない。

---

## 2026-07-17: 合成歪みベンチ — 3エンジンの歪み耐性実測（M3）

歪みエレキの評価が凍結 GT 17ノート×1窓しかない問題への対策として、
合成歪みベンチ `eval_data/distorted_synth/` を構築し、3エンジンで実測した。

### ベンチセットの構成

| 項目 | 値 |
|---|---|
| ソース | `eval_data/guitarset/`（dev 10トラック、読み取りのみ。holdout は不使用） |
| 歪み処理 | pedalboard 0.9.24（Spotify）。crunch（プリゲイン+14dB → tanh drive 18dB → トーン → LPF 5.5kHz）/ highgain（コンプ 4:1 → +26dB → tanh drive 32dB → ハードクリップ −3dB → ミッドスクープ → LPF 4.5kHz）の2強度 × 10トラック = 20クリップ。全パラメータは `eval_data/distorted_synth/README.md` と各 manifest.json に記録、生成スクリプト `make_distorted.py` 同梱 |
| GT | 元 JAMS の**バイト同一コピー**（歪み処理は波形整形・IIR・コンプのみでタイミング/ピッチ不変のため流用可能。これが合成方式の利点） |
| 限界 | **合成歪みであり実アンプ録音ではない**（ソースはアコギ mic 録音）。エンジンの相対比較・劣化傾向の把握用で、実歪みエレキへの一般化は保証されない |
| 歪み度の目安 | クレストファクタ（00_BN3 実測）: 原音 8.7 → crunch 2.4 → highgain 1.4 |

### 条件

| 項目 | 値 |
|---|---|
| コマンド | `uv run python -m guitartab eval --eval-data eval_data/distorted_synth/{crunch,highgain} --engine basicpitch --engine yourmt3 --bp-onset-threshold 0.75 --bp-frame-threshold 0.4 --bp-minimum-note-length 100`（各1回実測） |
| MuScriptor | エンジン未統合のため同条件を別測: small / MPS / `instruments=["distorted_electric_guitar"]` / 生成パラメータデフォルト、`guitartab.eval.metrics` で同一採点（onset tol 50ms、offset 不評価）。20ジョブ計 85.1s、mean RTF 0.139 |
| 評価 | mir_eval、onset tolerance 50ms、pitch 同一半音、offset 不評価（クリーン dev ベンチと同一） |

### mean（10トラック）× 2強度 × 3エンジン

クリーン dev の基準値: basic-pitch tuned 0.864 / YourMT3+ 0.922（学習データ汚染の注記あり・参考値）/
MuScriptor small 0.868（クリーンは acoustic_guitar プロンプト。歪みでは distorted_electric_guitar を使用）。

| エンジン | クリーン dev F1 | crunch P | R | F1（Δクリーン） | highgain P | R | F1（Δクリーン） |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic-pitch（M1 tuned 構成） | 0.864 | 0.821 | 0.858 | **0.836**（−0.028） | 0.781 | 0.742 | **0.757**（−0.107） |
| YourMT3+（YPTF.MoE+Multi noPS） | 0.922 | 0.846 | 0.877 | **0.860**（−0.062） | 0.598 | 0.629 | **0.598**（−0.324） |
| MuScriptor small | 0.868 | 0.781 | 0.778 | **0.772**（−0.096） | 0.508 | 0.727 | **0.592**（−0.276） |

### トラック別 F1（crunch / highgain）

| track | bp cr | bp hg | ymt3 cr | ymt3 hg | ms cr | ms hg |
|---|---:|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | 0.847 | 0.807 | 0.899 | 0.758 | 0.882 | 0.708 |
| 00_Rock2-85-F_solo | 0.862 | 0.659 | 0.919 | 0.709 | 0.807 | 0.613 |
| 01_Funk1-97-C_solo | 0.817 | 0.694 | 0.837 | 0.686 | 0.691 | 0.634 |
| 02_Jazz2-110-Bb_solo | 0.874 | 0.835 | 0.824 | 0.234 | 0.729 | 0.706 |
| 02_SS1-68-E_solo | 0.675 | 0.641 | 0.747 | 0.485 | 0.650 | 0.467 |
| 03_Jazz1-200-B_solo | 0.814 | 0.767 | 0.857 | 0.678 | 0.766 | 0.500 |
| 04_Funk2-119-G_solo | 0.844 | 0.791 | 0.896 | 0.755 | 0.874 | 0.492 |
| 04_Rock3-148-C_solo | 0.800 | 0.777 | 0.830 | 0.612 | 0.699 | 0.612 |
| 05_BN2-166-Ab_solo | 0.912 | 0.770 | 0.884 | 0.425 | 0.857 | 0.560 |
| 05_SS2-107-Ab_solo | 0.918 | 0.824 | 0.908 | 0.500 | 0.763 | 0.627 |
| **mean** | **0.836** | **0.757** | **0.860** | **0.598** | **0.772** | **0.592** |

### 観察（実測に基づく）

- **basic-pitch が最も歪みに強い**（highgain でも F1 0.757、劣化 −0.107 で最小）。
  劣化モードは Recall 低下（0.880 → 0.742）= 取りこぼし型で、Precision は 0.781 を維持。
- **YourMT3+ は crunch では首位（0.860）だが highgain で崩壊**（−0.324、10トラック中
  4トラックが F1 ≤ 0.50、02_Jazz2 は 0.234 で est 23 vs ref 54 の大量取りこぼし）。
  クリーン dev の優位はクリーン GuitarSet が学習データに含まれる効果を含むため、
  ドメインが離れるほど失われる、という一貫した傾向（凍結 GT 参考測定とも整合）。
- **MuScriptor は distorted_electric_guitar プロンプトでも highgain で過検出型に劣化**
  （P 0.508 / R 0.727、est が ref の 1.5〜2倍のトラックが多数）。Recall は3エンジン中
  最高水準を維持しており、閾値・生成パラメータ調整での改善余地はあるが、無調整では
  basic-pitch に及ばない。凍結 GT 参考測定（過検出が主因で F1 0.157）と同型のエラーモード。
- M3 エンジン選定への示唆: 「クリーンで強いエンジンほど歪みで強い」は**成立しない**。
  歪みドメインの改善は本ベンチ（20クリップ・GT 計836ノート×2）で回して、
  凍結 GT 17ノートは最終確認に回すのが妥当。
- 注意: 本ベンチは合成歪みで、basic-pitch の学習ドメイン（アコギ由来のスペクトル包絡が
  残る）に近い可能性がある。実アンプ録音での順位逆転はあり得るため、実歪みデータでの
  追試を M3 内で計画すること（IDMT-SMT-Guitar 等、Zenodo 直接DL可: record 7544110）。

---

## 2026-07-18: MuScriptor 生成パラメータスイープ（M3・合成歪みベンチ）

前節で MuScriptor small の highgain 劣化が過検出型（P 0.508 / R 0.727）だったことを受け、
`TranscriptionModel.transcribe()` の**生成パラメータのみ**をスイープした。
**注意: 本スイープは合成歪みベンチ（highgain 10クリップ）でのチューニングであり、
実アンプ録音への一般化は保証されない**（特にプロンプトの知見は後述の理由で合成ベンチ依存の可能性が高い）。

### 条件

| 項目 | 値 |
|---|---|
| 環境 | `.venv-muscriptor`（muscriptor 0.2.1）、small、device=mps、batch_size=4 |
| 主戦場 | `eval_data/distorted_synth/highgain` 10クリップ（読み取りのみ、GT は同梱 JAMS） |
| 評価 | `guitartab.eval.metrics`（mir_eval、onset tol 50ms、offset 不評価）。前節と同一 |
| 探索対象 | `transcribe()` の公開生成パラメータ全部: `use_sampling`/`temperature`、`cfg_coef`（classifier-free guidance）、`beam_size`、`instruments`（プロンプト） |
| 探索不能だった候補 | `top_k`/`top_p`（`transcribe()` 内部で 0 に固定）、チャンク重複（5秒固定チャンク・重複なしで変更 API なし）、`max_gen_len`（2000 固定）、`beam_length_score_alpha`（`generate()` にはあるが `transcribe()` から非公開） |
| 再現性 | greedy（デフォルト）は決定論的。ベースライン再現試行は前節の実測（P 0.508 / R 0.727 / F1 0.592）と完全一致 |

### 全試行（highgain 10クリップ mean、各行1回実測、計21回 + 打ち切り1回）

デフォルト: instruments=["distorted_electric_guitar"]、greedy（use_sampling=False）、cfg_coef=1.0、beam_size=1。
表は変更点のみ表記。ac=acoustic_guitar、dist=distorted_electric_guitar、clean=clean_electric_guitar。

| # | 構成（デフォルトからの変更） | P | R | F1 | est計（ref計836） |
|---|---|---:|---:|---:|---:|
| 1 | （デフォルト = ベースライン再現） | 0.508 | 0.727 | 0.592 | 1266 |
| 2 | cfg 1.5 | 0.495 | 0.625 | 0.541 | 1046 |
| 3 | cfg 2.0 | 0.401 | 0.519 | 0.372 | 6292 |
| — | cfg 3.0 | 実行打ち切り（cfg 2.0 で縮退・悪化傾向が明確なため） | | | |
| 4 | beam 2 | 0.482 | 0.776 | 0.585 | 1396 |
| 5 | sampling T=0.7（seed 0） | 0.460 | 0.733 | 0.553 | 1433 |
| 6 | prompt clean | 0.485 | 0.711 | 0.566 | 1286 |
| 7 | prompt ac | 0.537 | 0.735 | 0.610 | 1162 |
| 8 | cfg 0.75 | 0.346 | 0.733 | 0.438 | 3398 |
| 9 | prompt なし（無条件） | 0.147 | 0.676 | 0.219 | 14061 |
| 10 | prompt ac+dist | 0.515 | 0.756 | 0.605 | 1259 |
| 11 | prompt ac + cfg 1.25 | 0.541 | 0.759 | 0.620 | 1181 |
| 12 | prompt ac + cfg 1.5 | 0.570 | 0.742 | 0.636 | 1134 |
| 13 | prompt ac + cfg 2.0 | 0.461 | 0.593 | 0.489 | 2876 |
| 14 | prompt ac + cfg 1.75 | 0.451 | 0.679 | 0.500 | 4089 |
| 15 | **prompt ac+dist + cfg 1.5（ベスト）** | **0.604** | **0.802** | **0.685** | 1125 |
| 16 | prompt ac+dist + cfg 1.75 | 0.505 | 0.794 | 0.569 | 3751 |
| 17 | prompt ac+dist + cfg 1.25 | 0.552 | 0.777 | 0.641 | 1178 |
| 18 | prompt ac+dist + cfg 1.6 | 0.545 | 0.797 | 0.625 | 2174 |
| 19 | prompt ac+dist + cfg 1.4 | 0.579 | 0.789 | 0.662 | 1168 |
| 20 | prompt ac+clean+dist + cfg 1.5 | 0.581 | 0.791 | 0.664 | 1137 |
| 21 | prompt ac+dist + cfg 1.5 + beam 2 | 0.586 | 0.797 | 0.670 | 1184 |

単因子の傾向（実測に基づく）:

- **プロンプトが最大の単因子**。ac（0.610）> ac+dist（0.605）> dist（0.592）> clean（0.566）>> なし（0.219、他楽器も転写して est 14061 に爆発）。
- **cfg_coef は dist 単独プロンプトでは単調に有害**（1.5→0.541、2.0→0.372）だが、
  **ac を含むプロンプトと組むと 1.25〜1.5 で改善**（ac+dist×1.5 = 0.685）。
  ただし **1.6 以上で縮退が発生**（特定トラックで est が 2〜4千ノートに暴走。
  cfg 2.0 dist は 02_SS1/03_Jazz1/05_BN2 で est 1800〜1953）。動作点 1.5 は崖の直前にある。
- **beam_size は F1 を改善しない**（beam2 単独 0.585、ベスト構成+beam2 0.670）。
  R は上がるが P が下がり、実行時間は約4倍（beam2+cfg で 10クリップ 22.7分 vs greedy 65秒）。
- **sampling は greedy に劣る**（T0.7 で 0.553）。
- cfg < 1（0.75）は縮退気味で悪化（0.438）。

### ベスト構成の3セット確認（各1回実測）

ベスト構成: `instruments=["acoustic_guitar", "distorted_electric_guitar"]`, `cfg_coef=1.5`, greedy, beam 1。
highgain での mean RTF 0.166（MPS。CFG で計算が約2倍になるがなお実用域）。

| セット | 調整前（前節） | 調整後 P | R | F1 | Δ | basic-pitch（M1 tuned） |
|---|---:|---:|---:|---:|---:|---:|
| highgain | 0.592 | 0.604 | 0.802 | **0.685** | +0.093 | **0.757** |
| crunch | 0.772 | 0.799 | 0.862 | **0.826** | +0.054 | **0.836** |
| クリーン dev | 0.868 | 0.881 | 0.878 | **0.879** | +0.011 | 0.864 |

クリーンでのプロンプト差分分離（各1回実測、いずれも cfg 1.5）:

| クリーン dev 構成 | P | R | F1 |
|---|---:|---:|---:|
| prompt ac+dist + cfg 1.5（ベスト構成そのまま） | 0.881 | 0.878 | 0.879 |
| prompt ac のみ + cfg 1.5 | 0.872 | 0.847 | 0.857 |
| （参考）prompt ac のみ + cfg 1.0 = 前節ベースライン | 0.853 | 0.886 | 0.868 |

→ クリーンでは cfg 1.5 単独は微減（0.868→0.857）だが、ac+dist プロンプトとの組合せで
+0.011 の純増（0.879）。**ベスト構成はクリーン性能を犠牲にしない**（この dev セット上では）。

### ベスト構成トラック別（highgain）

| track | P | R | F1 | ref | est |
|---|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | 0.663 | 0.873 | 0.754 | 79 | 104 |
| 00_Rock2-85-F_solo | 0.582 | 0.752 | 0.656 | 137 | 177 |
| 01_Funk1-97-C_solo | 0.690 | 0.750 | 0.719 | 104 | 113 |
| 02_Jazz2-110-Bb_solo | 0.689 | 0.778 | 0.730 | 54 | 61 |
| 02_SS1-68-E_solo | 0.439 | 0.714 | 0.543 | 70 | 114 |
| 03_Jazz1-200-B_solo | 0.500 | 0.860 | 0.632 | 43 | 74 |
| 04_Funk2-119-G_solo | 0.513 | 0.815 | 0.629 | 124 | 197 |
| 04_Rock3-148-C_solo | 0.567 | 0.831 | 0.674 | 71 | 104 |
| 05_BN2-166-Ab_solo | 0.699 | 0.866 | 0.773 | 67 | 83 |
| 05_SS2-107-Ab_solo | 0.694 | 0.782 | 0.735 | 87 | 98 |
| **mean** | **0.604** | **0.802** | **0.685** | | |

### FP の性質（ベスト構成 highgain、追加分析）

- 残存 FP 460 個の音価は中央値 220ms（80ms 未満はわずか 4%）。
  **短音スプリアスではないため、最小音価の後処理フィルタでは精度を回復できない**。
- ギター音域外（MIDI < 40 = E2 未満）の FP は 72 個（16%）。音域フィルタの上限効果は P +0.03 程度。
- 残りの FP は実在音価帯の誤ピッチ/誤オンセットで、生成・後処理どちらでも切り分け困難。

### 判定と観察（実測に基づく）

1. **highgain で basic-pitch 0.757 に届かない**（調整後ベスト 0.685、差 −0.072）。
   crunch でも 0.826 vs 0.836 で僅差の 2 位。生成パラメータの改善余地は実在した（+0.093）が、
   逆転には不足。
2. **M3 エンジン戦略の推奨: 歪みは basic-pitch（M1 tuned 構成）を主エンジンに維持**。
   MuScriptor はクリーンで首位（0.879 vs 0.864、しかもクリーン dev に対する
   パラメータ探索はしていない）のため、**クリーン/歪みでのエンジン使い分けが現状の最適**。
   アンサンブルは、highgain で R 0.802（bp は 0.742）と相補的な誤りモードを持つため
   理論上の伸び代はあるが、MuScriptor に confidence 出力がなく単純統合では P が悪化する。
   優先度は実歪みデータでの追試より下。
3. **プロンプト知見の合成ベンチ依存性（重要）**: ベストプロンプトに acoustic_guitar が
   入るのは、本ベンチのソースがアコギ mic 録音（スペクトル包絡が残存）だからである
   可能性が高い。実アンプ録音では ac+dist の優位は保証されない。実歪みデータ
   （IDMT-SMT-Guitar 等）での追試までは、この構成を「合成ベンチ最適」としてのみ扱うこと。
4. cfg 1.5 は縮退の崖（1.6+で暴走）の直前の動作点であり、未知の入力での安定性リスクがある。
   採用する場合は est ノート数の暴走検知（例: 音声長×10 ノート超で警告）を入れるべき。

---

## 2026-07-18: エンジン統合後の再現確認（MuScriptor 正式統合・M3）

`--engine muscriptor` 統合（`transcribe/muscriptor.py` + `_muscriptor_runner.py`、デフォルト = ベスト構成 ac+dist / cfg 1.5 / batch 4 / MPS / greedy、暴走検知 30 notes/sec）後、`uv run python -m guitartab eval --engine muscriptor`（dev 10トラック・1回実測）で mean P 0.882 / R 0.878 / F1 **0.879** — 前節スイープのベスト構成（0.881/0.878/0.879）を再現（P の +0.001 は丸め内）。

---

## 2026-07-18: M4a リズム量子化 — 一定テンポの拍推定+量子化（ゲート判定: 不合格）

実装: `src/guitartab/rhythm/`（estimate = librosa 候補 A、quantize = 最近傍スナップ、
rhythm.json）、`eval/rhythm_metrics.py` + `eval/rhythm_benchmark.py`（`eval --rhythm`）。
合成リズムベンチ `eval_data/rhythm_synth/`（dev 10 トラック × 5 変種、Karplus-Strong 合成、
生成スクリプト・manifest 同梱）。評価はすべて「GT ノート onset + 音声」入力の
量子化ステージ単独評価（転写誤差を混ぜない。設計 §4.3-4）。

### ベースライン（librosa.beat.beat_track 素の出力、dev 10、§4.4 の正式再実測）

| 指標 | 値 |
|---|---|
| TempoAcc1 | 4/10 |
| TempoAcc2 | 7/10 |
| Beat F-measure mean（DP 拍そのまま） | 0.433 |

### 本実装（候補 A: librosa T+AC ピーク候補 → 格子適合+DP アクセント+アンカー選択 → 回帰ポリッシュ）dev 10

| 指標 | 本実装 | ベースライン比 |
|---|---|---|
| TempoAcc1 | 5/10 | +1 |
| TempoAcc2 | 9/10 | +2 |
| Beat F-measure mean | 0.548 | +0.115 |
| Beat CMLt / AMLt mean | 0.389 / 0.700 | — |
| 量子化変位 中央値（トラック中央値の範囲） | 12〜24ms | 参考統計 |

トラック別の主な誤り: 半テンポ選択（Rock2→42.5、Jazz2→55、BN2→83）、
2 倍（SS1 68→136.5）、族外（Jazz1 200→115.7 のみ A2 失敗）。
テンポ正解でも拍ラベル（位相の 16 分単位シフト）失敗で beatF=0 が 1 件（Rock3）。

### 合成リズムベンチ（GPA = per-note tick 一致率、10 クリップ mean）

| 変種 | TempoAcc1 | TempoAcc2 | GPA | ゲート基準 | 判定 |
|---|---:|---:|---:|---|---|
| clean（一定テンポ） | 6/10 | 10/10 | **0.403** | GPA ≥ 0.95 | **不合格** |
| slow08（0.8×） | 8/10 | 10/10 | 0.502 | — | — |
| fast125（1.25×） | 5/10 | 9/10 | 0.202 | — | — |
| jitter σ=10ms | 7/10 | 9/10 | 0.397 | — | — |
| jitter σ=20ms | 6/10 | 9/10 | **0.286** | GPA ≥ 0.85 | **不合格** |

clean の GPA 損失の分解（診断）:
- 半テンポ選択が 4/10 クリップ（GPA ≈ 0）— 8 分主体のクリップでは半テンポ格子も
  完全適合し、onset 列だけではテンポレベルが原理的に決まらない（設計 §1.3 の
  「オクターブ曖昧性はオンセット列に内在する」の再確認）。
- テンポ正解・格子正解だが拍ラベル（定数 tick シフト）誤りが 2/10
  （Rock2: +42 tick シフト除去後 GPA 0.927、Jazz1: −9 tick 除去後 0.698）。
- 完全一致（GPA=1.0）は 4/10。

E2E モード（basic-pitch 転写経由、clean）: GPA 0.402 — 単独評価 0.403 と同等。
**転写誤差による量子化の劣化はほぼゼロ**（設計リスク 7 は本ベンチでは顕在化せず）。

### Holdout ゲート判定（8 トラック、1 回のみ実行）

| 指標 | 実測 | ゲート基準 | 判定 |
|---|---:|---|---|
| TempoAcc2 | 7/8 | 8/8 | **不合格**（05_SS3: 92.0 vs GT 98、6.1% ずれで族外） |
| TempoAcc1 | 4/8 | ≥ 6/8 | **不合格**（Jazz3→1/2、Rock1→約2倍、BN1-147→1/2） |
| Beat F-measure mean | 0.469 | ≥ 0.80 | **不合格**（テンポ正解 4 トラック中 03_BN1 は拍ラベル誤りで 0.000） |

**M4a ゲート: 不合格。** dev でのチューニング（候補生成・選択規則・位相決定の
構成要素別実測を反復）では dev Acc2 9/10・beatF 0.55 が上限で、holdout でも同傾向。

### 誤り構造の分析（M4b への申し送り）

1. **テンポオクターブは onset 列から原理的に決まらないケースが多い**: 8 分主体の
   演奏では半テンポ格子も完全適合し、チャンス補正付き適合度は疎な格子を系統的に
   優遇する（逆に補正を弱めると 2 倍テンポが常勝）。単一スカラー補正では
   dev 内でも両立不能なトラック対が存在することを実測で確認
   （例: Funk2 は強い補正が必要、BN2 は弱い補正が必要）。
2. **拍位相（拍ラベル）は 16 分格子の 1/4 拍シフト不変性のため、ノートからは
   mod P/4 でしか決まらない**。拍ラベルはアクセント情報が必要だが、
   シンコペーション（ボサノバ・ロックの先行アタック）ではオンセット包絡・
   librosa DP 拍・ノート出現率のすべてが裏拍側を指すトラックがある
   （GT テンポ固定でも位相正解は最良基準で 8/10 が上限だった）。
3. 帰結: **候補 B（Beat This!、学習済みビートトラッカー）の実測比較が M4b の
   最優先事項**。アクセント・拍レベルの判断は学習モデルに寄せ、本実装の
   格子適合+回帰ポリッシュは「tick 精度の微調整」に役割を限定するのが有望。
   量子化スナップ・rhythm.json・メトリクス・ベンチはそのまま流用できる。
4. 収穫: 回帰ポリッシュによりテンポ正解時の精度は十分（beatF 0.97〜0.99、
   合成 GPA 1.0）。E2E でも劣化しない。失敗は「レベルとラベルの選択」に集中している。

再現コマンド:

```
uv run python -m guitartab eval --rhythm --eval-data eval_data/guitarset
uv run python -m guitartab eval --rhythm --eval-data eval_data/rhythm_synth/clean
uv run python -m guitartab eval --rhythm --eval-data eval_data/rhythm_synth/clean --engine basicpitch  # E2E
python eval_data/rhythm_synth/make_rhythm_synth.py  # ベンチ再生成
```

---

## 2026-07-18: M4b 入口 — Beat This! 比較（候補 B、dev のみ・holdout 温存）

M4a 申し送り 3（「候補 B の実測比較が最優先」）の実施。Beat This!（CPJKU、ISMIR 2024、
MIT）を TempoEstimator として統合し、M4a の librosa 候補 A と同一ベンチ・同一メトリクスで
比較した。**判定: dev で librosa 版を明確に上回らなかったため holdout は未実施（温存）**。

### セットアップ（実測）

| 項目 | 値 |
|---|---|
| venv | 専用 `.venv-beatthis`（uv、CPython 3.11.3）。beat-this 1.1.0 + torch 2.13.0 + mir_eval + soundfile |
| 導入時の注意 | (1) `beat_this.model.pl_module` が mir_eval を hard import するため mir_eval が必須。(2) torchaudio 2.11 の torchcodec 必須化で `torchaudio.load` が落ちるが、beat_this は soundfile フォールバックを内蔵しており **soundfile の追加で解消**（torchcodec 不要） |
| チェックポイント | final0（77.3MB）。JKU クラウド（cloud.cp.jku.at、直接 HTTPS）から torch.hub キャッシュへ。curl 実測 82 秒。Google Drive 経由なし |
| 統合 | `src/guitartab/rhythm/beatthis.py` + `_beatthis_runner.py`（他エンジンと同じ別 venv サブプロセス方式。venv 解決: 引数 > GUITARTAB_BEATTHIS_PYTHON > `.venv-beatthis/bin/python`）。`eval --rhythm --rhythm-estimator beatthis` で選択 |
| デバイス・速度 | CPU / MPS とも動作。30 秒クリップのサブプロセス合計（モデルロード込み）: **CPU 1.8s / MPS 1.9s**（プロセス内推論のみは CPU 1.3s / MPS 3.4s）。86 秒実曲ミックスで 2.2s。**MPS に速度メリットなし → CPU をデフォルト** |

### 統合方式（拍列 → 一定テンポ + 位相）

素の最小二乗フィットは不成立を実測: Beat This! の生の拍列には冒頭スプリアス・欠落があり
（Rock2: raw 拍列の beatF 0.757 → 素 LS フィット後 0.372。テンポ 0.24% 誤差の 30 秒
ドリフトで ±70ms を割る）。採用構成は M4a 申し送りどおり役割分担:
**テンポレベル = Beat This! 拍列の median IOI** / **精密テンポ + 位相 mod P/4 =
M4a の格子適合走査+回帰ポリッシュ（レベル固定 ±8%）** / **拍ラベル k∈{0..3} =
Beat This! 拍列の circular mean 位相のスナップ**（librosa DP 拍の差し替え）。
他に 4 変種（top-k ポリッシュ再評価 / 拍列ロバスト LS / ロバスト LS+ポリッシュ /
circular mean 位相直接使用）を dev で実測したが、いずれも採用構成を上回らなかった
（beatF 0.403〜0.507）。ヒューリスティック積層を避けるためここで打ち切り。

### dev 10 トラック比較（GT ノート onset + ギターステム音声入力、各 1 回実測）

| 指標 | librosa 候補 A（M4a 再掲） | Beat This!（採用構成） |
|---|---:|---:|
| TempoAcc1 | 5/10 | **6/10** |
| TempoAcc2 | 9/10 | **10/10** |
| Beat F-measure mean | **0.548** | 0.527 |
| Beat CMLt / AMLt mean | 0.389 / 0.700 | 0.385 / 0.669 |

Beat This! のトラック別（refBPM → estBPM、A1A2、beatF）:

| track | est | A1 | A2 | beatF |
|---|---:|:-:|:-:|---:|
| 00_BN3-119-G | 59.5 | x | o | 0.000 |
| 00_Rock2-85-F | 85.0 | o | o | 0.982 |
| 01_Funk1-97-C | 97.1 | o | o | 0.987 |
| 02_Jazz2-110-Bb | 54.9 | x | o | 0.659 |
| 02_SS1-68-E | 66.7 | o | o | 0.048 |
| 03_Jazz1-200-B | 96.4 | x | o | 0.250 |
| 04_Funk2-119-G | 119.0 | o | o | 0.991 |
| 04_Rock3-148-C | 145.8 | o | o | 0.442 |
| 05_BN2-166-Ab | 83.0 | x | o | 0.658 |
| 05_SS2-107-Ab | 105.4 | o | o | 0.252 |

観察（実測に基づく）:

- **レベル正解トラックでは beatF 0.98〜0.99**（Rock2/Funk1/Funk2）と librosa 版の
  同種ケースと同水準以上。テンポ族は 10/10 で正しい（librosa 版は SS1 で 2 倍・
  Jazz1 で族外の誤りがあった）。
- 誤りは (a) ボサノバ・ジャズ系の**半テンポ選択** 4/10（BN3/Jazz2/Jazz1/BN2。
  Acc2 は正解 = メトリカルレベルの取り違えで、フルミックス学習のドメイン外である
  ギター単体入力での既知リスクどおり）、(b) SS 系（フィンガースタイル）で
  **拍追跡自体が崩壊**（SS2: raw beatF 0.075）、(c) BN3 は半テンポかつ裏拍位相で
  beatF 0.000。
- Rock3（145.8 vs GT 148）はレベル正解だがノート格子適合の精密化が 1.5% ずれる
  （演奏の系統的逸脱。librosa 版と同根の限界で、Beat This! 起因ではない）。

### 合成リズムベンチ GPA 比較（10 クリップ mean、単独評価モード）

| 変種 | librosa A1 / A2 / GPA（M4a 再掲） | Beat This! A1 / A2 / GPA |
|---|---|---|
| clean | 6/10 / 10/10 / **0.403** | 5/10 / 5/10 / 0.304 |
| slow08 | 8/10 / 10/10 / **0.502** | 4/10 / 6/10 / 0.202 |
| fast125 | 5/10 / 9/10 / 0.202 | 4/10 / 8/10 / 0.206 |
| jitter10 | 7/10 / 9/10 / **0.397** | 5/10 / 5/10 / 0.101 |
| jitter20 | 6/10 / 9/10 / **0.286** | 7/10 / 7/10 / 0.187 |

- **合成ベンチでは Beat This! が明確に劣る**。Karplus-Strong 合成音はさらに
  ドメイン外で、テンポレベルが**非オクターブ関係**（clean SS1: 90.7 vs GT 68、
  Funk2: 158.7 vs 119 = 4/3 等）に飛び、Acc2 まで崩れる（実データでは 10/10）。
  レベル固定 ±8% の統合方式は、レベル自体が誤ると回復手段がない。
- テンポ・位相が完全でも拍ラベルの定数 tick シフトで GPA が 0 になるケースは
  librosa 版と同じ（clean Rock2: est 85.0、変位 0 だが GPA 0.000）。

### 実曲スモーク（参考、GT なし・各 1 回実測）

B'z「ギリギリchop」（86.2 秒、work/wr7xTGTG-Mo）で入力ソース比較:

| 入力 | 拍数 / downbeat 数 | median IOI | IOI 標準偏差 |
|---|---|---|---|
| 原曲ミックス source.wav | 327 / 93 | 0.260s（≒231 BPM レベル） | **0.032s（一貫した追跡）** |
| ギターステム stems/guitar.wav | 147 / 66 | 0.400s | 0.659s（崩壊気味） |

フルミックスでは安定に追跡し、ギターステムでは崩れる —— 学習ドメインどおりの挙動で、
**M4b の「ミックス入力 vs ステム入力」実測比較（人手拍ラベル付き）で Beat This! を
再評価する価値がある**ことを示す。

### 判定と M4b への推奨（実測に基づく）

1. **holdout 未実施（温存）**: dev で librosa 版を明確に上回らなかった
   （テンポ Acc1/Acc2 は +1/+1 だが beatF −0.021、合成 GPA は全変種で同等以下）。
   M4a ゲート基準の holdout 判定は行っていない。
2. **ギター単体音声への Beat This! 単独差し替えは不採用**。強み（レベル正解時の
   拍精度、テンポ族の正確さ）と弱み（ボサ/ジャズの半テンポ、SS 系の崩壊、合成音の
   非オクターブ誤り）が librosa 版と相補的だが、dev 実測では選択規則を立てる
   識別シグナルが見つからなかった（変種 4 種で確認）。
3. **M4b 本実装への推奨**: Beat This! の再評価は実曲（フルミックス入力 + 人手拍
   ラベル）で行うのが本命（上記スモークの挙動 + 設計 §2.2 のミックス比較計画）。
   GuitarSet 系ベンチのゲート突破には、テンポレベル・拍位相の曖昧性解消に
   ノート列とは独立な手がかり（例: ミックスのドラム帯域）が必要というのが
   M4a から通算した実測の示唆。

再現コマンド:

```
uv run python -m guitartab eval --rhythm --rhythm-estimator beatthis --eval-data eval_data/guitarset
uv run python -m guitartab eval --rhythm --rhythm-estimator beatthis --eval-data eval_data/rhythm_synth/clean
```

---

## 2026-07-18: M4b — ミックス経路 実曲検証（「リズムはミックス、音符はステム」）

前節の推奨（ミックス入力での Beat This! 再評価）の実施。パイプラインに
quantize のリズム推定入力を選ぶ `--rhythm-source {stem,mix}`（transcribe、
default = stem = 従来動作。mix は work/{id}/source.wav を使い、転写ノートは
ステム由来のまま）と、`transcribe` / `quantize` サブコマンドの
`--rhythm-estimator {librosa,beatthis}` を追加した。配線テスト 7 件追加
（pytest 計 173 件通過）。

### 実曲検証（B'z「ギリギリchop」86.2 秒、work/wr7xTGTG-Mo、参考値）

**人手拍ラベルのない 1 曲の参考検証であり、M4b ゲート判定ではない**（ゲートは
設計 §6 のとおり人手拍ラベル付き実曲 ≥2 曲で行う）。参照テンポは公式 TAB 譜
由来の既知情報 **117 BPM・4/4**（docs/RESEARCH_2026-07-17.md §1.1）。
notes.json は既存の basic-pitch（M1 構成）× guitar ステム転写の 210 ノートを
4 構成で共通使用。各構成 1 回実測。

| 構成 | 推定テンポ (rhythm.json) | vs 117 | 族判定（117 × {1/3,1/2,1,2,3} の ±4%） |
|---|---:|---:|---|
| librosa × stem | 108.0 | −7.7% | 族外 |
| librosa × mix | 108.0 | −7.7% | 族外 |
| beatthis × stem | 145.2 | +24.1% | 族外 |
| **beatthis × mix** | **242.8** | +107.5% | **2 倍族（234.0 の +3.7%）— 唯一の族正解** |

生の拍列（推定器の前段トラッカー出力、同一音声・各 1 回実測）:

| トラッカー × 入力 | 拍数 | median IOI | IOI std | IOI 換算テンポ |
|---|---:|---:|---:|---:|
| librosa beat_track × stem | 164 | 0.511s | 0.029s | 117.5 |
| librosa beat_track × mix | 159 | 0.511s | 0.020s | 117.5 |
| Beat This! × stem | 147 | 0.400s | **0.659s（崩壊）** | — |
| Beat This! × mix | 327 | 0.260s | 0.032s | 230.8（8 分レベル） |

観察（実測に基づく）:

1. **ミックス入力はどちらのトラッカーにも一貫した拍列を与える**（IOI std
   0.020〜0.032s）。Beat This! のステム崩壊（std 0.659s）は前節スモークの再確認。
2. **librosa の生 beat_track はステム・ミックスとも 117.5 BPM（参照 +0.4%）を
   当てているのに、M4a の候補選択層が両構成とも 108.0 に上書きして族外へ落とす**。
   両構成の最終値はビット同一（108.04694…）で、選択層のスコアがノート格子適合
   （4 構成共通の同一ノート列）に支配され、音声側の手がかり（librosa アンカー
   117.5）が負けていることを示す。歪みエレキ実曲では「選択層が生トラッカーより
   悪い」という新規の実測ファクト。
3. beatthis × mix は 8 分レベルの拍列（median IOI 0.260s）を median-IOI レベル
   決定がそのまま採用して 2 倍族（242.8 = 2 × 121.4）。レベル固定 ±8% の
   精密化では半分レベルに戻る余地がない。
4. 4 構成とも 117±4%（1 倍レベル）には入らなかった。

### mix 経路 rhythm.json での MusicXML 再生成（beatthis × mix）

| 出力 | 小節数 | テンポ表記 | divisions |
|---|---:|---:|---:|
| 従来の固定 120BPM 近似（work の output.musicxml） | 42 | 120 | 4 |
| beatthis × mix の rhythm.json | 85 | 242.8 | 12 |

2 倍族推定のため「倍テンポ表記」になり小節数も約 2 倍（42 → 85）。
参考の算術換算: 半分レベルの 121.4 BPM・4/4 なら 86.2 秒 ≒ 44 小節に相当し、
公式 TAB の曲想（117 BPM・4/4）に整合するのはそちらのレベル。

### M4b 本判定（人手拍ラベル付き実曲）に向けた示唆（実測に基づく）

- ミックス経路は拍列の一貫性で明確に優位。残る誤りは**テンポレベル（オクターブ）
  の選択に集約**された（librosa は選択層の退行、Beat This! は 8 分レベル追跡）。
- 本判定（人手拍ラベル ≥2 曲、Beat F ≥ 0.70）の比較群には、
  (a) ミックス入力時に M4a 候補選択層をバイパスして生 beat_track を使う構成、
  (b) Beat This! のテンポレベル折り畳み（例: 拍あたりノート密度による 2 倍族判定）
  を含めて実測すべき、というのが本検証の示唆。

再現コマンド:

```
uv run python -m guitartab quantize work/wr7xTGTG-Mo/notes.json \
  --rhythm-estimator beatthis --audio work/wr7xTGTG-Mo/source.wav --out <OUT>/rhythm.json
uv run python -m guitartab musicxml work/wr7xTGTG-Mo/tab.json \
  --rhythm <OUT>/rhythm.json --out <OUT>/output.musicxml
uv run python -m guitartab transcribe --url <URL> --rhythm-source mix --rhythm-estimator beatthis
```

---

## 2026-07-18: M4b — ミックス入力の音声トラッカー信頼モード（候補選択層バイパス）

前節の示唆 (a)（ミックス入力時に M4a 候補選択層をバイパスして生 beat_track を
使う構成）の実装と検証。**1 曲の参考検証であり M4b ゲート判定ではない**
（ゲートは人手拍ラベル付き実曲 ≥2 曲で行う。holdout 不使用・GT 不変更）。

### 実装

`LibrosaConstantTempoEstimator(trust_tracker=True)`（`rhythm/estimate.py`）:
テンポ = **生 beat_track 拍列の頑健 LS フィット**（median-IOI で拍インデックスを
丸めて回帰。トラッカーのテンポから 6% 超ずれたらトラッカーのテンポへ
フォールバック）、ノート格子適合は**位相 phi16 の決定のみ**（`_fine_fit` 1 回。
`_refine` = 候補族の走査・選択・polish は一切呼ばない）、拍ラベル k∈{0..3} は
従来どおりテンポ固定 DP 拍のスナップ。`BeatThisTempoEstimator` も
`trust_tracker` を受けるが、beatthis 主経路は構造上すでにトラッカー信頼
（レベル = median IOI、ノートはレベル固定 ±8% のみ）のため、拍が取れなかった
ときの librosa フォールバックへの伝播のみ。配線: `transcribe --rhythm-source mix`
で自動有効化（stem は従来どおり無効）/ `quantize --trust-tracker` /
`run_transcribe_pipeline(rhythm_estimator=None, rhythm_source="mix")` でも自動。
配線・バイパスのテスト 8 件追加（pytest 計 181 件通過）。

設計上の実測メモ: 当初は「レベル固定 ±8% の格子適合走査 + polish」で
精密化する案を実測したが、本実曲では格子適合が走査端まで滑って
**trust モードでも 108.0 に退行**（117.5 × 0.92 = 108.1 が走査端）。
テンポレベルどころか ±8% でもノート格子にテンポを触らせてはならない、が
本曲の実測。よって位相のみに限定した。

### stem 経路の回帰ゼロ証明（実測、各 1 回）

trust_tracker はデフォルト False で、stem 経路は従来コードパスをそのまま通る。
実装後の再実測は M4a/M4b の記録値と完全一致:

| ベンチ | 実装後 | 記録値（M4a/M4b） |
|---|---|---|
| GuitarSet dev 10（librosa × stem） | Acc1 5/10 / Acc2 9/10 / beatF 0.548 / CMLt 0.389 / AMLt 0.700 | 同一 |
| rhythm_synth clean（librosa） | Acc1 6/10 / Acc2 10/10 / GPA 0.403 | 同一 |
| ギリギリchop librosa × stem（trust なし） | 108.0 | 108.0 |
| ギリギリchop librosa × mix（trust なし） | 108.0 | 108.0 |

### 実曲検証（B'z「ギリギリchop」86.2 秒、参照 117 BPM・4/4、各 1 回実測）

| 構成 | 推定テンポ | vs 117 | 族判定 |
|---|---:|---:|---|
| **librosa × mix × trust** | **118.0** | **+0.9%** | **1 倍レベル正解（117.5±1 に入る）** |
| beatthis × mix × trust | 242.8 | +107.5% | 2 倍族のまま（前節と同値） |

- librosa × mix × trust の内訳: 生 beat_track テンポ 117.45 / 拍列 163 拍の
  LS フィット 118.02 → rhythm.json tempo_bpm 118.0。前節の 108.0（選択層）が
  解消し、生トラッカーの族に入った。
- beatthis × mix × trust は不変（242.8）: Beat This! は 8 分レベルの拍列
  （median IOI 0.260s）自体を返しており、trust モードは「トラッカーを信頼する」
  方向の変更のため 2 倍族はそのまま。レベル折り畳み（前節の示唆 (b)）は未実装。

### mix trust 経路 rhythm.json での MusicXML 再生成（librosa × mix × trust）

| 出力 | 小節数 | テンポ表記 | 拍子 |
|---|---:|---:|---|
| librosa × mix × trust | 41 | 118 | 4/4 |
| （参考）beatthis × mix、前節 | 85 | 242.8 | 4/4 |
| （参考）固定 120BPM 近似 | 42 | 120 | 4/4 |

41 小節 × 4 拍 / 118 BPM ≒ 83.4 秒 = 転写ノートの実スパンに一致
（86.2 秒の音声末尾は無音/ノートなし）。公式 TAB の曲想（117 BPM・4/4、
86.2 秒 ≒ 42〜44 小節）と整合するレベルに入った。

### 残課題（実測に基づく）

- beatthis のテンポレベル折り畳み（拍あたりノート密度等による 2 倍族判定、
  前節示唆 (b)）は未実装。
- 本検証は 1 曲参考。M4b 本判定は人手拍ラベル付き実曲 ≥2 曲・Beat F ≥ 0.70 で
  行うこと（trust モードの beatF はまだ未計測）。

再現コマンド:

```
uv run python -m guitartab quantize work/wr7xTGTG-Mo/notes.json \
  --trust-tracker --audio work/wr7xTGTG-Mo/source.wav --out <OUT>/rhythm.json
uv run python -m guitartab musicxml work/wr7xTGTG-Mo/tab.json \
  --rhythm <OUT>/rhythm.json --out <OUT>/output.musicxml
uv run python -m guitartab transcribe --url <URL> --rhythm-source mix  # trust 自動有効化
uv run python -m guitartab eval --rhythm --eval-data eval_data/guitarset  # stem 回帰確認
```
