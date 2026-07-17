# MuScriptor small 実重み実測結果（2026-07-17）

環境: `.venv-muscriptor`（muscriptor 0.2.1 / torch 2.13.0 / Python 3.11）、device=mps、batch_size=4。
重み: `MuScriptor/muscriptor-small`（ゲート承認済み HF_TOKEN で自動DL成功、HF キャッシュ 393MB、safetensors）。
評価: `guitartab.eval.metrics`（mir_eval、onset tolerance 50ms、pitch 同一半音、offset 不評価）。
生成パラメータはすべてデフォルト（use_sampling=False、beam_size=1。チューニングなし）。

## 実行速度（実重み・MPS）

| 測定 | 値 |
|---|---|
| モデルロード | 初回（DL込み）16.2s / キャッシュ後 1.2〜1.7s |
| GuitarSet dev 20ジョブ（計632s音声） | 計62.1s、mean RTF **0.103**（min 0.056 / max 0.164） |
| 歪みステム 86.2s | 14.9s（RTF 0.173） |
| 原曲ミックス 86.2s | 34.9s（RTF 0.405、ノート密度が高いほど遅い） |

検証時ワーストケース（ランダム重み・EOSなし・batch 2）は RTF 1.73。実重みでは EOS 早期終了により
クリーン音源で約 **17倍**、密なミックスでも約 **4倍** 高速化。dangling note（終端イベント欠落）は全ジョブで 0。

## dev 10トラック（GuitarSet solo、JAMS GT、tol 50ms）

### instruments=["acoustic_guitar"]（採用候補）

| track | P | R | F1 | ref | est |
|---|---:|---:|---:|---:|---:|
| 00_BN3-119-G_solo | 0.924 | 0.924 | 0.924 | 79 | 79 |
| 00_Rock2-85-F_solo | 0.886 | 0.905 | 0.895 | 137 | 140 |
| 01_Funk1-97-C_solo | 0.746 | 0.846 | 0.793 | 104 | 118 |
| 02_Jazz2-110-Bb_solo | 0.980 | 0.907 | 0.942 | 54 | 50 |
| 02_SS1-68-E_solo | 0.761 | 0.771 | 0.766 | 70 | 71 |
| 03_Jazz1-200-B_solo | 0.816 | 0.930 | 0.870 | 43 | 49 |
| 04_Funk2-119-G_solo | 0.887 | 0.887 | 0.887 | 124 | 124 |
| 04_Rock3-148-C_solo | 0.759 | 0.845 | 0.800 | 71 | 79 |
| 05_BN2-166-Ab_solo | 0.897 | 0.910 | 0.904 | 67 | 68 |
| 05_SS2-107-Ab_solo | 0.871 | 0.931 | 0.900 | 87 | 93 |
| **mean (10)** | **0.853** | **0.886** | **0.868** | | |

### instruments=["clean_electric_guitar"]（試行記録、不採用）

mean P=0.848 R=0.772 F1=**0.803**。全トラックで acoustic_guitar と同等以下
（音源がアコギ mic 録音なので妥当）。Recall が大きく低下（0.886→0.772）。
トラック別: 00_BN3 0.911 / 00_Rock2 0.852 / 01_Funk1 0.764 / 02_Jazz2 0.804 / 02_SS1 0.698 /
03_Jazz1 0.870 / 04_Funk2 0.873 / 04_Rock3 0.687 / 05_BN2 0.800 / 05_SS2 0.776。

### basic-pitch 比較（同一 dev・同一評価）

| 構成 | P | R | F1 |
|---|---:|---:|---:|
| basic-pitch デフォルト | 0.748 | 0.871 | 0.798 |
| basic-pitch tuned（24回スイープの最良） | 0.855 | 0.880 | 0.864 |
| **MuScriptor small acoustic_guitar（無チューニング）** | 0.853 | 0.886 | **0.868** |

MuScriptor はパラメータ探索なしで basic-pitch の tuned 構成をわずかに上回る（+0.004）。
過適合リスクなし（この dev セットに対する調整を一切していない）。
単体 F1<0.80 は 3トラック（01_Funk1 0.793、02_SS1 0.766、04_Rock3 0.800 丁度）。
basic-pitch tuned の worst 02_SS1（0.735）は MuScriptor で 0.766。

## 歪みエレキ参考測定（凍結 GT 17ノート、正式ベンチではない）

手順は BENCHMARKS.md の basic-pitch 参考測定と同一: GT 先頭ノート対応 + 10ms 刻みで
オフセット 20〜40s を総当たりし、ベスト整合のみ報告。GT・コードへの焼き込みなし。

| 入力 | instruments | est総数 | ベストoffset | 窓内 P | R | F1@50ms | F1@200ms | F1@50ms chroma |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stems/guitar.wav（Demucs分離） | distorted_electric_guitar | 613 | 22.77s | 0.118 | 0.235 | **0.157** | 0.235 | 0.157 |
| source.wav（原曲ミックス直接） | distorted_electric_guitar | 1318 | 26.33s | 0.072 | 0.294 | **0.116** | 0.116 | 0.116 |
| （参考）basic-pitch stem | — | 210 | 23.62s | 0.231 | 0.176 | **0.200** | 0.200 | 0.267 |

観察（実測に基づく）:
- ステム入力の窓内マッチは 4/17（tol 200ms で 6/17）。basic-pitch の参考 F1 0.200 を下回るが、
  Recall は 0.235 vs 0.176 で上回る。主因は過検出（86s で 613ノート、窓内 34 vs basic-pitch 13）。
- basic-pitch と異なり **高フレット音は出せている**: 全出力のピッチ上限 72（basic-pitch は窓内上限 64）、
  MIDI≥66 が 73ノート。ただし GT 窓内（22.5〜27.6s）では上昇速弾き（54→72）に対応する高音列は
  出ておらず、窓内はコード状の中音域が支配的。timing が細かく割れた出力になっている。
- 原曲ミックス直接はさらに悪化（1318ノート、他楽器の混入が支配的）。**Demucs 分離の省略は不可**。
- basic-pitch のベストオフセット 23.62s に MuScriptor ステム出力を当てると F1 0.048（200ms で 0.190）。
  オフセット同定自体が推定品質に依存する点に注意（参考値扱いの根拠）。

## 所見（M0 エンジン採用判断向け）

1. クリーンギター（M1 領域）: MuScriptor small は無調整で dev F1 0.868 と、basic-pitch の
   チューニング済み 0.864 と同等（わずかに上）。basic-pitch は dev でチューニングした値なので、
   汎化面では MuScriptor が有利な可能性がある（holdout 8トラックでの確認を推奨）。
2. 速度は実用域（RTF 約 0.1、モデルロード 1.2s）。venv 分離は basic-pitch 同様に必要。
3. 歪みエレキ（M3 領域）: 「高フレットを検出できない」という basic-pitch の構造的欠陥は
   MuScriptor には当てはまらない（ピッチ上限 72 まで出力）が、この曲の参考 F1 は 0.157 と
   basic-pitch 0.200 を下回った。過検出と timing の割れが主因で、生成パラメータ
   （use_sampling/beam_size/cfg_coef）やチャンク境界処理の調整余地は未探索。
4. ミックス直接入力による Demucs 省略は現状不可。
5. medium/large は未使用（small のみで dev 目標水準に到達したため。指示どおり large 禁止、
   medium は必要性が生じず未実行）。
6. ライセンス注意: 重みは CC BY-NC（非商用限定）。

## 生成物（スクラッチパッド）

- `run_muscriptor.py` — バッチ転写ランナー（.venv-muscriptor で実行）
- `notes/` — 全 notes.json（dev 20 + smoke 1 + 歪み 2）
- `dev_results.json` / `dist_results.json` / `dev_timing.jsonl` / `dist_timing.jsonl`
