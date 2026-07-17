# distorted_synth — 合成歪みベンチマーク（M3 評価用）

作成: 2026-07-17。GuitarSet dev セット（`eval_data/guitarset/`、10トラック）の audio に
デジタル歪み処理を施した 2強度 × 10トラック = 20クリップ。

## 重要な限界

**これは合成歪みであり、実アンプ・実エフェクタの録音ではない。**
ソースはアコースティックギターの mic 録音（GuitarSet audio_mono-mic）であり、
エレキギターのピックアップ特性・アンプの動的挙動・キャビネット/マイクの実応答は含まない。
実歪みエレキ（例: B'z 凍結 GT）への一般化は保証されない。エンジンの相対比較・
劣化傾向の把握に使い、絶対性能の主張には使わないこと。

## GT（アノテーション）

`annotations/*.jams` は `eval_data/guitarset/annotations/` の**バイト同一コピー（無改変）**。
歪み処理は波形整形（メモリレス）・IIRフィルタ・コンプレッサのみで、時間伸縮・
ピッチシフト・ディレイ系を含まないため、ノートのタイミング・ピッチは変化しない
（IIR の群遅延はサブ ms オーダーで、評価の onset tolerance 50ms に対して無視できる）。
したがって元 GT をそのまま流用できる。**このファイル群も凍結扱い（変更・再生成禁止）。**

## 生成手順（再現方法）

- ツール: [pedalboard](https://github.com/spotify/pedalboard) 0.9.24（Spotify、C++ 実装）、
  soundfile 0.14.0、numpy 2.4.6、Python 3.11。
- スクリプト: 本ディレクトリの `make_distorted.py`（処理チェーンは下表と各
  `manifest.json` の `chain` フィールドにも完全記録。決定論的処理で再現可能）。
- 入出力: 44.1kHz mono PCM_16 → 同一フォーマット。サンプル数は入力と同一。
- 処理後にピークを -1.0 dBFS に正規化（トラックごとの正規化ゲインは manifest に記録）。

### crunch（中程度の歪み）

| # | プラグイン | パラメータ |
|---|---|---|
| 1 | HighpassFilter | cutoff 60 Hz |
| 2 | Gain（プリゲイン） | +14 dB |
| 3 | Distortion（tanh 波形整形） | drive 18 dB |
| 4 | PeakFilter（プレゼンス） | 2200 Hz, +3 dB, Q 0.9 |
| 5 | LowpassFilter（キャビネット風） | cutoff 5500 Hz |

### highgain（強い歪み + コンプレッション感）

| # | プラグイン | パラメータ |
|---|---|---|
| 1 | HighpassFilter | cutoff 60 Hz |
| 2 | Compressor | threshold −30 dB, ratio 4:1, attack 2 ms, release 120 ms |
| 3 | Gain（プリゲイン） | +26 dB |
| 4 | Distortion（tanh 波形整形） | drive 32 dB |
| 5 | Clipping（ハードクリップ） | threshold −3 dB |
| 6 | PeakFilter（ミッドスクープ） | 700 Hz, −4 dB, Q 0.7 |
| 7 | LowpassFilter（キャビネット風） | cutoff 4500 Hz |

歪み度の目安（00_BN3、実測）: クレストファクタ 8.7（原音）→ 2.4（crunch）→ 1.4（highgain）。

## 出典

- ソース音源/アノテーション: GuitarSet（Zenodo record 3371780,
  https://zenodo.org/record/3371780）の audio_mono-mic 変種。
  トラック選定は dev セット（`eval_data/guitarset/manifest.json`）と同一。
  **holdout（`eval_data/guitarset_holdout/`）は含まない**（holdout は閾値検証専用のため）。

## 使い方

```
uv run python -m guitartab eval --eval-data eval_data/distorted_synth/crunch   [--engine ...]
uv run python -m guitartab eval --eval-data eval_data/distorted_synth/highgain [--engine ...]
```

配置は GuitarSet 形式（annotations/ + audio/ の prefix 対応）で、benchmark.py が
そのまま発見できる。実測記録は docs/BENCHMARKS.md の「合成歪みベンチ」節を参照。
