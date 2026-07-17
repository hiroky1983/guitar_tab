# Guitar TAB Transcriber v2 設計書

作成: 2026-07-17（v1 postmortem と 2026年7月時点のツール調査に基づく）

## 目的

YouTube URL を入力に、ギターパートの TAB 譜を出力する Python CLI。

```
python -m guitartab transcribe --url <YouTube URL>
```

## v1 の失敗分析（postmortem）

v1 は Ralph Loop による自動反復改善で開発されたが、以下の理由で頓挫した。
詳細な実態は `docs/SYSTEM_LIMITATIONS.md`（v1 開発時の正直な限界報告）を参照。

| # | 失敗 | 原因 | v2 での対策 |
|---|------|------|-----------|
| 1 | 実精度 29.41% で頭打ち | Basic Pitch が歪みエレキに不適（G3・高フレット音を検出不能）。学習データの GuitarSet はアコギのみ | 転写エンジンを抽象化し、歪みエレキ対応モデルを含む複数エンジンをベンチで選定 |
| 2 | 「精度100%」の虚偽達成 | ground truth をシステム出力から再生成した（自己参照評価） | GT は人間製で**凍結**（`eval_data/gt/`）。GT の自動更新を禁止。mir_eval 標準メトリクスで評価 |
| 3 | 未検証ヒューリスティック約800行 | テストなしで「期待精度 +N%」を机上で積算 | 変更ごとに固定ベンチを**実測**。実測なしの精度主張を禁止 |
| 4 | リズム量子化の失敗 | ドラムビート推定に依存した量子化が不正確でコード内無効化 | 量子化は独立した後段ステージに分離。音符検出が固まるまで着手しない |
| 5 | 1曲への過剰適合 | B'z「ギリギリchop」（歪み・速弾き・パームミュート）のみで開発 | GuitarSet（公開・人間アノテーション付き）で易→難の段階ゲート |

## 設計原則

1. **評価ファースト** — パイプラインより先に評価ハーネスを作る。凍結 GT + `mir_eval` の note-level Precision/Recall/F1。
2. **段階ゲート** — クリーン単音 → クリーン和音 → 歪みエレキ、の順で攻略。前段の基準を実測でクリアするまで次へ進まない。
3. **エンジン差し替え可能** — audio→notes は Protocol で抽象化し、複数実装を同一ベンチで比較して選ぶ。
4. **ステージごとの中間成果物** — 各ステージは入出力をファイルに残す。単体実行・キャッシュ・デバッグ可能。

## パイプライン

```
1. download    yt-dlp                        → work/{id}/source.wav
2. separate    Demucs htdemucs_6s            → work/{id}/stems/guitar.wav
3. transcribe  <エンジン抽象化: 下記>          → work/{id}/notes.json
4. tab         運指割当（コスト最小化 DP）     → work/{id}/tab.json
5. render      ASCII tab（最優先）/ MIDI / MusicXML
6. (M4以降)    リズム量子化・小節割り
```

- `NoteEvent = {onset_sec, offset_sec, midi_pitch, velocity, confidence}` を notes.json の共通スキーマとする。
- separate は動作実績のある `~/myspace/sound_stems/stems.py` を移植する（v1 でもここまでは成功していた）。

## ツール選定（2026-07 調査結果）

### audio → notes（音符検出）

| 順位 | エンジン | 理由 | リスク |
|---|---|---|---|
| 1 | **MuScriptor**（Kyutai, 2026-07公開, `pip install muscriptor`） | **"distorted electric guitar" を明示サポートする唯一のモデル**。v1 失敗の直接対策。170k曲の実音源で学習 | 公開直後で実績薄。Apple Silicon 動作未確認。重みは CC BY-NC（非商用） |
| 2 | **basic-pitch**（ベースライン兼フォールバック） | 枯れて軽量、Apple Silicon 実績あり（Python 3.10 限定） | 歪み・低音弦に弱い（v1 で実証済み） |
| 3 | YourMT3+（予備） | PyTorch/MPS で Mac 動作の可能性 | pre-release 状態、セットアップが重い |

非採用: MT3（開発停止・JAX で重い）、QMUL ドメイン適応モデル（歪み非対応と論文明記）、TART（コード未公開）、trimplexx/music-transcription（重み配布なし）。

**M0 で MuScriptor と basic-pitch の2エンジンを同一ベンチにかけ、実測で採用を決める。**

### notes → tab（運指割当）

1. **自前 DP（コスト最小化）** — v1 の `transcriber.py` にある弦/フレット割当ロジック（開放弦ボーナス・ポジション移動ペナルティ）を叩き台に、[gtrsnipe](https://github.com/scottvr/gtrsnipe) のコスト関数（フレット幅・手移動・弦切替）を参考に再実装。
2. Open-Fret（Fretting-Transformer 非公式実装）は**不採用** — 学習済み重みなし・動作実績確認不可。

### 出力

- 第一目標: **ASCII tab**（テキストで diff 可能 = テストしやすい）
- 第二: MIDI（耳で検証できる）
- 第三: MusicXML（Guitar Pro / MuseScore 連携）。v1 の LilyPond 依存は廃止（重い・デバッグ困難）。

## リポジトリ構成（v2）

```
guitar_tab/
  pyproject.toml
  src/guitartab/
    cli.py               # transcribe / separate / eval サブコマンド
    pipeline.py          # ステージ実行とキャッシュ
    download.py          # v1 youtube.py を流用
    separate.py          # sound_stems/stems.py を移植
    transcribe/
      base.py            # NoteEvent, TranscriberEngine Protocol
      basicpitch.py
      muscriptor.py
    tab/
      fingering.py       # DP 運指割当
      render_ascii.py
      render_midi.py
    eval/
      metrics.py         # mir_eval ラッパー
      benchmark.py       # eval_data/ 一括評価・エンジン比較表出力
  tests/
  eval_data/
    gt/                  # 凍結 GT（v1 の ground_truth.json + GuitarSet 抜粋）
  docs/
    DESIGN.md            # 本書
    SYSTEM_LIMITATIONS.md  # v1 の限界報告（歴史的記録）
    AI_AGENT_GUIDE.md      # v1 の改善手順書（歴史的記録）
  _archive/              # v1 実験ファイル一式（git 管理外）
```

## 流用する既存資産

| 資産 | 行き先 |
|---|---|
| `~/myspace/sound_stems/stems.py`（Demucs 分離、動作実績あり） | `separate.py` |
| `guitartab_transcriber/youtube.py`（yt-dlp DL） | `download.py` |
| `guitartab_transcriber/transcriber.py` の運指コスト最小化 | `tab/fingering.py` の叩き台 |
| `guitartab_transcriber/tab_format.py` の ASCII 出力 | `render_ascii.py` の叩き台 |
| `eval_data/gt/ground_truth.json` + `image.png`（人間製 GT） | 最終ボス曲の評価用（凍結） |

v1 の倍音補正・パームミュート検出・グリッドスナップ等のヒューリスティック群は**持ち込まない**（未検証のまま積み上がったものなので、ベンチで必要性が実証されたものだけ個別に再導入する）。

## マイルストーン

- **M0: 骨組み + 評価ハーネス** — 新パッケージ構成、GuitarSet から評価用サンプル取得、MuScriptor / basic-pitch の2エンジンをベンチ実測して採用決定。
- **M1: クリーンギターで note F1 ≥ 0.80**（GuitarSet 基準）— ここをクリアするまで歪み曲に触らない。
- **M2: tab 割当 + ASCII 出力** — 運指 DP、演奏可能性の検証。
- **M3: 歪みエレキ対応** — Demucs 分離品質の検証、エンジンの歪み耐性実測、必要なら前処理。
- **M4: リズム量子化・小節割り** — 音符検出が安定してから着手。

## 開発ルール（v1 の教訓）

1. `eval_data/gt/` 配下の GT ファイルは**人間の確認なしに変更・追加・再生成しない**。
2. 精度は `benchmark.py` の実測値のみを正とする。「期待精度」「見込み」を報告・記録しない。
3. ヒューリスティック追加は、追加前後のベンチ実測差分をセットで記録する。
