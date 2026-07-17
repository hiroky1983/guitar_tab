# guitartab (v2)

YouTube URL からギターパートの TAB 譜を生成する Python CLI。
設計は `docs/DESIGN.md`、v1 の失敗分析と選定根拠は `docs/RESEARCH_2026-07-17.md` を参照。

```
python -m guitartab transcribe --url <YouTube URL>
```

パイプライン: download (yt-dlp) → separate (Demucs htdemucs_6s) → transcribe（エンジン差し替え可能）。
各ステージは `work/{id}/` に中間成果物（source.wav / stems/guitar.wav / notes.json）を残し、
既存ファイルがあればスキップします（`--force` で再実行）。

旧実装 v1 は `guitartab_transcriber/` + `main.py` に参照用として残っています（メンテ対象外・インストール対象外）。

## セットアップ

Python 3.11 / [uv](https://docs.astral.sh/uv/) 前提。

```bash
uv sync          # .venv 作成 + 本体依存 + dev（pytest）インストール
```

### 外部ツール

- **ffmpeg**（必須）: `brew install ffmpeg` — yt-dlp の WAV 変換に使用
- **demucs**（separate ステージで必要）: 本体依存には含めていません。使うときに
  ```bash
  uv pip install demucs torchaudio torchcodec
  ```
  torchaudio 2.11+ は `torchaudio.load()` に torchcodec（+ ffmpeg）が必須
  （2026-07-17 実測: demucs 4.1.0 / torch 2.13.0 / torchcodec 0.15.0 / ffmpeg 8.0.1 で動作確認）。

### basic-pitch の運用（重要）

basic-pitch は Apple Silicon では **Python 3.10 限定**のため、本体 venv（3.11）には
インストールしません（`uv sync` では入りません）。専用の 3.10 venv を作り、
guitartab はそこの python をサブプロセスとして呼び出します
（`src/guitartab/transcribe/basicpitch.py` + `_basicpitch_runner.py`）:

```bash
uv venv --python 3.10 .venv-basicpitch
uv pip install --python .venv-basicpitch/bin/python basic-pitch "numba<0.61" "llvmlite<0.44" "setuptools<81"
```

ピンの理由（2026-07-17 時点、外すと壊れる）:

- `numba<0.61` / `llvmlite<0.44` — 最新 llvmlite (0.48) は Python 3.10 向け wheel がなく
  ソースビルドに失敗する
- `setuptools<81` — resampy が `pkg_resources` を import しており、setuptools 81+ では
  削除済みのため実行時に落ちる

venv の場所はデフォルトでプロジェクト直下の `.venv-basicpitch/bin/python` を探します。
別の場所に置いた場合は次のどちらかで指定します:

```bash
export GUITARTAB_BASICPITCH_PYTHON=/path/to/venv/bin/python
# または
python -m guitartab transcribe --url <URL> --basicpitch-python /path/to/venv/bin/python
```

（Linux 等 3.11 で basic-pitch が動く環境なら extras `uv pip install -e '.[basicpitch]'` で
本体 venv に同居させ、`--basicpitch-python .venv/bin/python` を指定する運用も可能です）

### YourMT3+ の運用

YourMT3+ は GPL/Apache 混在ライセンスのため**コード・チェックポイントをリポジトリに
同梱しません**（`third_party/` は gitignore 済み）。basic-pitch と同じ別 venv
サブプロセス方式で動かします
（`src/guitartab/transcribe/yourmt3.py` + `_yourmt3_runner.py`。
検証記録は `docs/YOURMT3_VERIFICATION_2026-07-17.md`）:

1. コード+チェックポイントを `third_party/yourmt3/` に配置する。
   実体は HF Space `mimbres/YourMT3`（gated ではない）。必要なのは `amt/` ディレクトリと
   `amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`
   （YPTF.MoE+Multi noPS、538MB）。
2. 専用 venv を作る（Python 3.11、バージョン固定が必要）:

   ```bash
   uv venv --python 3.11 .venv-yourmt3
   uv pip install --python .venv-yourmt3/bin/python \
     torch torchaudio soundfile "transformers==4.45.1" "numpy==1.26.4" \
     pytorch-lightning einops librosa mido mir_eval
   ```

   （2026-07-17 動作確認構成: torch 2.13.0 / torchaudio 2.11.0 / lightning 2.6.5 /
   soundfile 0.14.0。transformers と numpy のピンを外すと壊れる）

場所を変える場合は環境変数 `GUITARTAB_YOURMT3_PYTHON` / `GUITARTAB_YOURMT3_HOME`、
または CLI の `--yourmt3-python` / `--yourmt3-home` で指定します。推論デバイスは
デフォルト CPU（M2 では MPS とほぼ同速のため安定側）。`GUITARTAB_YOURMT3_DEVICE=mps`
または `--yourmt3-device mps` で切替可能です。

## 使い方

```bash
# YouTube → notes.json（work/{video_id}/notes.json）
python -m guitartab transcribe --url <YouTube URL>

# ギターステム抽出のみ（work/{id}/stems/guitar.wav）
python -m guitartab separate --url <YouTube URL>
python -m guitartab separate --input path/to/audio.wav

# ベンチセット一括評価（エンジン比較表を出力）
python -m guitartab eval --eval-data eval_data --engine basicpitch
```

## 評価データ

- `eval_data/gt/` は**凍結 ground truth**（人間製）。変更・追加・再生成は禁止
  （v1 はここを自動再生成して評価を無効化した。`docs/DESIGN.md` 開発ルール参照）。
- ベンチアイテムの配置規約: `eval_data/items/<id>/` に音声（`audio.wav` 等）と
  GT（`gt.jams` / `gt.json` / `ground_truth.json`）のペアを置く。
  GuitarSet の JAMS（`note_midi` namespace）と v1 形式（time/string/fret/duration）に対応。
- 精度指標は `mir_eval` の note-level Precision/Recall/F1
  （onset tolerance 50ms、pitch は同一半音、offset 不問）。

## テスト

```bash
uv run pytest
```
