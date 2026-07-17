# MuScriptor Apple Silicon 実機検証結果（2026-07-17）

M0 の実機検証タスクの結果記録。検証環境: M2 / 24GB / macOS。

## 結論

**MPS で動作する。条件付きで v2 エンジンとして採用可。**
ただし公式重みはゲート付き HF リポジトリにあり、**ユーザーの HuggingFace アカウントでのライセンス同意 + HF_TOKEN 設定が必要**（未完了。下記ブロッカー参照）。

## 検証結果サマリ

| 項目 | 結果 |
|---|---|
| インストール | 成功。muscriptor 0.2.1 + torch 2.13.0、Python 3.11（公式要件 3.10–3.12）。venv: `.venv-muscriptor` |
| MPS 動作 | end-to-end 動作確認済み。ただし `device="mps"` の**明示指定が必須**（デフォルトは CUDA→CPU） |
| 推論速度 | MPS + batch_size=2 で 17.3s / 10秒音声（ワーストケース）。batch=1 だと CPU より遅い。**batch_size>=4 を明示すべき** |
| 重み | `hf://MuScriptor/muscriptor-small` は**ゲート付き**（CC BY-NC 同意必要）。匿名 DL は 401。GitHub Releases にも重みなし |
| 品質 | **未検証**。ランダム初期化の同一アーキテクチャ（102M）で演算互換性のみ実証。転写品質は重み入手後に GuitarSet ベンチで実測すること |

## ブロッカー（ユーザー操作が必要）

1. https://huggingface.co/MuScriptor/muscriptor-small のゲートを承認（連絡先共有 + CC BY-NC 同意）
2. `HF_TOKEN` を環境に設定

## 実装メモ（transcribe/muscriptor.py 向け）

```python
from muscriptor import TranscriptionModel
from muscriptor.events import NoteStartEvent, NoteEndEvent

model = TranscriptionModel.load_model("small", device="mps")  # 要 HF_TOKEN
for ev in model.transcribe("guitar.wav",
                           instruments=["distorted_electric_guitar"],
                           batch_size=4):
    ...
```

- NoteEvent マッピング: onset=NoteStartEvent.start_time / offset=NoteEndEvent.end_time（index 突合）/ pitch=int。**velocity・confidence はモデルが出さない**ため定数埋め（velocity=100 相当、confidence=1.0）
- 楽器指定は正確な名前 `distorted_electric_guitar`（"dist" 略記は CLI 用。`muscriptor.tokenizer.mt3.resolve_instrument_names` で解決可能）。全35クラス
- instruments は出力フィルタではなく生成条件付け（ClassConditioner）
- 音声は 16kHz リサンプル・5秒チャンク処理
- ライセンス: コード MIT / 重み CC BY-NC（**非商用限定**）
