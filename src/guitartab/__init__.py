"""guitartab v2 — YouTube URL からギター TAB 譜を生成するパイプライン。

設計は docs/DESIGN.md を参照。v2 の原則:
- 評価ファースト（凍結 GT + mir_eval）
- 転写エンジンは TranscriberEngine Protocol で差し替え可能
- ステージごとに work/{id}/ へ中間成果物を残しキャッシュ
"""

from guitartab.transcribe.base import NoteEvent, TranscriberEngine

__all__ = ["NoteEvent", "TranscriberEngine"]
