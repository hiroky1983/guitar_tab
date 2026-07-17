"""basic-pitch エンジン（サブプロセス実行方式）。

basic-pitch は Apple Silicon では Python 3.10 限定のため、本体 venv（3.11）には
インストールせず、専用 venv の python で _basicpitch_runner.py を実行して
notes.json を受け取る。venv パスの指定方法（優先順）:

1. コンストラクタ引数 venv_python
2. 環境変数 GUITARTAB_BASICPITCH_PYTHON
3. プロジェクト直下の .venv-basicpitch/bin/python（デフォルト）

セットアップ手順は README「basic-pitch の運用」を参照。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from guitartab.transcribe.base import NoteEvent, load_notes

ENV_VAR = "GUITARTAB_BASICPITCH_PYTHON"
DEFAULT_VENV_PYTHON = Path(".venv-basicpitch") / "bin" / "python"

_RUNNER = Path(__file__).parent / "_basicpitch_runner.py"

_SETUP_HINT = (
    "basic-pitch venv python not found: {python}\n"
    "Set up a dedicated Python 3.10 venv, e.g.:\n"
    "    uv venv --python 3.10 .venv-basicpitch\n"
    "    uv pip install --python .venv-basicpitch/bin/python basic-pitch\n"
    f"then pass venv_python= or set {ENV_VAR}."
)


class BasicPitchEngine:
    """Spotify basic-pitch によるベースライン転写エンジン（別 venv サブプロセス実行）。

    onset_threshold 以下のキーワード引数は basic_pitch.inference.predict() の
    ネイティブ推論パラメータで、None のものは渡さない（= predict() のデフォルト。
    従来動作と同一）。ランナー側 PREDICT_PARAMS と対応。
    """

    name = "basicpitch"

    def __init__(
        self,
        venv_python: Path | str | None = None,
        *,
        onset_threshold: float | None = None,
        frame_threshold: float | None = None,
        minimum_note_length: float | None = None,
        minimum_frequency: float | None = None,
        maximum_frequency: float | None = None,
        melodia_trick: bool | None = None,
    ):
        resolved = venv_python or os.environ.get(ENV_VAR) or DEFAULT_VENV_PYTHON
        self.venv_python = Path(resolved)
        params = {
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "minimum_note_length": minimum_note_length,
            "minimum_frequency": minimum_frequency,
            "maximum_frequency": maximum_frequency,
            "melodia_trick": melodia_trick,
        }
        self.predict_params: dict = {k: v for k, v in params.items() if v is not None}

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not self.venv_python.exists():
            raise RuntimeError(_SETUP_HINT.format(python=self.venv_python))

        with tempfile.TemporaryDirectory(prefix="guitartab-basicpitch-") as tmp:
            out_json = Path(tmp) / "notes.json"
            cmd = [
                str(self.venv_python),
                str(_RUNNER),
                str(audio_path),
                str(out_json),
            ]
            if self.predict_params:
                cmd.append(json.dumps(self.predict_params))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"basic-pitch runner failed (exit {proc.returncode}): "
                    f"{' '.join(cmd)}\n{proc.stderr}"
                )
            return load_notes(out_json)
