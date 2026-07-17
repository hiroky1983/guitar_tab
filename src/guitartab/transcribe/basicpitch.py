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
    """Spotify basic-pitch によるベースライン転写エンジン（別 venv サブプロセス実行）。"""

    name = "basicpitch"

    def __init__(self, venv_python: Path | str | None = None):
        resolved = venv_python or os.environ.get(ENV_VAR) or DEFAULT_VENV_PYTHON
        self.venv_python = Path(resolved)

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
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"basic-pitch runner failed (exit {proc.returncode}): "
                    f"{' '.join(cmd)}\n{proc.stderr}"
                )
            return load_notes(out_json)
