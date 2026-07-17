"""YourMT3+ エンジン（サブプロセス実行方式）。

YourMT3+ は GPL/Apache 混在ライセンスのためコードをリポジトリに同梱せず、
gitignore 済みの third_party/yourmt3/ に別途配置して（HF Space mimbres/YourMT3
から取得）、専用 venv の python で _yourmt3_runner.py を実行して notes.json を
受け取る（basicpitch.py と同じ疎結合方式。docs/YOURMT3_VERIFICATION_2026-07-17.md）。

venv パスの指定方法（優先順）:

1. コンストラクタ引数 venv_python
2. 環境変数 GUITARTAB_YOURMT3_PYTHON
3. プロジェクト直下の .venv-yourmt3/bin/python（デフォルト）

コード+チェックポイントの場所は home 引数 > 環境変数 GUITARTAB_YOURMT3_HOME >
third_party/yourmt3。デバイスは device 引数 > 環境変数 GUITARTAB_YOURMT3_DEVICE >
"cpu"（M2 では MPS と同速のため安定側の CPU をデフォルトとする）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from guitartab.transcribe.base import NoteEvent, load_notes

ENV_VAR = "GUITARTAB_YOURMT3_PYTHON"
DEFAULT_VENV_PYTHON = Path(".venv-yourmt3") / "bin" / "python"

HOME_ENV_VAR = "GUITARTAB_YOURMT3_HOME"
DEFAULT_HOME = Path("third_party") / "yourmt3"

DEVICE_ENV_VAR = "GUITARTAB_YOURMT3_DEVICE"
DEFAULT_DEVICE = "cpu"

_RUNNER = Path(__file__).parent / "_yourmt3_runner.py"

_SETUP_HINT = (
    "YourMT3 venv python not found: {python}\n"
    "Set up a dedicated Python 3.11 venv (transformers==4.45.1 / numpy==1.26.4), "
    "see README, then pass venv_python= or set " + ENV_VAR + "."
)

_HOME_HINT = (
    "YourMT3 home not found or invalid (missing amt/src): {home}\n"
    "Download the YourMT3 code + checkpoint (HF Space mimbres/YourMT3) into "
    "third_party/yourmt3, or pass home= / set " + HOME_ENV_VAR + "."
)


class YourMT3Engine:
    """YourMT3+ (YPTF.MoE+Multi noPS) による転写エンジン（別 venv サブプロセス実行）。

    注意: 学習データに GuitarSet を含むため、GuitarSet ベンチの数値は
    汚染の可能性があり参考値扱い（docs/YOURMT3_VERIFICATION_2026-07-17.md）。
    """

    name = "yourmt3"

    def __init__(
        self,
        venv_python: Path | str | None = None,
        *,
        home: Path | str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ):
        resolved = venv_python or os.environ.get(ENV_VAR) or DEFAULT_VENV_PYTHON
        self.venv_python = Path(resolved)
        resolved_home = home or os.environ.get(HOME_ENV_VAR) or DEFAULT_HOME
        self.home = Path(resolved_home)
        self.device = device or os.environ.get(DEVICE_ENV_VAR) or DEFAULT_DEVICE
        self.batch_size = batch_size

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not self.venv_python.exists():
            raise RuntimeError(_SETUP_HINT.format(python=self.venv_python))
        if not (self.home / "amt" / "src").is_dir():
            raise RuntimeError(_HOME_HINT.format(home=self.home))

        params: dict = {
            "home": str(self.home.resolve()),
            "device": self.device,
        }
        if self.batch_size is not None:
            params["batch_size"] = self.batch_size

        with tempfile.TemporaryDirectory(prefix="guitartab-yourmt3-") as tmp:
            out_json = Path(tmp) / "notes.json"
            cmd = [
                str(self.venv_python),
                str(_RUNNER),
                str(audio_path),
                str(out_json),
                json.dumps(params),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
            if proc.returncode != 0:
                raise RuntimeError(
                    f"yourmt3 runner failed (exit {proc.returncode}): "
                    f"{' '.join(cmd)}\n{proc.stderr}"
                )
            return load_notes(out_json)
