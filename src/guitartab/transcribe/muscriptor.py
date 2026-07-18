"""MuScriptor エンジン（サブプロセス実行方式）。

MuScriptor（Kyutai, 2026-07 公開）は torch 依存のため本体 venv には入れず、
専用 venv の python で _muscriptor_runner.py を実行して notes.json を受け取る
（basicpitch.py / yourmt3.py と同じ疎結合方式）。重みはゲート付き HF リポジトリ
（CC BY-NC・非商用限定）にあり、初回 DL には HF_TOKEN が必要
（`.env` → `python -m guitartab` 起動時に os.environ へ読み込まれ、
サブプロセスにそのまま継承される）。検証記録は
docs/MUSCRIPTOR_VERIFICATION_2026-07-17.md / docs/MUSCRIPTOR_RESULTS_2026-07-17.md。

venv パスの指定方法（優先順）:

1. コンストラクタ引数 venv_python
2. 環境変数 GUITARTAB_MUSCRIPTOR_PYTHON
3. プロジェクト直下の .venv-muscriptor/bin/python（デフォルト）

デバイスは device 引数 > 環境変数 GUITARTAB_MUSCRIPTOR_DEVICE > "mps"
（MPS は明示指定が必須のためデフォルトで指定。batch_size>=4 とセットで実用速度）。

デフォルト生成パラメータ instruments=["acoustic_guitar", "distorted_electric_guitar"] +
cfg_coef=1.5 は 2026-07-18 スイープのベスト構成（クリーン dev F1 0.879、
docs/BENCHMARKS.md）。cfg_coef 1.6 以上は生成が縮退暴走する実測があるため、
ランナー側にノート数暴走検知（デフォルト 30 notes/sec 超過で exit 3）を備える。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from guitartab.transcribe.base import NoteEvent, load_notes

ENV_VAR = "GUITARTAB_MUSCRIPTOR_PYTHON"
DEFAULT_VENV_PYTHON = Path(".venv-muscriptor") / "bin" / "python"

DEVICE_ENV_VAR = "GUITARTAB_MUSCRIPTOR_DEVICE"
DEFAULT_DEVICE = "mps"

DEFAULT_MODEL = "small"
DEFAULT_INSTRUMENTS = ("acoustic_guitar", "distorted_electric_guitar")
DEFAULT_CFG_COEF = 1.5
DEFAULT_BATCH_SIZE = 4
DEFAULT_MAX_NOTES_PER_SEC = 30.0

# ランナー側 EXIT_RUNAWAY と手動同期（本体からランナーを import しない方針のため）
EXIT_RUNAWAY = 3

_RUNNER = Path(__file__).parent / "_muscriptor_runner.py"

_SETUP_HINT = (
    "muscriptor venv python not found: {python}\n"
    "Set up a dedicated venv (Python 3.11, `uv pip install muscriptor`), "
    "see README, then pass venv_python= or set " + ENV_VAR + ".\n"
    "Note: the weights are gated on HF (CC BY-NC) and need HF_TOKEN in .env."
)

_RUNAWAY_HINT = (
    "muscriptor generation runaway detected (> {limit:g} notes/sec):\n"
    "{detail}\n"
    "既知の縮退モード（cfg_coef 1.6 以上で発生、docs/BENCHMARKS.md 2026-07-18 スイープ）。"
    "--ms-cfg-coef を下げるか、入力音声（無音・非楽音でないか）を確認してください。"
)


class MuScriptorEngine:
    """MuScriptor small による転写エンジン（別 venv サブプロセス実行）。

    クリーンギターの正式エンジン（docs/DESIGN.md「エンジン採用決定（2026-07-18）」）。
    歪みエレキは basic-pitch（M1 tuned 構成）が上回るため使い分ける。
    重みは CC BY-NC（非商用限定）。
    """

    name = "muscriptor"

    def __init__(
        self,
        venv_python: Path | str | None = None,
        *,
        model: str | None = None,
        instruments: list[str] | None = None,
        cfg_coef: float | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        use_sampling: bool = False,
        temperature: float | None = None,
        max_notes_per_sec: float | None = None,
    ):
        resolved = venv_python or os.environ.get(ENV_VAR) or DEFAULT_VENV_PYTHON
        self.venv_python = Path(resolved)
        self.model = model or DEFAULT_MODEL
        if instruments is not None and not instruments:
            raise ValueError("instruments must contain at least one instrument name")
        self.instruments = (
            list(instruments) if instruments is not None else list(DEFAULT_INSTRUMENTS)
        )
        self.cfg_coef = cfg_coef if cfg_coef is not None else DEFAULT_CFG_COEF
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE
        self.device = device or os.environ.get(DEVICE_ENV_VAR) or DEFAULT_DEVICE
        self.use_sampling = use_sampling
        self.temperature = temperature
        self.max_notes_per_sec = (
            max_notes_per_sec
            if max_notes_per_sec is not None
            else DEFAULT_MAX_NOTES_PER_SEC
        )

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not self.venv_python.exists():
            raise RuntimeError(_SETUP_HINT.format(python=self.venv_python))

        params: dict = {
            "model": self.model,
            "device": self.device,
            "batch_size": self.batch_size,
            "instruments": self.instruments,
            "cfg_coef": self.cfg_coef,
            "max_notes_per_sec": self.max_notes_per_sec,
        }
        if self.use_sampling:
            params["use_sampling"] = True
            if self.temperature is not None:
                params["temperature"] = self.temperature

        with tempfile.TemporaryDirectory(prefix="guitartab-muscriptor-") as tmp:
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
            if proc.returncode == EXIT_RUNAWAY:
                raise RuntimeError(
                    _RUNAWAY_HINT.format(
                        limit=self.max_notes_per_sec, detail=proc.stderr.strip()
                    )
                )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"muscriptor runner failed (exit {proc.returncode}): "
                    f"{' '.join(cmd)}\n{proc.stderr}"
                )
            return load_notes(out_json)
