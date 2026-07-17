"""BasicPitchEngine のパラメータ受け渡しとランナーの引数検証。

実際の basic-pitch 推論は行わない（別 venv 前提のため）。
subprocess をモックしてコマンドラインの組み立てを検証する。
"""

import json
import subprocess
import sys
from pathlib import Path

from guitartab.transcribe.basicpitch import BasicPitchEngine

RUNNER = (
    Path(__file__).parent.parent
    / "src"
    / "guitartab"
    / "transcribe"
    / "_basicpitch_runner.py"
)


def _make_engine(tmp_path, **kwargs) -> BasicPitchEngine:
    fake_python = tmp_path / "python"
    fake_python.write_text("")  # 存在チェックを通すだけ
    return BasicPitchEngine(venv_python=fake_python, **kwargs)


def _capture_cmd(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        out_json = Path(cmd[3])
        out_json.write_text(json.dumps({"schema": 1, "notes": []}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_default_params_keep_legacy_cmd(monkeypatch, tmp_path):
    """パラメータ未指定なら従来どおり params 引数なし（後方互換）。"""
    engine = _make_engine(tmp_path)
    assert engine.predict_params == {}
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    engine.transcribe(audio)
    assert len(captured["cmd"]) == 4  # python runner audio out_json


def test_predict_params_passed_as_json(monkeypatch, tmp_path):
    engine = _make_engine(
        tmp_path,
        onset_threshold=0.7,
        frame_threshold=0.4,
        minimum_note_length=150.0,
        minimum_frequency=75.0,
        melodia_trick=False,
    )
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    engine.transcribe(audio)
    assert len(captured["cmd"]) == 5
    params = json.loads(captured["cmd"][4])
    assert params == {
        "onset_threshold": 0.7,
        "frame_threshold": 0.4,
        "minimum_note_length": 150.0,
        "minimum_frequency": 75.0,
        "melodia_trick": False,
    }
    # 未指定の maximum_frequency は渡さない
    assert "maximum_frequency" not in params


def test_runner_rejects_unknown_params():
    """ランナーは未知パラメータを basic_pitch import 前に拒否する。"""
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "a.wav", "out.json", '{"bogus": 1}'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "unknown predict() params" in proc.stderr


def test_runner_rejects_invalid_json():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "a.wav", "out.json", "not-json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "invalid params_json" in proc.stderr


def test_cli_flags_map_to_predict_params():
    """CLI の --bp-* フラグが build_engine 経由で predict_params に写る。"""
    import argparse

    from guitartab.cli import _add_common_engine_args, build_engine

    parser = argparse.ArgumentParser()

    _add_common_engine_args(parser)
    args = parser.parse_args(
        [
            "--bp-onset-threshold",
            "0.7",
            "--bp-minimum-note-length",
            "150",
            "--bp-minimum-frequency",
            "75",
            "--bp-no-melodia-trick",
        ]
    )
    engine = build_engine("basicpitch", args)
    assert engine.predict_params == {
        "onset_threshold": 0.7,
        "minimum_note_length": 150.0,
        "minimum_frequency": 75.0,
        "melodia_trick": False,
    }
