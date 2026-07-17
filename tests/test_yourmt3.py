"""YourMT3Engine のコマンド組み立て・venv/home 解決とランナーの引数検証。

実際の YourMT3 推論は行わない（別 venv + 538MB チェックポイント前提のため）。
subprocess をモックしてコマンドラインの組み立てを検証する（test_basicpitch.py と同型）。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from guitartab.transcribe import yourmt3 as ym
from guitartab.transcribe.yourmt3 import YourMT3Engine

RUNNER = (
    Path(__file__).parent.parent
    / "src"
    / "guitartab"
    / "transcribe"
    / "_yourmt3_runner.py"
)


def _make_home(tmp_path) -> Path:
    home = tmp_path / "yourmt3_home"
    (home / "amt" / "src").mkdir(parents=True)
    return home


def _make_engine(tmp_path, **kwargs) -> YourMT3Engine:
    fake_python = tmp_path / "python"
    fake_python.write_text("")  # 存在チェックを通すだけ
    kwargs.setdefault("home", _make_home(tmp_path))
    return YourMT3Engine(venv_python=fake_python, **kwargs)


def _capture_cmd(monkeypatch):
    captured: dict = {}

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        out_json = Path(cmd[3])
        out_json.write_text(json.dumps({"schema": 1, "notes": []}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_cmd_layout_and_default_params(monkeypatch, tmp_path):
    """cmd は python runner audio out_json params_json の5要素、device デフォルト cpu。"""
    engine = _make_engine(tmp_path)
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    engine.transcribe(audio)
    cmd = captured["cmd"]
    assert len(cmd) == 5
    assert cmd[1] == str(RUNNER)
    assert cmd[2] == str(audio)
    params = json.loads(cmd[4])
    assert params["device"] == "cpu"
    assert params["home"] == str((tmp_path / "yourmt3_home").resolve())
    assert "batch_size" not in params  # 未指定は渡さない（ランナー側デフォルト）


def test_device_and_batch_size_passed(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path, device="mps", batch_size=4)
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    engine.transcribe(audio)
    params = json.loads(captured["cmd"][4])
    assert params["device"] == "mps"
    assert params["batch_size"] == 4


def test_env_var_resolution(monkeypatch, tmp_path):
    """venv/home/device は環境変数からも解決される（引数がない場合）。"""
    fake_python = tmp_path / "envpython"
    fake_python.write_text("")
    home = _make_home(tmp_path)
    monkeypatch.setenv(ym.ENV_VAR, str(fake_python))
    monkeypatch.setenv(ym.HOME_ENV_VAR, str(home))
    monkeypatch.setenv(ym.DEVICE_ENV_VAR, "mps")
    engine = YourMT3Engine()
    assert engine.venv_python == fake_python
    assert engine.home == home
    assert engine.device == "mps"


def test_arg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ym.ENV_VAR, "/nonexistent/python")
    monkeypatch.setenv(ym.DEVICE_ENV_VAR, "mps")
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    engine = YourMT3Engine(venv_python=fake_python, device="cpu")
    assert engine.venv_python == fake_python
    assert engine.device == "cpu"


def test_default_paths(monkeypatch):
    monkeypatch.delenv(ym.ENV_VAR, raising=False)
    monkeypatch.delenv(ym.HOME_ENV_VAR, raising=False)
    monkeypatch.delenv(ym.DEVICE_ENV_VAR, raising=False)
    engine = YourMT3Engine()
    assert engine.venv_python == Path(".venv-yourmt3") / "bin" / "python"
    assert engine.home == Path("third_party") / "yourmt3"
    assert engine.device == "cpu"


def test_missing_audio_raises(tmp_path):
    engine = _make_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.transcribe(tmp_path / "missing.wav")


def test_missing_venv_raises_setup_hint(tmp_path):
    engine = YourMT3Engine(
        venv_python=tmp_path / "no-such-python", home=_make_home(tmp_path)
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="venv python not found"):
        engine.transcribe(audio)


def test_missing_home_raises_setup_hint(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    engine = YourMT3Engine(venv_python=fake_python, home=tmp_path / "no-such-home")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="home not found"):
        engine.transcribe(audio)


def test_runner_failure_raises(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path)

    def fail_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail_run)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="yourmt3 runner failed"):
        engine.transcribe(audio)


# --- ランナーの引数検証（重い import なしで即終了する経路のみ実行） ---


def test_runner_rejects_unknown_params():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "a.wav", "out.json", '{"bogus": 1}'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "unknown runner params" in proc.stderr


def test_runner_rejects_invalid_json():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "a.wav", "out.json", "not-json"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "invalid params_json" in proc.stderr


def test_runner_rejects_missing_home(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "a.wav",
            "out.json",
            json.dumps({"home": str(tmp_path / "nope")}),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "YourMT3 home not found" in proc.stderr


def test_runner_rejects_missing_audio(tmp_path):
    home = tmp_path / "home"
    (home / "amt" / "src").mkdir(parents=True)
    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            str(tmp_path / "missing.wav"),
            "out.json",
            json.dumps({"home": str(home)}),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "audio not found" in proc.stderr


def test_runner_usage_without_args():
    proc = subprocess.run(
        [sys.executable, str(RUNNER)], capture_output=True, text=True
    )
    assert proc.returncode == 2
    assert "usage:" in proc.stderr


def test_cli_flags_map_to_engine():
    """CLI の --yourmt3-* フラグが build_engine 経由でエンジンに写る。"""
    import argparse

    from guitartab.cli import _add_common_engine_args, build_engine

    parser = argparse.ArgumentParser()
    _add_common_engine_args(parser)
    args = parser.parse_args(
        [
            "--yourmt3-python",
            "/opt/ym/python",
            "--yourmt3-home",
            "/opt/ym/home",
            "--yourmt3-device",
            "mps",
        ]
    )
    engine = build_engine("yourmt3", args)
    assert engine.venv_python == Path("/opt/ym/python")
    assert engine.home == Path("/opt/ym/home")
    assert engine.device == "mps"
