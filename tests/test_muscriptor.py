"""MuScriptorEngine のコマンド組み立て・venv 解決とランナーの引数検証・暴走検知。

実際の MuScriptor 推論は行わない（別 venv + ゲート付き HF 重み前提のため）。
subprocess をモックしてコマンドラインの組み立てを検証する（test_yourmt3.py と同型）。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from guitartab.transcribe import _muscriptor_runner as runner_mod
from guitartab.transcribe import muscriptor as ms
from guitartab.transcribe.muscriptor import MuScriptorEngine

RUNNER = (
    Path(__file__).parent.parent
    / "src"
    / "guitartab"
    / "transcribe"
    / "_muscriptor_runner.py"
)


def _make_engine(tmp_path, **kwargs) -> MuScriptorEngine:
    fake_python = tmp_path / "python"
    fake_python.write_text("")  # 存在チェックを通すだけ
    return MuScriptorEngine(venv_python=fake_python, **kwargs)


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
    """cmd は python runner audio out_json params_json の5要素。
    デフォルトはベンチ採用構成（ac+dist / cfg 1.5 / batch 4 / mps / greedy / 暴走閾値 30）。
    """
    monkeypatch.delenv(ms.DEVICE_ENV_VAR, raising=False)
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
    assert params["model"] == "small"
    assert params["device"] == "mps"
    assert params["batch_size"] == 4
    assert params["instruments"] == ["acoustic_guitar", "distorted_electric_guitar"]
    assert params["cfg_coef"] == 1.5
    assert params["max_notes_per_sec"] == 30.0
    # greedy がデフォルト（sampling は明示要求時のみ渡す）
    assert "use_sampling" not in params
    assert "temperature" not in params


def test_param_passthrough(monkeypatch, tmp_path):
    engine = _make_engine(
        tmp_path,
        instruments=["distorted_electric_guitar"],
        cfg_coef=1.25,
        batch_size=8,
        device="cpu",
        use_sampling=True,
        temperature=0.7,
        max_notes_per_sec=10,
    )
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    engine.transcribe(audio)
    params = json.loads(captured["cmd"][4])
    assert params["instruments"] == ["distorted_electric_guitar"]
    assert params["cfg_coef"] == 1.25
    assert params["batch_size"] == 8
    assert params["device"] == "cpu"
    assert params["use_sampling"] is True
    assert params["temperature"] == 0.7
    assert params["max_notes_per_sec"] == 10


def test_env_var_resolution(monkeypatch, tmp_path):
    """venv/device は環境変数からも解決される（引数がない場合）。"""
    fake_python = tmp_path / "envpython"
    fake_python.write_text("")
    monkeypatch.setenv(ms.ENV_VAR, str(fake_python))
    monkeypatch.setenv(ms.DEVICE_ENV_VAR, "cpu")
    engine = MuScriptorEngine()
    assert engine.venv_python == fake_python
    assert engine.device == "cpu"


def test_arg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ms.ENV_VAR, "/nonexistent/python")
    monkeypatch.setenv(ms.DEVICE_ENV_VAR, "cpu")
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    engine = MuScriptorEngine(venv_python=fake_python, device="mps")
    assert engine.venv_python == fake_python
    assert engine.device == "mps"


def test_default_paths(monkeypatch):
    monkeypatch.delenv(ms.ENV_VAR, raising=False)
    monkeypatch.delenv(ms.DEVICE_ENV_VAR, raising=False)
    engine = MuScriptorEngine()
    assert engine.venv_python == Path(".venv-muscriptor") / "bin" / "python"
    assert engine.device == "mps"
    assert engine.instruments == ["acoustic_guitar", "distorted_electric_guitar"]
    assert engine.cfg_coef == 1.5
    assert engine.batch_size == 4
    assert engine.max_notes_per_sec == 30.0


def test_empty_instruments_rejected():
    with pytest.raises(ValueError, match="instruments"):
        MuScriptorEngine(instruments=[])


def test_missing_audio_raises(tmp_path):
    engine = _make_engine(tmp_path)
    with pytest.raises(FileNotFoundError):
        engine.transcribe(tmp_path / "missing.wav")


def test_missing_venv_raises_setup_hint(tmp_path):
    engine = MuScriptorEngine(venv_python=tmp_path / "no-such-python")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="venv python not found"):
        engine.transcribe(audio)


def test_runner_failure_raises(monkeypatch, tmp_path):
    engine = _make_engine(tmp_path)

    def fail_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail_run)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="muscriptor runner failed"):
        engine.transcribe(audio)


def test_runaway_exit_code_formats_error(monkeypatch, tmp_path):
    """exit 3（暴走検知）は整形されたエラーになり、stderr のノート数を含める。"""
    engine = _make_engine(tmp_path)
    stderr = (
        "runaway generation detected: 4200 notes in 60.0s audio "
        "= 70.0 notes/sec (limit 30 notes/sec)"
    )

    def runaway_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr=stderr)

    monkeypatch.setattr(subprocess, "run", runaway_run)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="runaway") as excinfo:
        engine.transcribe(audio)
    msg = str(excinfo.value)
    assert "4200 notes" in msg
    assert "cfg_coef" in msg  # 対処のヒント（縮退モードの既知情報）を含む


# --- 暴走検知ロジック（ランナーの純粋関数。重い import なし） ---


def test_is_runaway_threshold():
    assert runner_mod.is_runaway(4200, 60.0, 30.0)  # 70 notes/sec > 30
    assert not runner_mod.is_runaway(1200, 60.0, 30.0)  # 20 notes/sec
    assert not runner_mod.is_runaway(1800, 60.0, 30.0)  # ちょうど 30 は許容
    assert not runner_mod.is_runaway(4200, 60.0, 0)  # 閾値 0 = 無効
    assert not runner_mod.is_runaway(100, 0.0, 30.0)  # 長さ不明はスキップ


def test_runner_exit_code_constant_in_sync():
    """エンジン側 EXIT_RUNAWAY はランナー側と手動同期（ずれたら暴走が握り潰される）。"""
    assert ms.EXIT_RUNAWAY == runner_mod.EXIT_RUNAWAY == 3


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


def test_runner_rejects_empty_instruments():
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "a.wav", "out.json", '{"instruments": []}'],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "instruments" in proc.stderr


def test_runner_rejects_missing_audio(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(RUNNER), str(tmp_path / "missing.wav"), "out.json"],
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
    """CLI の --ms-* フラグが build_engine 経由でエンジンに写る。"""
    import argparse

    from guitartab.cli import _add_common_engine_args, build_engine

    parser = argparse.ArgumentParser()
    _add_common_engine_args(parser)
    args = parser.parse_args(
        [
            "--muscriptor-python",
            "/opt/ms/python",
            "--ms-instruments",
            "acoustic_guitar, distorted_electric_guitar",
            "--ms-cfg-coef",
            "1.4",
            "--ms-batch-size",
            "2",
            "--ms-device",
            "cpu",
        ]
    )
    engine = build_engine("muscriptor", args)
    assert engine.venv_python == Path("/opt/ms/python")
    assert engine.instruments == ["acoustic_guitar", "distorted_electric_guitar"]
    assert engine.cfg_coef == 1.4
    assert engine.batch_size == 2
    assert engine.device == "cpu"


def test_cli_defaults_are_best_config(monkeypatch):
    """フラグ未指定でベンチ採用のベスト構成に配線される（dev F1 0.879 の再現条件）。"""
    import argparse

    from guitartab.cli import _add_common_engine_args, build_engine

    monkeypatch.delenv(ms.DEVICE_ENV_VAR, raising=False)
    parser = argparse.ArgumentParser()
    _add_common_engine_args(parser)
    args = parser.parse_args([])
    engine = build_engine("muscriptor", args)
    assert engine.instruments == ["acoustic_guitar", "distorted_electric_guitar"]
    assert engine.cfg_coef == 1.5
    assert engine.batch_size == 4
    assert engine.device == "mps"
    assert engine.model == "small"


def test_cli_rejects_empty_instruments():
    import argparse

    from guitartab.cli import _add_common_engine_args, build_engine

    parser = argparse.ArgumentParser()
    _add_common_engine_args(parser)
    args = parser.parse_args(["--ms-instruments", " , "])
    with pytest.raises(SystemExit, match="ms-instruments"):
        build_engine("muscriptor", args)
