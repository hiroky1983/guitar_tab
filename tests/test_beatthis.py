"""BeatThisTempoEstimator のコマンド組み立て・venv 解決とランナーの引数検証。

実際の Beat This! 推論は行わない（別 venv + 78MB チェックポイント前提のため）。
subprocess をモックしてコマンドラインの組み立てを検証する（test_muscriptor.py と同型）。
拍列 → 一定テンポのフィット関数は合成拍列で検証する。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from guitartab.rhythm import beatthis as bt
from guitartab.rhythm.beatthis import (
    BeatThisTempoEstimator,
    beats_to_bpm,
    fit_constant_tempo,
    snap_beat_label,
)
from guitartab.rhythm.estimate import TempoEstimate

RUNNER = (
    Path(__file__).parent.parent
    / "src"
    / "guitartab"
    / "rhythm"
    / "_beatthis_runner.py"
)


def _make_estimator(tmp_path, **kwargs) -> BeatThisTempoEstimator:
    fake_python = tmp_path / "python"
    fake_python.write_text("")  # 存在チェックを通すだけ
    return BeatThisTempoEstimator(venv_python=fake_python, **kwargs)


def _capture_cmd(monkeypatch, *, beats=None, downbeats=None):
    captured: dict = {}
    beats = beats if beats is not None else [0.0, 0.5, 1.0, 1.5]
    downbeats = downbeats if downbeats is not None else [0.0]

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        out_json = Path(cmd[3])
        out_json.write_text(
            json.dumps(
                {"schema": 1, "beats_sec": beats, "downbeats_sec": downbeats}
            )
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def test_cmd_layout_and_default_params(monkeypatch, tmp_path):
    """cmd は python runner audio out_json params_json の5要素。
    デフォルトは final0 / cpu / dbn なし。
    """
    monkeypatch.delenv(bt.DEVICE_ENV_VAR, raising=False)
    estimator = _make_estimator(tmp_path)
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    estimator.track_beats(audio)
    cmd = captured["cmd"]
    assert len(cmd) == 5
    assert cmd[1] == str(RUNNER)
    assert cmd[2] == str(audio)
    params = json.loads(cmd[4])
    assert params == {"checkpoint": "final0", "device": "cpu", "dbn": False}


def test_param_passthrough(monkeypatch, tmp_path):
    estimator = _make_estimator(
        tmp_path, checkpoint="small0", device="mps", dbn=True
    )
    captured = _capture_cmd(monkeypatch)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    estimator.track_beats(audio)
    params = json.loads(captured["cmd"][4])
    assert params == {"checkpoint": "small0", "device": "mps", "dbn": True}


def test_env_var_resolution(monkeypatch, tmp_path):
    """venv/device は環境変数からも解決される（引数がない場合）。"""
    fake_python = tmp_path / "envpython"
    fake_python.write_text("")
    monkeypatch.setenv(bt.ENV_VAR, str(fake_python))
    monkeypatch.setenv(bt.DEVICE_ENV_VAR, "mps")
    estimator = BeatThisTempoEstimator()
    assert estimator.venv_python == fake_python
    assert estimator.device == "mps"


def test_arg_overrides_env(monkeypatch, tmp_path):
    monkeypatch.setenv(bt.ENV_VAR, "/nonexistent/python")
    monkeypatch.setenv(bt.DEVICE_ENV_VAR, "mps")
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    estimator = BeatThisTempoEstimator(venv_python=fake_python, device="cpu")
    assert estimator.venv_python == fake_python
    assert estimator.device == "cpu"


def test_default_paths(monkeypatch):
    monkeypatch.delenv(bt.ENV_VAR, raising=False)
    monkeypatch.delenv(bt.DEVICE_ENV_VAR, raising=False)
    estimator = BeatThisTempoEstimator()
    assert estimator.venv_python == Path(".venv-beatthis") / "bin" / "python"
    assert estimator.device == "cpu"
    assert estimator.checkpoint == "final0"
    assert estimator.dbn is False


def test_missing_audio_raises(tmp_path):
    estimator = _make_estimator(tmp_path)
    with pytest.raises(FileNotFoundError):
        estimator.track_beats(tmp_path / "missing.wav")


def test_missing_venv_raises_setup_hint(tmp_path):
    estimator = BeatThisTempoEstimator(venv_python=tmp_path / "no-such-python")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="venv python not found"):
        estimator.track_beats(audio)


def test_runner_failure_raises(monkeypatch, tmp_path):
    estimator = _make_estimator(tmp_path)

    def fail_run(cmd, capture_output, text):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")

    monkeypatch.setattr(subprocess, "run", fail_run)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    with pytest.raises(RuntimeError, match="beatthis runner failed"):
        estimator.track_beats(audio)


# --- 拍列 → 一定テンポ + 位相のフィット（純関数、推論なし） ---


def test_beats_to_bpm_median_ioi():
    beats = np.arange(0.25, 20.0, 0.5)  # 120 BPM
    assert beats_to_bpm(beats) == pytest.approx(120.0)


def test_beats_to_bpm_folds_into_range():
    # 15 BPM 相当（周期 4 秒）→ 2 倍折り畳みで 60 BPM
    assert beats_to_bpm([0.0, 4.0, 8.0, 12.0]) == pytest.approx(60.0)


def test_beats_to_bpm_too_few_beats():
    assert beats_to_bpm([1.0]) is None
    assert beats_to_bpm([]) is None


def test_fit_constant_tempo_recovers_grid():
    period, phase0 = 0.5, 0.25
    beats = phase0 + np.arange(40) * period
    fit = fit_constant_tempo(beats)
    assert fit is not None
    bpm, phase = fit
    assert bpm == pytest.approx(120.0, rel=1e-6)
    assert phase == pytest.approx(phase0, abs=1e-6)


def test_fit_constant_tempo_tolerates_missing_beats():
    period = 0.5
    beats = np.delete(np.arange(40) * period, [3, 10, 11, 25])
    fit = fit_constant_tempo(beats)
    assert fit is not None
    assert fit[0] == pytest.approx(120.0, rel=1e-6)


def test_fit_constant_tempo_too_few():
    assert fit_constant_tempo([1.0]) is None


def test_snap_beat_label_picks_quarter_shift():
    # 真の拍位相 0.30、phi16 = 0.05（P/4 = 0.125 の 2 シフト分ずれ）
    period = 0.5
    beats = 0.30 + np.arange(30) * period
    phase = snap_beat_label(beats, 120.0, 0.05)
    assert phase == pytest.approx(0.30, abs=1e-6)


def test_snap_beat_label_empty_beats_returns_phi16():
    assert snap_beat_label([], 120.0, 0.1) == pytest.approx(0.1)


# --- estimate() の経路（subprocess モック / フォールバック） ---


def test_estimate_returns_protocol_result(monkeypatch, tmp_path):
    """音声ありの正常系: Beat This! のレベルで TempoEstimate を返す。"""
    estimator = _make_estimator(tmp_path)
    beats = list(0.25 + np.arange(60) * 0.5)  # 120 BPM、位相 0.25
    _capture_cmd(monkeypatch, beats=beats)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    onsets = list(0.25 + np.arange(0, 30, 0.25))  # 8 分刻みのノート
    result = estimator.estimate(onsets, audio_path=audio)
    assert isinstance(result, TempoEstimate)
    assert result.estimator == "beatthis_constant"
    assert result.bpm == pytest.approx(120.0, rel=0.01)
    assert 0.0 <= result.grid_origin_sec < result.beat_period_sec
    assert "fallback" not in result.params  # 正常経路はフォールバックなし


def test_estimate_no_audio_falls_back_to_librosa_notes_only(tmp_path):
    """音声なしはサブプロセスを起動せず librosa のノートのみ経路へ委譲する。"""
    estimator = BeatThisTempoEstimator(venv_python=tmp_path / "missing")
    onsets = list(np.arange(0, 30, 0.5))
    result = estimator.estimate(onsets)  # venv 不在でも失敗しないこと
    assert result.estimator == "beatthis_constant"
    assert result.params["fallback"] == "librosa_notes_only"


def test_estimate_few_beats_uses_ls_fit_fallback(monkeypatch, tmp_path):
    """拍が 2 個未満なら librosa（音声あり）フォールバックのパラメータが記録される。"""
    estimator = _make_estimator(tmp_path)
    _capture_cmd(monkeypatch, beats=[1.0])

    fallback_estimate = TempoEstimate(100.0, 0.1, "librosa_constant", {})
    monkeypatch.setattr(
        estimator._librosa,
        "estimate",
        lambda onsets, audio_path=None: fallback_estimate,
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    result = estimator.estimate([0.0, 0.5, 1.0], audio_path=audio)
    assert result.estimator == "beatthis_constant"
    assert result.bpm == 100.0
    assert result.params["fallback"] == "librosa_audio"


def test_estimate_few_onsets_uses_beat_ls_fit(monkeypatch, tmp_path):
    """ノート 8 個未満は拍列の素の最小二乗フィット（格子適合が立たないため）。"""
    estimator = _make_estimator(tmp_path)
    beats = list(np.arange(40) * 0.5)
    _capture_cmd(monkeypatch, beats=beats)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"")
    result = estimator.estimate([0.0, 0.5, 1.0], audio_path=audio)
    assert result.bpm == pytest.approx(120.0, rel=1e-6)
    assert result.params["fallback"] == "beats_ls_fit"


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
