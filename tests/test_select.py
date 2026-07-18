"""エンジン自動選択（--engine auto、transcribe/select.py）のテスト。

実推論は行わない。合成波形で判定器の境界を、モック/キャッシュ済み notes.json で
配線を検証する。閾値の実測根拠は docs/BENCHMARKS.md「エンジン自動選択」節。
"""

import json
import math

import numpy as np
import pytest
import soundfile as sf

from guitartab import pipeline
from guitartab.pipeline import run_transcribe_pipeline
from guitartab.transcribe.base import NoteEvent, save_notes
from guitartab.transcribe.basicpitch import BasicPitchEngine
from guitartab.transcribe.muscriptor import MuScriptorEngine
from guitartab.transcribe.select import (
    BASICPITCH_TUNED_PRESET,
    CREST_FACTOR_THRESHOLD,
    ENGINE_SELECTION_FILENAME,
    SPECTRAL_FLATNESS_THRESHOLD,
    AutoEngineSelector,
    DistortionFeatures,
    classify_distortion,
    compute_distortion_features,
    muscriptor_unavailable_reason,
)

SR = 22050


def _write_wav(path, y):
    sf.write(str(path), y.astype(np.float64), SR)
    return path


def _clean_like_wav(path):
    """鋭いアタック + 静かな持続部のピッキング = 高クレスト・低平坦度。"""
    t = np.arange(SR * 2) / SR
    y = np.zeros_like(t)
    for start in (0.0, 0.5, 1.0, 1.5):
        seg = (t >= start) & (t < start + 0.3)
        tt = t[seg] - start
        env = np.exp(-tt / 0.005) + 0.25 * np.exp(-tt / 0.1)
        y[seg] = env * np.sin(2 * math.pi * 220.0 * tt)
    return _write_wav(path, y / np.max(np.abs(y)))


def _distorted_like_wav(path):
    """ハードクリップした持続音 = 低クレスト（矩形波状）。"""
    t = np.arange(SR * 2) / SR
    y = np.clip(10.0 * np.sin(2 * math.pi * 220.0 * t), -1.0, 1.0)
    return _write_wav(path, y)


# ---------------------------------------------------------------------------
# 判定器


def test_preset_constants_are_m1_tuned():
    """auto の basic-pitch プリセット = M1 tuned 構成（docs/BENCHMARKS.md）。"""
    assert BASICPITCH_TUNED_PRESET == {
        "onset_threshold": 0.75,
        "frame_threshold": 0.4,
        "minimum_note_length": 100.0,
    }


def test_thresholds_between_measured_clusters():
    """閾値は実測クラスタ（クリーン最小 6.85 / crunch 最大 3.67 等）の間にある。"""
    assert 3.67 < CREST_FACTOR_THRESHOLD < 6.85
    assert 0.254 < SPECTRAL_FLATNESS_THRESHOLD < 0.788


def test_clean_like_waveform_classified_clean(tmp_path):
    wav = _clean_like_wav(tmp_path / "clean.wav")
    f = compute_distortion_features(wav)
    assert f.crest_factor >= CREST_FACTOR_THRESHOLD
    assert f.spectral_flatness < SPECTRAL_FLATNESS_THRESHOLD
    assert classify_distortion(f) == "clean"


def test_clipped_waveform_classified_distorted(tmp_path):
    wav = _distorted_like_wav(tmp_path / "dist.wav")
    f = compute_distortion_features(wav)
    assert f.crest_factor < CREST_FACTOR_THRESHOLD  # 矩形波状 ≈ 1.0
    assert classify_distortion(f) == "distorted"


def test_silence_classified_distorted(tmp_path):
    """完全無音は判定不能 → distorted 側（デフォルトエンジン）に倒す。"""
    wav = _write_wav(tmp_path / "silence.wav", np.zeros(SR))
    assert classify_distortion(compute_distortion_features(wav)) == "distorted"


def test_classify_rule_boundary_cases():
    """判定ルールの境界（実測値による回帰）。"""
    # 実曲ステム wr7xTGTG-Mo/guitar.wav: クレスト単独ではクリーン誤判定、
    # 平坦度 OR 条件で distorted に救済される（実測 6.30 / 0.788）
    real_stem = DistortionFeatures(6.30, 0.788, 0.434, 86.2)
    assert real_stem.crest_factor >= CREST_FACTOR_THRESHOLD
    assert classify_distortion(real_stem) == "distorted"
    # クリーン dev の典型（00_BN3 実測 8.59 / 0.144）
    assert classify_distortion(DistortionFeatures(8.59, 0.144, 0.008, 30.0)) == "clean"
    # crunch の最も浅い実測（00_Rock2 3.67 / 0.231）はクレストで distorted
    assert (
        classify_distortion(DistortionFeatures(3.67, 0.231, 0.009, 30.0)) == "distorted"
    )


# ---------------------------------------------------------------------------
# MuScriptor 可用性チェック


def test_muscriptor_unavailable_when_venv_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    reason = muscriptor_unavailable_reason(tmp_path / "no-such-python")
    assert reason is not None and "venv python not found" in reason


def test_muscriptor_unavailable_when_no_hf_token(tmp_path, monkeypatch):
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    reason = muscriptor_unavailable_reason(fake_python)
    assert reason is not None and "HF_TOKEN" in reason


def test_muscriptor_available(tmp_path, monkeypatch):
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    assert muscriptor_unavailable_reason(fake_python) is None


# ---------------------------------------------------------------------------
# AutoEngineSelector.resolve（実推論なし）


def test_resolve_distorted_selects_basicpitch_with_preset(tmp_path):
    wav = _distorted_like_wav(tmp_path / "dist.wav")
    selector = AutoEngineSelector()
    selection_path = tmp_path / ENGINE_SELECTION_FILENAME
    engine = selector.resolve(wav, selection_path=selection_path)
    assert isinstance(engine, BasicPitchEngine)
    assert engine.predict_params == BASICPITCH_TUNED_PRESET

    payload = json.loads(selection_path.read_text())
    assert payload["verdict"] == "distorted"
    assert payload["engine"] == "basicpitch"
    assert payload["engine_params"] == BASICPITCH_TUNED_PRESET
    assert payload["fallback_reason"] is None
    assert payload["thresholds"]["crest_factor"] == CREST_FACTOR_THRESHOLD


def test_resolve_clean_selects_muscriptor(tmp_path, monkeypatch):
    wav = _clean_like_wav(tmp_path / "clean.wav")
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    selector = AutoEngineSelector(muscriptor_python=fake_python)
    selection_path = tmp_path / ENGINE_SELECTION_FILENAME
    engine = selector.resolve(wav, selection_path=selection_path)
    assert isinstance(engine, MuScriptorEngine)

    payload = json.loads(selection_path.read_text())
    assert payload["verdict"] == "clean"
    assert payload["engine"] == "muscriptor"
    assert payload["fallback_reason"] is None
    # 採用構成（ac+dist / cfg 1.5）がそのまま記録される
    assert payload["engine_params"]["cfg_coef"] == 1.5


def test_resolve_clean_falls_back_when_muscriptor_missing(tmp_path, monkeypatch, capsys):
    wav = _clean_like_wav(tmp_path / "clean.wav")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    selector = AutoEngineSelector(muscriptor_python=tmp_path / "no-such-python")
    selection_path = tmp_path / ENGINE_SELECTION_FILENAME
    engine = selector.resolve(wav, selection_path=selection_path)
    assert isinstance(engine, BasicPitchEngine)
    assert engine.predict_params == BASICPITCH_TUNED_PRESET
    assert "falling back to basic-pitch" in capsys.readouterr().err

    payload = json.loads(selection_path.read_text())
    assert payload["verdict"] == "clean"
    assert payload["engine"] == "basicpitch"
    assert "venv python not found" in payload["fallback_reason"]


def test_resolve_clean_falls_back_without_hf_token(tmp_path, monkeypatch):
    wav = _clean_like_wav(tmp_path / "clean.wav")
    fake_python = tmp_path / "python"
    fake_python.write_text("")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    engine = AutoEngineSelector(muscriptor_python=fake_python).resolve(wav)
    assert isinstance(engine, BasicPitchEngine)


def test_resolve_bp_overrides_win_over_preset(tmp_path):
    wav = _distorted_like_wav(tmp_path / "dist.wav")
    selector = AutoEngineSelector(bp_overrides={"onset_threshold": 0.6})
    engine = selector.resolve(wav)
    assert engine.predict_params == {**BASICPITCH_TUNED_PRESET, "onset_threshold": 0.6}


# ---------------------------------------------------------------------------
# CLI / パイプライン配線


def test_cli_engine_auto_builds_selector():
    from guitartab.cli import build_engine, main

    class Args:
        basicpitch_python = None
        muscriptor_python = None
        bp_onset_threshold = 0.6
        bp_frame_threshold = None
        bp_minimum_note_length = None
        bp_minimum_frequency = None
        bp_maximum_frequency = None
        bp_no_melodia_trick = False
        ms_instruments = None
        ms_cfg_coef = None
        ms_batch_size = None
        ms_device = None

    selector = build_engine("auto", Args())
    assert isinstance(selector, AutoEngineSelector)
    assert selector.bp_overrides == {"onset_threshold": 0.6}
    assert selector.ms_kwargs == {}

    # eval は auto を受け付けない（ベンチ条件を明示するため）
    with pytest.raises(SystemExit):
        main(["eval", "--engine", "auto"])


def test_pipeline_auto_resolves_after_separate_stage(tmp_path, monkeypatch):
    """auto 配線: notes.json キャッシュ済みなら実推論なしで判定記録だけが残る。"""
    work_root = tmp_path / "work"
    work_dir = work_root / "vid"
    work_dir.mkdir(parents=True)
    source = _distorted_like_wav(work_dir / "source.wav")
    save_notes(
        [NoteEvent(0.1, 0.4, 45), NoteEvent(0.6, 0.9, 50)], work_dir / "notes.json"
    )
    monkeypatch.setattr(
        pipeline, "stage_download", lambda url, root, force=False: source
    )

    notes_path = run_transcribe_pipeline(
        "https://example.test/x",
        AutoEngineSelector(),
        work_root=work_root,
        separate=False,
        quantize=False,
    )
    assert notes_path == work_dir / "notes.json"
    payload = json.loads((work_dir / ENGINE_SELECTION_FILENAME).read_text())
    assert payload["verdict"] == "distorted"
    assert payload["engine"] == "basicpitch"
    assert payload["audio"].endswith("source.wav")
