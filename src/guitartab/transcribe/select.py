"""クリーン/歪みの自動判定によるエンジン自動選択（`--engine auto`）。

エンジン戦略（docs/DESIGN.md「エンジン採用決定（2026-07-18）」、実測で確定）:

- クリーンギター: MuScriptor（dev F1 0.879）
- 歪みエレキ: basic-pitch M1 tuned 構成（合成 highgain F1 0.757）

本モジュールはギターステム音声から歪み度を軽量ヒューリスティックで判定し、
上記の使い分けを自動化する。判定器の設計と 30 クリップ + 実曲ステムでの
実測（30/30 + 実曲正解）は docs/BENCHMARKS.md「エンジン自動選択」節を参照。

判定ルール（2026-07-19 実測に基づく）:

    distorted  ⇔  クレストファクタ < 5.0  or  スペクトル平坦度(500-5000Hz) ≥ 0.40

- クレストファクタ（ピーク / アクティブ区間 RMS）は合成歪みベンチの
  クリーン 10 / crunch 10 / highgain 10 を完全分離する
  （クリーン最小 6.85 vs crunch 最大 3.67。eval_data/distorted_synth/README.md の
  実測 8.7 → 2.4 → 1.4 と整合）。
- ただし実曲の Demucs ステム（work/wr7xTGTG-Mo/stems/guitar.wav、歪みエレキ）は
  無音・ブリード区間とミックス由来のダイナミクスでクレストファクタが 6.30 まで
  戻るため、クレストファクタ単独では「クリーン」に誤判定する。歪みが生む
  倍音間の相互変調成分を捉えるスペクトル平坦度（同ステム 0.79 vs クリーン最大
  0.25）を OR 条件で併用して救済する。
- 既知の限界: 実録音の正例は上記 1 ステムのみで、実クリーンエレキの
  Demucs ステムは未測定。閾値は本 31 クリップ上の in-sample 選択。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from guitartab.transcribe.base import TranscriberEngine
from guitartab.transcribe.basicpitch import BasicPitchEngine
from guitartab.transcribe.muscriptor import (
    DEFAULT_VENV_PYTHON as MUSCRIPTOR_DEFAULT_VENV_PYTHON,
)
from guitartab.transcribe.muscriptor import ENV_VAR as MUSCRIPTOR_ENV_VAR
from guitartab.transcribe.muscriptor import MuScriptorEngine

# ---------------------------------------------------------------------------
# 判定閾値（docs/BENCHMARKS.md「エンジン自動選択」2026-07-19 実測で決定）
# クリーン最小クレスト 6.85 / crunch 最大 3.67 の間、および
# クリーン最大平坦度 0.254 / 実曲歪みステム 0.788 の間に置いた。
CREST_FACTOR_THRESHOLD = 5.0
SPECTRAL_FLATNESS_THRESHOLD = 0.40

# 特徴量計算の定数
FRAME_SIZE = 2048
HOP_SIZE = 512
ACTIVE_RMS_RATIO = 1e-2  # グローバルピーク比 -40dB 以上のフレームを有効区間とする
FLATNESS_BAND_HZ = (500.0, 5000.0)

# auto 選択時の basic-pitch プリセット = M1 tuned 構成
# （onset 0.75 / frame 0.4 / min_note_length 100ms。
# docs/BENCHMARKS.md 2026-07-17 ネイティブパラメータスイープで dev F1 0.864、
# holdout 0.887 で M1 ゲート通過、合成 highgain 0.757 で歪み首位の構成）
BASICPITCH_TUNED_PRESET: dict = {
    "onset_threshold": 0.75,
    "frame_threshold": 0.4,
    "minimum_note_length": 100.0,
}

ENGINE_SELECTION_FILENAME = "engine_selection.json"
ENGINE_SELECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DistortionFeatures:
    """歪み判定に使う音響特徴量（有効区間ベース）。"""

    crest_factor: float  # グローバルピーク / 有効区間 RMS
    spectral_flatness: float  # 500-5000Hz 帯の幾何平均/算術平均（有効区間平均スペクトル）
    hf_ratio_2000hz: float  # 2kHz 以上のパワー比（参考記録用。判定には未使用）
    duration_sec: float


def compute_distortion_features(audio_path: Path) -> DistortionFeatures:
    """音声ファイルから歪み判定特徴量を計算する（本体 venv のみで動作）。"""
    y, sr = sf.read(str(audio_path), dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    duration = len(y) / sr if sr else 0.0
    if len(y) < FRAME_SIZE:
        y = np.pad(y, (0, FRAME_SIZE - len(y)))
    peak = float(np.max(np.abs(y)))
    if peak <= 0.0:
        # 完全無音: 判定不能。distorted 側（= デフォルトエンジン basic-pitch）に倒す。
        return DistortionFeatures(0.0, 1.0, 0.0, duration)

    n_frames = (len(y) - FRAME_SIZE) // HOP_SIZE + 1
    idx = np.arange(FRAME_SIZE)[None, :] + HOP_SIZE * np.arange(n_frames)[:, None]
    frames = y[idx]
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active = rms > peak * ACTIVE_RMS_RATIO
    if not active.any():
        active = rms > 0.0
    active_frames = frames[active]

    crest = peak / (float(np.sqrt(np.mean(active_frames**2))) + 1e-12)

    window = np.hanning(FRAME_SIZE)
    spectra = np.abs(np.fft.rfft(active_frames * window, axis=1))
    power = (spectra**2).mean(axis=0)
    freqs = np.fft.rfftfreq(FRAME_SIZE, 1.0 / sr)
    total = float(power.sum()) + 1e-12
    hf_ratio = float(power[freqs >= 2000.0].sum()) / total
    lo, hi = FLATNESS_BAND_HZ
    band = power[(freqs >= lo) & (freqs <= hi)] + 1e-20
    flatness = float(np.exp(np.mean(np.log(band))) / np.mean(band))
    return DistortionFeatures(
        crest_factor=float(crest),
        spectral_flatness=flatness,
        hf_ratio_2000hz=hf_ratio,
        duration_sec=duration,
    )


def classify_distortion(features: DistortionFeatures) -> str:
    """特徴量から "clean" / "distorted" を返す（モジュール docstring の判定ルール）。"""
    if (
        features.crest_factor < CREST_FACTOR_THRESHOLD
        or features.spectral_flatness >= SPECTRAL_FLATNESS_THRESHOLD
    ):
        return "distorted"
    return "clean"


def muscriptor_unavailable_reason(
    venv_python: Path | str | None = None,
) -> str | None:
    """MuScriptor が使えない理由を返す（使えるなら None）。

    チェックは docs/STATUS の運用前提と同じ 2 点のみ:
    専用 venv の python が存在するか / HF_TOKEN があるか
    （重みはゲート付き HF リポジトリのため。`.env` からの自動読込は
    `python -m guitartab` 起動時に済んでいる前提）。
    """
    resolved = Path(
        venv_python
        or os.environ.get(MUSCRIPTOR_ENV_VAR)
        or MUSCRIPTOR_DEFAULT_VENV_PYTHON
    )
    if not resolved.exists():
        return f"muscriptor venv python not found: {resolved}"
    if not os.environ.get("HF_TOKEN"):
        return "HF_TOKEN is not set (required for gated MuScriptor weights)"
    return None


@dataclass(frozen=True)
class EngineSelection:
    """auto 判定の記録（work/{id}/engine_selection.json に保存する内容）。"""

    audio: str
    verdict: str  # "clean" | "distorted"
    features: DistortionFeatures
    engine: str  # 実際に選択したエンジン名
    engine_params: dict
    fallback_reason: str | None = None

    def to_payload(self) -> dict:
        return {
            "schema": ENGINE_SELECTION_SCHEMA_VERSION,
            "audio": self.audio,
            "verdict": self.verdict,
            "features": asdict(self.features),
            "thresholds": {
                "crest_factor": CREST_FACTOR_THRESHOLD,
                "spectral_flatness": SPECTRAL_FLATNESS_THRESHOLD,
            },
            "engine": self.engine,
            "engine_params": self.engine_params,
            "fallback_reason": self.fallback_reason,
        }


class AutoEngineSelector:
    """`--engine auto` の実体。separate 後のステム音声で判定してエンジンを構築する。

    TranscriberEngine ではない（transcribe を持たない）。パイプラインは
    stage_transcribe の前に resolve() で実エンジンへ差し替えること。

    bp_overrides は BASICPITCH_TUNED_PRESET を上書きする明示指定
    （CLI の --bp-* フラグ由来。プリセットとマージし、明示指定が勝つ）。
    ms_kwargs は MuScriptorEngine へそのまま渡す追加引数。
    """

    name = "auto"

    def __init__(
        self,
        *,
        basicpitch_python: Path | str | None = None,
        muscriptor_python: Path | str | None = None,
        bp_overrides: dict | None = None,
        ms_kwargs: dict | None = None,
    ):
        self.basicpitch_python = basicpitch_python
        self.muscriptor_python = muscriptor_python
        self.bp_overrides = dict(bp_overrides or {})
        self.ms_kwargs = dict(ms_kwargs or {})

    def _build_basicpitch(self) -> BasicPitchEngine:
        params = {**BASICPITCH_TUNED_PRESET, **self.bp_overrides}
        return BasicPitchEngine(venv_python=self.basicpitch_python, **params)

    def resolve(
        self,
        audio_path: Path,
        *,
        selection_path: Path | None = None,
    ) -> TranscriberEngine:
        """音声を判定してエンジンを返す。selection_path があれば判定記録を保存する。"""
        audio_path = Path(audio_path)
        features = compute_distortion_features(audio_path)
        verdict = classify_distortion(features)

        fallback_reason: str | None = None
        if verdict == "clean":
            fallback_reason = muscriptor_unavailable_reason(self.muscriptor_python)
            if fallback_reason is None:
                engine: TranscriberEngine = MuScriptorEngine(
                    venv_python=self.muscriptor_python, **self.ms_kwargs
                )
            else:
                print(
                    f"warning: engine auto: verdict=clean but MuScriptor is "
                    f"unavailable ({fallback_reason}); falling back to "
                    "basic-pitch (M1 tuned preset). See README "
                    "「MuScriptor の運用」 for setup.",
                    file=sys.stderr,
                )
                engine = self._build_basicpitch()
        else:
            engine = self._build_basicpitch()

        engine_params = (
            dict(engine.predict_params)
            if isinstance(engine, BasicPitchEngine)
            else {
                "instruments": engine.instruments,
                "cfg_coef": engine.cfg_coef,
                "batch_size": engine.batch_size,
                "device": engine.device,
            }
        )
        selection = EngineSelection(
            audio=str(audio_path),
            verdict=verdict,
            features=features,
            engine=engine.name,
            engine_params=engine_params,
            fallback_reason=fallback_reason,
        )
        print(
            f"engine auto: crest={features.crest_factor:.2f} "
            f"flatness={features.spectral_flatness:.3f} -> {verdict} -> "
            f"engine={engine.name}",
            file=sys.stderr,
        )
        if selection_path is not None:
            selection_path = Path(selection_path)
            selection_path.parent.mkdir(parents=True, exist_ok=True)
            selection_path.write_text(
                json.dumps(selection.to_payload(), indent=1, ensure_ascii=False)
            )
            print(f"wrote {selection_path}", file=sys.stderr)
        return engine
