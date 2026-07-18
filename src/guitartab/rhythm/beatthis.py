"""Beat This! による一定テンポ + 位相の推定（M4b 候補 B、サブプロセス実行方式）。

Beat This!（CPJKU、ISMIR 2024、MIT）は Transformer の beat/downbeat トラッカー。
torch 依存のため本体 venv には入れず、専用 venv の python で
_beatthis_runner.py を実行して拍列・ダウンビート列を受け取る
（transcribe/{basicpitch,yourmt3,muscriptor}.py と同じ疎結合方式）。

拍列からの一定テンポ + 位相の決定（M4a の quantize / rhythm.json は不変更）:

1. **テンポレベル（オクターブ）**: Beat This! 拍列の median IOI で決める
   （学習モデルの判断に委ねる。M4a 候補 A の失敗が「レベルとラベルの選択」に
   集中していたため — docs/BENCHMARKS.md M4a 節の申し送り 3）。
2. **精密テンポ + 位相 mod P/4**: M4a の格子適合走査 + 回帰ポリッシュ
   （LibrosaConstantTempoEstimator._refine）をレベル固定（±8%）で流用する。
   dev 実測で Beat This! の生の拍列は一定テンポフィットに耐えない
   （冒頭のスプリアス拍・欠落があり、素の最小二乗ではテンポに 0.2% 級の
   誤差が残って 30 秒でビートが 70ms 超ドリフトする。Rock2 で raw beatF
   0.757 → 素 LS フィット 0.372 に劣化する実測）。テンポ正解時の格子精度は
   ポリッシュが実績値（beatF 0.97〜0.99、M4a 申し送り 4）。
3. **拍ラベル k ∈ {0..3}**: 16 分格子は 1/4 拍シフト不変で位相はノートから
   mod P/4 でしか決まらないため、Beat This! 拍列の circular mean 位相を
   候補格子へスナップして選ぶ（M4a の librosa DP 拍を Beat This! に差し替え）。

音声なし（onsets のみ）のときは Beat This! は使えないため、
LibrosaConstantTempoEstimator のノートのみフォールバック経路に委譲する。

venv パスの指定方法（優先順）:

1. コンストラクタ引数 venv_python
2. 環境変数 GUITARTAB_BEATTHIS_PYTHON
3. プロジェクト直下の .venv-beatthis/bin/python（デフォルト）

デバイスは device 引数 > 環境変数 GUITARTAB_BEATTHIS_DEVICE > "cpu"
（30 秒クリップの実測で CPU 1.3s / MPS 3.4s と CPU が速い。モデルロード込みの
サブプロセス 1 回は約 5 秒）。チェックポイント final0（78MB）は初回に
JKU クラウド（直接 HTTPS）から torch.hub キャッシュへ自動 DL される。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from guitartab.rhythm.estimate import (
    BPM_MAX,
    BPM_MIN,
    LibrosaConstantTempoEstimator,
    TempoEstimate,
)

ENV_VAR = "GUITARTAB_BEATTHIS_PYTHON"
DEFAULT_VENV_PYTHON = Path(".venv-beatthis") / "bin" / "python"

DEVICE_ENV_VAR = "GUITARTAB_BEATTHIS_DEVICE"
DEFAULT_DEVICE = "cpu"

DEFAULT_CHECKPOINT = "final0"

_RUNNER = Path(__file__).parent / "_beatthis_runner.py"

_SETUP_HINT = (
    "beatthis venv python not found: {python}\n"
    "Set up a dedicated venv (`uv venv --python 3.11 .venv-beatthis && "
    "uv pip install --python .venv-beatthis/bin/python beat-this mir_eval soundfile`), "
    "then pass venv_python= or set " + ENV_VAR + "."
)


def beats_to_bpm(beats_sec) -> float | None:
    """拍列の median IOI からテンポレベル（BPM）を決める。

    範囲外は 2 倍 / 半分に折り畳む。拍 2 個未満・非正の周期は None
    （呼び出し側でフォールバック）。
    """
    t = np.asarray(sorted(float(b) for b in beats_sec))
    if len(t) < 2:
        return None
    period0 = float(np.median(np.diff(t)))
    if period0 <= 0:
        return None
    bpm = 60.0 / period0
    while bpm < BPM_MIN:
        bpm *= 2.0
    while bpm > BPM_MAX:
        bpm /= 2.0
    return bpm


def fit_constant_tempo(beats_sec) -> tuple[float, float] | None:
    """拍列を t = phase + k * period の最小二乗でフィットして (bpm, phase) を返す。

    拍インデックス k は median IOI で丸めて推定する（拍の欠落 = k の飛びに頑健。
    ただしスプリアス拍には弱い — ノート onset が使えるときは _refine 経路が優先）。
    拍が 2 個未満、または結果が BPM 範囲外なら None（呼び出し側でフォールバック）。
    """
    t = np.asarray(sorted(float(b) for b in beats_sec))
    if len(t) < 2:
        return None
    ioi = np.diff(t)
    period0 = float(np.median(ioi))
    if period0 <= 0:
        return None
    k = np.round((t - t[0]) / period0)
    a_mat = np.stack([np.ones(len(t)), k], axis=1)
    sol, *_ = np.linalg.lstsq(a_mat, t, rcond=None)
    phase, period = float(sol[0]), float(sol[1])
    if period <= 0:
        return None
    bpm = 60.0 / period
    if not BPM_MIN <= bpm <= BPM_MAX:
        return None
    return bpm, phase


def snap_beat_label(
    beats_sec, bpm: float, phi16: float
) -> float:
    """Beat This! 拍列の circular mean 位相を phi16 + k·P/4 (k∈{0..3}) へスナップ。

    戻り値は拍の位相（[0, P) に正規化）。拍が空なら phi16 をそのまま返す。
    """
    period = 60.0 / bpm
    t = np.asarray([float(b) for b in beats_sec])
    if len(t) == 0:
        return phi16 % period
    angles = 2j * np.pi * (t % period) / period
    ph_bt = float(np.angle(np.mean(np.exp(angles))) / (2 * np.pi) * period % period)
    k = int(np.round((ph_bt - phi16) / (period / 4))) % 4
    return (phi16 + k * period / 4) % period


class BeatThisTempoEstimator:
    """候補 B: Beat This! の拍列を一定テンポ + 位相へフィット（モジュール docstring 参照）。"""

    name = "beatthis_constant"

    def __init__(
        self,
        venv_python: Path | str | None = None,
        *,
        checkpoint: str = DEFAULT_CHECKPOINT,
        device: str | None = None,
        dbn: bool = False,
        trust_tracker: bool = False,
    ) -> None:
        resolved = venv_python or os.environ.get(ENV_VAR) or DEFAULT_VENV_PYTHON
        self.venv_python = Path(resolved)
        self.checkpoint = checkpoint
        self.device = device or os.environ.get(DEVICE_ENV_VAR) or DEFAULT_DEVICE
        self.dbn = dbn
        # 音声トラッカー信頼モード（M4b）。Beat This! 経路は構造上すでに
        # トラッカー信頼（テンポレベル = median IOI、ノート格子はレベル固定
        # ±8% の精密化のみ）なので主経路の挙動は変わらず、拍が取れなかった
        # ときの librosa フォールバックにのみ伝播する。
        self.trust_tracker = trust_tracker
        # レベル固定の精密化（_refine）とノートのみフォールバックに流用
        self._librosa = LibrosaConstantTempoEstimator(trust_tracker=trust_tracker)

    def _params(self, *, fallback: str | None) -> dict:
        params: dict = {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "dbn": self.dbn,
            "trust_tracker": self.trust_tracker,
        }
        if fallback is not None:
            params["fallback"] = fallback
        return params

    def track_beats(self, audio_path: Path) -> tuple[list[float], list[float]]:
        """サブプロセスで Beat This! を実行し (beats_sec, downbeats_sec) を返す。"""
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        if not self.venv_python.exists():
            raise RuntimeError(_SETUP_HINT.format(python=self.venv_python))

        params = {
            "checkpoint": self.checkpoint,
            "device": self.device,
            "dbn": self.dbn,
        }
        with tempfile.TemporaryDirectory(prefix="guitartab-beatthis-") as tmp:
            out_json = Path(tmp) / "beats.json"
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
                    f"beatthis runner failed (exit {proc.returncode}): "
                    f"{' '.join(cmd)}\n{proc.stderr}"
                )
            payload = json.loads(out_json.read_text())
        return payload["beats_sec"], payload["downbeats_sec"]

    def estimate(
        self, onsets_sec, *, audio_path: Path | None = None
    ) -> TempoEstimate:
        onsets = np.asarray(sorted(float(t) for t in onsets_sec))

        if audio_path is None:
            # Beat This! は音声必須。ノートのみは librosa 版のフォールバック経路へ委譲
            fb = self._librosa.estimate(onsets, audio_path=None)
            return TempoEstimate(
                fb.bpm,
                fb.grid_origin_sec,
                self.name,
                self._params(fallback="librosa_notes_only"),
            )

        beats, _downbeats = self.track_beats(audio_path)
        bpm0 = beats_to_bpm(beats)
        if bpm0 is None:
            # 拍が取れなかった: librosa 版（音声あり）へ全面フォールバック
            fb = self._librosa.estimate(onsets, audio_path=audio_path)
            return TempoEstimate(
                fb.bpm,
                fb.grid_origin_sec,
                self.name,
                self._params(fallback="librosa_audio"),
            )

        if len(onsets) < 8:
            # 格子適合が立たない: 拍列だけの素の最小二乗フィット
            fit = fit_constant_tempo(beats)
            bpm, phase = fit if fit is not None else (bpm0, float(beats[0]))
            return TempoEstimate(
                bpm,
                phase % (60.0 / bpm),
                self.name,
                self._params(fallback="beats_ls_fit"),
            )

        # レベル固定（±8%）の格子適合走査 + 回帰ポリッシュで精密テンポと phi16 を決め、
        # 拍ラベル k は Beat This! 拍列からスナップする（モジュール docstring 参照）
        _ex, bpm, phi16 = self._librosa._refine(onsets, bpm0)
        phase = snap_beat_label(beats, bpm, phi16)
        return TempoEstimate(
            bpm, phase % (60.0 / bpm), self.name, self._params(fallback=None)
        )


__all__ = [
    "BeatThisTempoEstimator",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DEVICE",
    "DEFAULT_VENV_PYTHON",
    "DEVICE_ENV_VAR",
    "ENV_VAR",
    "beats_to_bpm",
    "fit_constant_tempo",
    "snap_beat_label",
]
