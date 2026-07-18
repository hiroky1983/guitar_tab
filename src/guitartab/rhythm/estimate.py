"""一定テンポ + 位相の推定（M4a 候補 A: librosa + ノート onset 統合）。

docs/DESIGN_M4_QUANTIZATION.md §2 の設計を dev 10 トラックでの実測反復で
具体化したもの（構成要素ごとの実測は docs/BENCHMARKS.md「M4a」節）:

1. 候補生成: librosa.beat.beat_track のテンポ T × 有理数係数族
   {1/3,1/2,2/3,3/4,4/5,1,5/4,4/3,3/2,5/3,2,3} に、onset 包絡の自己相関
   上位ピーク × {1/2,1,2} を加える（librosa T が 4/3・5/4 等の非オクターブ
   関係に落ちるトラックが dev 実測で 4/10 あり、係数族だけでは正解族を
   候補に入れられないため）。
2. 各候補を ±8% の格子適合走査でリファインし、さらに「最近傍格子割当 →
   (位相, 周期) の最小二乗」の反復（polish）でテンポを高精度化する
   （クリック録音で ~0.1% 精度。1% の誤差でも 30 秒でビートが 70ms 超
   ドリフトして Beat F を毀損する）。
3. 位相の 2 段決定: 16 分格子は 1/4 拍シフト不変（剰余集合 {0,3,6,9} が
   3 tick シフトで自己一致）のため、ノートからは位相が mod P/4 でしか
   決まらない。拍ラベル k∈{0..3} は「テンポ固定の librosa DP ビート追跡
   （tightness=400）の circular median 位相」を候補格子へスナップして選ぶ。
4. 候補選択: score = ex + w_dp·dp_acc − w_anchor·anchor。
   - ex: 16 分+3 連格子への onset 適合（Gauss σ=25ms、一様チャンス補正）
   - dp_acc: 選ばれた拍位置での onset 包絡アクセント（±50ms 窓、正規化）
   - anchor: |log2(bpm / librosa T)|（音声トラッカーのレベルからの乖離）

音声なし（onsets のみ）のフォールバックは BPM 幾何走査 + ex のみ
（設計 §2.1 の IOI 格子投票相当の補助経路。精度は落ちる）。

音声トラッカー信頼モード（trust_tracker=True、M4b）: 実曲ミックス入力では
生 beat_track がテンポ族を正しく当てるのに、上記候補選択層（ノート格子適合
スコア支配）がテンポレベルを上書きして族外へ落とす実測がある
（docs/BENCHMARKS.md「M4b — ミックス経路 実曲検証」: 生 117.5 → 選択層 108.0）。
このモードではテンポ = 生 beat_track 拍列の頑健 LS フィット（トラッカーの
テンポから 6% 超ずれたらトラッカーのテンポへフォールバック）とし、
ノート格子適合は位相の決定のみに使う —— 候補族の生成・選択は行わず、
ノートがテンポを動かすことを許さない（レベル固定 ±8% の走査でも実曲では
格子適合が走査端の 108.0 まで滑る実測があった）。pipeline の
rhythm_source="mix" 時に自動でこのモードになる（stem 経路は従来どおり）。

Beat This!（候補 B）は TempoEstimator Protocol を実装すれば差し替え可能
（M4a では未統合）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from guitartab.rhythm.schema import ALLOWED_TICK_RESIDUES, DIVISIONS_PER_QUARTER

BPM_MIN, BPM_MAX = 40.0, 240.0

# librosa T に掛ける有理数係数族（オクターブ族 + dev 実測で必要になった中間比）
DEFAULT_CANDIDATE_FACTORS = (
    1 / 3, 1 / 2, 2 / 3, 3 / 4, 4 / 5, 1.0, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 2.0, 3.0
)
# 自己相関ピークに掛ける係数
AC_PEAK_FACTORS = (0.5, 1.0, 2.0)

DEFAULT_NOTE_TOLERANCE_SEC = 0.025  # 格子適合のガウス幅
DEFAULT_DP_WEIGHT = 0.5            # dp_acc の重み
DEFAULT_ANCHOR_WEIGHT = 0.3        # anchor の重み

_SR = 22050
_HOP = 512

# 拍内の許容格子位置（拍を 1 とした比率）: 16 分 + 3 連
_GRID_FRACTIONS = np.array(
    sorted(r / DIVISIONS_PER_QUARTER for r in ALLOWED_TICK_RESIDUES)
)


@dataclass(frozen=True)
class TempoEstimate:
    """一定テンポ推定の結果。

    grid_origin_sec はある拍の物理時刻（格子の位相、[0, 拍周期) に正規化）。
    格子は grid_origin_sec + k × (60/bpm) に拍、拍内に 16 分 + 3 連の細分。
    """

    bpm: float
    grid_origin_sec: float
    estimator: str
    params: dict = field(default_factory=dict)

    @property
    def beat_period_sec(self) -> float:
        return 60.0 / self.bpm


@runtime_checkable
class TempoEstimator(Protocol):
    """テンポ・位相推定の差し替え可能インターフェース。

    onsets_sec = 転写ノートの onset 列（必須）、audio_path = 拍推定に使える
    音声（任意）。実装は音声なしでも動作しなければならない。
    """

    name: str

    def estimate(
        self, onsets_sec: Sequence[float], *, audio_path: Path | None = None
    ) -> TempoEstimate: ...


# ---------------------------------------------------------------------------
# 格子適合ユーティリティ
# ---------------------------------------------------------------------------


def _wrapped_distance(values: np.ndarray, period: float) -> np.ndarray:
    """周期 period の円周上での 0 からの距離（最大 period/2）。"""
    return np.abs(((values + period / 2.0) % period) - period / 2.0)


def _chance_fitness(period: float, sigma: float) -> float:
    """一様分布の onset が格子適合度に偶然寄与する期待値（数値積分）。"""
    t = np.linspace(0.0, period, 512, endpoint=False)
    diff = t[:, None] - _GRID_FRACTIONS[None, :] * period
    d = _wrapped_distance(diff, period).min(axis=1)
    return float(np.mean(np.exp(-((d / sigma) ** 2))))


def _fine_fit(
    onsets: np.ndarray, bpm: float, sigma: float
) -> tuple[float, float]:
    """(チャンス補正済み格子適合 ex, 位相 φ16) を返す。

    16 分格子は 1/4 拍シフト不変のため、位相探索は [0, P/4) で足りる。
    """
    period = 60.0 / bpm
    n_phase = max(24, int(np.ceil(period / 4 / 0.002)))
    phases = np.linspace(0.0, period / 4, n_phase, endpoint=False)
    offsets = _GRID_FRACTIONS * period
    diff = onsets[None, :, None] - phases[:, None, None] - offsets[None, None, :]
    d = _wrapped_distance(diff, period).min(axis=2)
    fit = np.mean(np.exp(-((d / sigma) ** 2)), axis=1)
    j = int(np.argmax(fit))
    return float(fit[j]) - _chance_fitness(period, sigma), float(phases[j])


_SNAP_BEAT_FRACS = np.concatenate([_GRID_FRACTIONS, [1.0]])


def _polish(
    onsets: np.ndarray, bpm: float, phase: float, *, iters: int = 3
) -> tuple[float, float]:
    """最近傍格子割当 → (位相, 周期) の最小二乗を反復してテンポを高精度化。

    格子セル半分以内のノートだけで回帰し、off-grid ノートの引きずりを避ける。
    """
    period = 60.0 / bpm
    for _ in range(iters):
        b = (onsets - phase) / period
        k = np.floor(b)
        frac = b - k
        idx = np.argmin(np.abs(frac[:, None] - _SNAP_BEAT_FRACS[None, :]), axis=1)
        g = k + _SNAP_BEAT_FRACS[idx]  # 拍単位の格子位置
        keep = np.abs(b - g) < (1 / DIVISIONS_PER_QUARTER) / 2
        if keep.sum() < 8:
            break
        a_mat = np.stack([np.ones(int(keep.sum())), g[keep]], axis=1)
        sol, *_ = np.linalg.lstsq(a_mat, onsets[keep], rcond=None)
        new_phase, new_period = float(sol[0]), float(sol[1])
        if not 60.0 / BPM_MAX < new_period < 60.0 / BPM_MIN:
            break
        phase, period = new_phase, new_period
    return 60.0 / period, phase


def _fit_beats_bpm(beats_sec: np.ndarray) -> float | None:
    """トラッカー拍列 t = phase + k·period の頑健 LS フィットで BPM を返す。

    拍インデックス k は median IOI で丸めて推定（拍の欠落 = k の飛びに頑健）。
    拍 2 個未満・非正周期・BPM 範囲外は None（呼び出し側でフォールバック）。
    音声トラッカー信頼モード用: 拍列全体で回帰するためテンポグラムのビン
    量子化より高精度で、一定テンポ描画時のドリフトを抑える。
    """
    t = np.sort(beats_sec[np.isfinite(beats_sec)])
    if len(t) < 2:
        return None
    period0 = float(np.median(np.diff(t)))
    if period0 <= 0:
        return None
    k = np.round((t - t[0]) / period0)
    a_mat = np.stack([np.ones(len(t)), k], axis=1)
    sol, *_ = np.linalg.lstsq(a_mat, t, rcond=None)
    period = float(sol[1])
    if period <= 0:
        return None
    bpm = 60.0 / period
    if not BPM_MIN <= bpm <= BPM_MAX:
        return None
    return bpm


def _ac_peaks(env: np.ndarray, *, top: int = 4) -> list[float]:
    """onset 包絡の自己相関からテンポ候補ピーク（BPM）を返す。"""
    import librosa

    ac = librosa.autocorrelate(env, max_size=int(4 * _SR / _HOP))
    lags = np.arange(1, len(ac))
    freqs = 60.0 * _SR / _HOP / lags
    mask = (freqs >= BPM_MIN) & (freqs <= BPM_MAX)
    vals = ac[1:]
    peaks = []
    for i in lags[mask][1:-1]:
        if vals[i - 1] > vals[i - 2] and vals[i - 1] > vals[i]:
            peaks.append((float(vals[i - 1]), float(freqs[i - 1])))
    peaks.sort(reverse=True)
    return [f for _, f in peaks[:top]]


class LibrosaConstantTempoEstimator:
    """候補 A: librosa + ノート onset 統合（モジュール docstring 参照）。"""

    name = "librosa_constant"

    def __init__(
        self,
        *,
        candidate_factors: Sequence[float] = DEFAULT_CANDIDATE_FACTORS,
        note_tolerance_sec: float = DEFAULT_NOTE_TOLERANCE_SEC,
        dp_weight: float = DEFAULT_DP_WEIGHT,
        anchor_weight: float = DEFAULT_ANCHOR_WEIGHT,
        refine_span: float = 0.08,
        refine_steps: int = 41,
        trust_tracker: bool = False,
    ) -> None:
        self.candidate_factors = tuple(candidate_factors)
        self.note_tolerance_sec = note_tolerance_sec
        self.dp_weight = dp_weight
        self.anchor_weight = anchor_weight
        self.refine_span = refine_span
        self.refine_steps = refine_steps
        self.trust_tracker = trust_tracker

    def _params(self, audio_used: bool) -> dict:
        return {
            "candidate_factors": list(self.candidate_factors),
            "note_tolerance_sec": self.note_tolerance_sec,
            "dp_weight": self.dp_weight,
            "anchor_weight": self.anchor_weight,
            "refine_span": self.refine_span,
            "audio_used": audio_used,
            "trust_tracker": self.trust_tracker,
        }

    def _refine(self, onsets: np.ndarray, bpm0: float) -> tuple[float, float, float]:
        """(ex, bpm, φ16) を返す: ±span 走査 → polish → 再評価。"""
        best = (-np.inf, bpm0, 0.0)
        for bpm in bpm0 * np.linspace(
            1 - self.refine_span, 1 + self.refine_span, self.refine_steps
        ):
            ex, phi = _fine_fit(onsets, bpm, self.note_tolerance_sec)
            if ex > best[0]:
                best = (ex, bpm, phi)
        _, bpm, phi = best
        bpm, phi = _polish(onsets, bpm, phi)
        if not BPM_MIN <= bpm <= BPM_MAX:
            bpm = best[1]
        ex, phi = _fine_fit(onsets, bpm, self.note_tolerance_sec)
        return ex, bpm, phi

    def _dp_phase_and_accent(
        self, env: np.ndarray, env_times: np.ndarray, duration: float,
        bpm: float, phi16: float,
    ) -> tuple[float, float]:
        """テンポ固定 DP で拍ラベル k を決め、(位相, アクセント量) を返す。"""
        import librosa

        period = 60.0 / bpm
        _tempo, dp_beats = librosa.beat.beat_track(
            onset_envelope=env, sr=_SR, hop_length=_HOP,
            bpm=bpm, tightness=400, units="time",
        )
        dp_beats = np.asarray(dp_beats, dtype=float)
        if len(dp_beats) >= 2:
            angles = 2j * np.pi * (dp_beats % period) / period
            ph_dp = float(
                np.angle(np.mean(np.exp(angles))) / (2 * np.pi) * period % period
            )
            k = int(np.round((ph_dp - phi16) / (period / 4))) % 4
        else:
            k = 0
        phase = (phi16 + k * period / 4) % period
        beats = np.arange(phase, duration, period)
        if len(beats) == 0:
            return phase, 0.0
        acc = 0.0
        for b in beats:
            i0 = int(np.searchsorted(env_times, b - 0.05))
            i1 = int(np.searchsorted(env_times, b + 0.05))
            acc += float(env[i0 : i1 + 1].sum())
        dp_acc = acc / len(beats) / (float(env.mean()) + 1e-9) / 5.0
        return phase, dp_acc

    def estimate(
        self, onsets_sec: Sequence[float], *, audio_path: Path | None = None
    ) -> TempoEstimate:
        onsets = np.asarray(sorted(float(t) for t in onsets_sec))
        params = self._params(audio_path is not None)

        if audio_path is None:
            return self._estimate_notes_only(onsets, params)

        import librosa

        y, _sr = librosa.load(audio_path, sr=_SR, mono=True)
        env = librosa.onset.onset_strength(y=y, sr=_SR, hop_length=_HOP)
        env_times = librosa.times_like(env, sr=_SR, hop_length=_HOP)
        duration = len(y) / _SR
        tempo_t, _beats = librosa.beat.beat_track(
            onset_envelope=env, sr=_SR, hop_length=_HOP, units="time"
        )
        tempo_t = float(np.atleast_1d(tempo_t)[0])

        if len(onsets) < 8:
            # ノートが少なすぎて格子適合が立たない: librosa の結果をそのまま
            period = 60.0 / tempo_t
            phase = float(_beats[0] % period) if len(np.atleast_1d(_beats)) else 0.0
            return TempoEstimate(tempo_t, phase, self.name, params)

        if self.trust_tracker:
            # 音声トラッカー信頼モード（モジュール docstring 参照）:
            # テンポ = 生 beat_track の拍列の頑健 LS フィット（トラッカーの
            # テンポから 6% 超ずれたらトラッカーのテンポへフォールバック）。
            # ノート格子適合は位相 phi16 の決定のみに使い、テンポは動かさない
            # （±8% のレベル固定走査でも実曲では格子適合が走査端まで滑って
            # 108.0 へ退行する実測があるため — docs/BENCHMARKS.md M4b 節）。
            bpm = _fit_beats_bpm(np.atleast_1d(np.asarray(_beats, dtype=float)))
            if bpm is None or abs(float(np.log2(bpm / tempo_t))) > np.log2(1.06):
                bpm = tempo_t
            _ex, phi16 = _fine_fit(onsets, bpm, self.note_tolerance_sec)
            phase, _dp_acc = self._dp_phase_and_accent(
                env, env_times, duration, bpm, phi16
            )
            return TempoEstimate(bpm, phase % (60.0 / bpm), self.name, params)

        seeds = [tempo_t * f for f in self.candidate_factors]
        seeds += [p * f for p in _ac_peaks(env) for f in AC_PEAK_FACTORS]

        candidates: list[tuple[float, float, float]] = []  # (score, bpm, phase)
        seen: list[float] = []
        for bpm0 in seeds:
            if not BPM_MIN <= bpm0 <= BPM_MAX:
                continue
            ex, bpm, phi16 = self._refine(onsets, bpm0)
            if any(abs(np.log2(bpm / s)) < 0.014 for s in seen):
                continue
            seen.append(bpm)
            phase, dp_acc = self._dp_phase_and_accent(
                env, env_times, duration, bpm, phi16
            )
            anchor = abs(float(np.log2(bpm / tempo_t)))
            score = ex + self.dp_weight * dp_acc - self.anchor_weight * anchor
            candidates.append((score, bpm, phase))

        if not candidates:
            return TempoEstimate(120.0, 0.0, self.name, params)
        _, bpm, phase = max(candidates, key=lambda c: c[0])
        return TempoEstimate(bpm, phase % (60.0 / bpm), self.name, params)

    def _estimate_notes_only(
        self, onsets: np.ndarray, params: dict
    ) -> TempoEstimate:
        """音声なしフォールバック: BPM 幾何走査 + 格子適合のみ。"""
        if len(onsets) < 2:
            return TempoEstimate(120.0, 0.0, self.name, params)
        # 粗い走査はテンポ誤差のドリフトで真のテンポの適合が下がり、走査格子上に
        # ぴったり乗るエイリアスに負け得る。上位候補を polish 後に再評価して選ぶ。
        scanned = []
        for bpm0 in np.geomspace(BPM_MIN * 1.02, BPM_MAX * 0.98, 60):
            ex, phi = _fine_fit(onsets, float(bpm0), self.note_tolerance_sec)
            scanned.append((ex, float(bpm0), phi))
        scanned.sort(reverse=True)
        best = (-np.inf, 120.0, 0.0)
        for _, bpm0, phi0 in scanned[:5]:
            bpm, phi = _polish(onsets, bpm0, phi0)
            if not BPM_MIN <= bpm <= BPM_MAX:
                bpm = bpm0
            ex, phi = _fine_fit(onsets, bpm, self.note_tolerance_sec)
            if ex > best[0]:
                best = (ex, bpm, phi)
        _, bpm, phi = best
        return TempoEstimate(bpm, phi % (60.0 / bpm), self.name, params)


__all__ = [
    "AC_PEAK_FACTORS",
    "BPM_MAX",
    "BPM_MIN",
    "DEFAULT_ANCHOR_WEIGHT",
    "DEFAULT_CANDIDATE_FACTORS",
    "DEFAULT_DP_WEIGHT",
    "DEFAULT_NOTE_TOLERANCE_SEC",
    "LibrosaConstantTempoEstimator",
    "TempoEstimate",
    "TempoEstimator",
]
