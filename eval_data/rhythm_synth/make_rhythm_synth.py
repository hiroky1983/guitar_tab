"""GuitarSet dev 10 トラックから合成リズムベンチ eval_data/rhythm_synth/ を生成する。

設計: docs/DESIGN_M4_QUANTIZATION.md §4.3。distorted_synth と同じ流儀
（生成スクリプト + manifest 同梱、ソースは読み取りのみ）。

手順:
1. 参照スコア: dev の note_midi onset を GT テンポ格子（16 分 + 3 連、
   quarter=12 tick、beat1 = t0 なので位相 0）へ機械的にスナップし、その結果を
   「真のスコア」と**定義**する（スナップの当否は問題にならない — 音源を
   このスコアから再合成するため、構成上 per-note の tick GT が厳密になる）。
2. 音源: 真のスコアを既知テンポで物理時刻へ展開し、Karplus-Strong 合成で
   レンダリングする（fluidsynth 未導入環境のための設計書記載の代替。
   レンダラとパラメータは manifest に記録）。
3. 変種（M4a: 一定テンポ + ジッタ）:
   - clean:    GT テンポそのまま
   - slow08:   テンポ 0.8×
   - fast125:  テンポ 1.25×
   - jitter10: GT テンポ + onset に σ=10ms のガウスジッタ（±3σ クリップ）
   - jitter20: 同 σ=20ms

各クリップ: eval_data/rhythm_synth/<variant>/<track_id>/{audio.wav, score.json}。
score.json の notes は (onset_sec, midi_pitch) 順で、onset_tick が per-note GT。

ソース（eval_data/guitarset/）へは一切書き込まない。
holdout は使わない（ゲート判定専用のため）。

実行: uv run python eval_data/rhythm_synth/make_rhythm_synth.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from guitartab.eval.loaders import load_jams_note_midi, load_jams_tempo
from guitartab.rhythm.schema import ALLOWED_TICK_RESIDUES, DIVISIONS_PER_QUARTER

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "eval_data" / "guitarset"
DST = REPO / "eval_data" / "rhythm_synth"

SR = 22050
TARGET_PEAK = 10 ** (-1.0 / 20)  # -1 dBFS
KS_DECAY = 0.996  # Karplus-Strong 弦減衰係数
NOTE_GAIN = 0.5
FADE_SEC = 0.01
SCORE_SCHEMA_VERSION = 1

VARIANTS = {
    # name: (tempo_factor, jitter_sigma_ms)
    "clean": (1.0, 0.0),
    "slow08": (0.8, 0.0),
    "fast125": (1.25, 0.0),
    "jitter10": (1.0, 10.0),
    "jitter20": (1.0, 20.0),
}

_SNAP_TICKS = sorted(ALLOWED_TICK_RESIDUES) + [DIVISIONS_PER_QUARTER]


def snap_tick(onset_sec: float, bpm: float) -> int:
    """位相 0・一定テンポの 16 分+3 連格子の最近傍 tick（タイは小さい側）。"""
    beats = onset_sec * bpm / 60.0
    beat_index = int(beats // 1)
    frac_ticks = (beats - beat_index) * DIVISIONS_PER_QUARTER
    best = min(_SNAP_TICKS, key=lambda r: (abs(frac_ticks - r), r))
    return beat_index * DIVISIONS_PER_QUARTER + best


def karplus_strong(freq_hz: float, dur_sec: float, rng: np.random.Generator) -> np.ndarray:
    """Karplus-Strong 撥弦合成（コムフィルタを lfilter で駆動）。"""
    n = max(1, int(round(dur_sec * SR)))
    delay = max(2, int(round(SR / freq_hz)))
    x = np.zeros(n)
    burst = rng.uniform(-1.0, 1.0, min(delay, n))
    x[: len(burst)] = burst
    a = np.zeros(delay + 2)
    a[0] = 1.0
    a[delay] = -KS_DECAY * 0.5
    a[delay + 1] = -KS_DECAY * 0.5
    y = lfilter([1.0], a, x)
    fade = min(n, max(1, int(FADE_SEC * SR)))
    y[-fade:] *= np.linspace(1.0, 0.0, fade)
    return y


def stable_seed(key: str) -> int:
    """プロセス間で安定な 32bit シード（組み込み hash は PYTHONHASHSEED 依存）。"""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def render_track(score_notes: list[dict], seed: int) -> np.ndarray:
    """スコア（物理時刻展開済み）をモノラル波形にミックスする。"""
    total = max(n["onset_sec"] + n["duration_sec"] for n in score_notes) + 0.5
    buf = np.zeros(int(total * SR) + 1)
    for i, n in enumerate(score_notes):
        rng = np.random.default_rng([seed, i])
        freq = 440.0 * 2 ** ((n["midi_pitch"] - 69) / 12)
        y = karplus_strong(freq, n["duration_sec"], rng) * NOTE_GAIN
        start = int(round(n["onset_sec"] * SR))
        buf[start : start + len(y)] += y
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf *= TARGET_PEAK / peak
    return buf


def main() -> None:
    manifest_src = json.loads((SRC / "manifest.json").read_text())
    tracks_meta = []
    for t in sorted(manifest_src["tracks"], key=lambda t: t["track_id"]):
        tid = t["track_id"]
        jams = REPO / t["annotation"]
        gt_tempo = load_jams_tempo(jams)
        notes = load_jams_note_midi(jams)  # onset, pitch ソート済み

        # 1. 真のスコア（GT テンポ格子スナップ。全変種で共通）
        tick_sec_gt = 60.0 / gt_tempo / DIVISIONS_PER_QUARTER
        score = [
            {
                "onset_tick": snap_tick(n.onset_sec, gt_tempo),
                "duration_ticks": max(
                    1, round((n.offset_sec - n.onset_sec) / tick_sec_gt)
                ),
                "midi_pitch": n.midi_pitch,
            }
            for n in notes
        ]

        for variant, (factor, sigma_ms) in VARIANTS.items():
            bpm = gt_tempo * factor
            tick_sec = 60.0 / bpm / DIVISIONS_PER_QUARTER
            seed = stable_seed(f"{tid}/{variant}")
            rng = np.random.default_rng(seed)
            rendered = []
            for s in score:
                jitter = 0.0
                if sigma_ms > 0:
                    jitter = float(
                        np.clip(rng.normal(0.0, sigma_ms / 1000), -3 * sigma_ms / 1000, 3 * sigma_ms / 1000)
                    )
                onset = max(0.0, s["onset_tick"] * tick_sec + jitter)
                rendered.append(
                    {
                        **s,
                        "onset_sec": round(onset, 6),
                        "duration_sec": round(s["duration_ticks"] * tick_sec, 6),
                    }
                )
            rendered.sort(key=lambda n: (n["onset_sec"], n["midi_pitch"]))

            out_dir = DST / variant / tid
            out_dir.mkdir(parents=True, exist_ok=True)
            audio = render_track(rendered, seed)
            sf.write(out_dir / "audio.wav", audio, SR, subtype="PCM_16")
            (out_dir / "score.json").write_text(
                json.dumps(
                    {
                        "schema_version": SCORE_SCHEMA_VERSION,
                        "source_track": tid,
                        "variant": variant,
                        "tempo_bpm": bpm,
                        "tempo_factor": factor,
                        "jitter_sigma_ms": sigma_ms,
                        "seed": seed,
                        "divisions_per_quarter": DIVISIONS_PER_QUARTER,
                        "time_signature": {"beats": 4, "beat_unit": 4},
                        "notes": rendered,
                    },
                    indent=1,
                )
            )
        tracks_meta.append({"track_id": tid, "gt_tempo_bpm": gt_tempo, "n_notes": len(score)})
        print(f"done {tid} ({len(score)} notes)")

    (DST / "manifest.json").write_text(
        json.dumps(
            {
                "source": "eval_data/guitarset (dev 10 tracks only; holdout NOT used)",
                "design": "docs/DESIGN_M4_QUANTIZATION.md §4.3",
                "generator": "make_rhythm_synth.py (this directory)",
                "renderer": {
                    "method": "karplus_strong",
                    "note": "fluidsynth 未導入環境のための設計書記載の代替合成",
                    "sample_rate": SR,
                    "decay": KS_DECAY,
                    "note_gain": NOTE_GAIN,
                    "fade_sec": FADE_SEC,
                    "peak_dbfs": -1.0,
                },
                "grid": {
                    "divisions_per_quarter": DIVISIONS_PER_QUARTER,
                    "allowed_tick_residues": list(ALLOWED_TICK_RESIDUES),
                    "phase": "beat1 = t0 (GuitarSet beat GT準拠)",
                },
                "variants": {
                    name: {"tempo_factor": f, "jitter_sigma_ms": s}
                    for name, (f, s) in VARIANTS.items()
                },
                "tracks": tracks_meta,
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    print(f"manifest written: {DST / 'manifest.json'}")


if __name__ == "__main__":
    main()
