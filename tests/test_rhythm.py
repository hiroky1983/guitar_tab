"""リズム量子化ステージ（rhythm/）のテスト。

- rhythm.json スキーマの往復
- 最近傍スナップの境界（タイブレーク・3 連 vs 16 分・負 tick の小節繰り上げ）
- テンポ推定（音声なしフォールバック）の合成ケース
- リズムメトリクス（TempoAcc / GPA / 変位統計）
- レンダラの rhythm 供給時の動作とフォールバック不変
"""

from pathlib import Path

import pytest

from guitartab.rhythm.estimate import LibrosaConstantTempoEstimator, TempoEstimate
from guitartab.rhythm.quantize import quantize_notes, snap_to_grid
from guitartab.rhythm.schema import (
    ALLOWED_TICK_RESIDUES,
    DIVISIONS_PER_QUARTER,
    Beat,
    Rhythm,
    RhythmNote,
    TempoPoint,
    load_rhythm,
    lookup_note_by_onset,
    save_rhythm,
)
from guitartab.eval.rhythm_metrics import (
    displacement_stats,
    evaluate_beats,
    grid_position_accuracy,
    tempo_acc1,
    tempo_acc2,
)
from guitartab.transcribe.base import NoteEvent


def _estimate(bpm: float = 120.0, origin: float = 0.0) -> TempoEstimate:
    return TempoEstimate(bpm=bpm, grid_origin_sec=origin, estimator="test")


def _note(onset: float, dur: float = 0.25, pitch: int = 60) -> NoteEvent:
    return NoteEvent(onset_sec=onset, offset_sec=onset + dur, midi_pitch=pitch)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_rhythm_json_roundtrip(tmp_path):
    rhythm = Rhythm(
        tempo_bpm=119.0,
        tempo_map=[TempoPoint(time_sec=0.25, bpm=119.0)],
        notes=[
            RhythmNote(onset_tick=0, duration_ticks=6, deviation_sec=-0.012),
            RhythmNote(onset_tick=9, duration_ticks=3, deviation_sec=0.02),
        ],
        beats=[Beat(time_sec=0.25, measure=1, position=1)],
        estimator="librosa_constant",
        estimator_params={"dp_weight": 0.5},
        audio_source="stems/guitar.wav",
    )
    path = tmp_path / "rhythm.json"
    save_rhythm(rhythm, path)
    loaded = load_rhythm(path)
    assert loaded == rhythm


def test_load_rhythm_rejects_unknown_schema(tmp_path):
    path = tmp_path / "rhythm.json"
    path.write_text('{"schema_version": 99}')
    with pytest.raises(ValueError, match="schema_version"):
        load_rhythm(path)


def test_tick_to_sec_and_onset_reconstruction():
    rhythm = Rhythm(
        tempo_bpm=120.0,
        tempo_map=[TempoPoint(time_sec=1.0, bpm=120.0)],
        notes=[RhythmNote(onset_tick=12, duration_ticks=6, deviation_sec=0.01)],
    )
    # 120BPM: 1 拍 = 0.5 秒、12 tick = 1 拍
    assert rhythm.tick_to_sec(12) == pytest.approx(1.5)
    assert rhythm.onset_sec_of(0) == pytest.approx(1.51)


def test_lookup_note_by_onset():
    rhythm = Rhythm(
        tempo_bpm=120.0,
        tempo_map=[TempoPoint(time_sec=0.0, bpm=120.0)],
        notes=[
            RhythmNote(onset_tick=0, duration_ticks=6, deviation_sec=0.0),
            RhythmNote(onset_tick=6, duration_ticks=3, deviation_sec=-0.02),
        ],
    )
    # notes[1] の元 onset = 0.25 - 0.02 = 0.23
    assert lookup_note_by_onset(rhythm, 0.23) == rhythm.notes[1]
    assert lookup_note_by_onset(rhythm, 0.231) == rhythm.notes[1]
    assert lookup_note_by_onset(rhythm, 0.4) is None


# ---------------------------------------------------------------------------
# snap / quantize
# ---------------------------------------------------------------------------


def test_snap_exact_grid_points():
    est = _estimate(bpm=120.0)  # 1 拍 = 0.5 秒、1 tick = 0.5/12 秒
    tick_sec = 0.5 / 12
    for residue in ALLOWED_TICK_RESIDUES:
        assert snap_to_grid(residue * tick_sec, est) == residue
    assert snap_to_grid(0.5, est) == 12  # 次拍の頭
    assert snap_to_grid(1.0, est) == 24


def test_snap_tie_breaks_to_lower_tick():
    est = _estimate(bpm=120.0)
    tick_sec = 0.5 / 12
    # 3.5 tick は 16 分(3) と 3 連(4) の中点 → 小さい側の 3
    assert snap_to_grid(3.5 * tick_sec, est) == 3
    # 中点 10.5 tick（9 と 12 の中点）→ 小さい側の 9
    assert snap_to_grid(10.5 * tick_sec, est) == 9


def test_snap_disallowed_residues_never_returned():
    est = _estimate(bpm=120.0)
    tick_sec = 0.5 / 12
    for i in range(0, 240):
        tick = snap_to_grid(i * tick_sec / 10, est)
        assert tick % 12 in ALLOWED_TICK_RESIDUES


def test_quantize_notes_basic():
    est = _estimate(bpm=120.0)
    notes = [
        _note(0.01, dur=0.25),    # tick 0 へ（deviation +0.01）、6 tick
        _note(0.125, dur=0.125),  # 16 分 = tick 3、3 tick
        _note(1.0 / 3, dur=0.1),  # 3 連 8 分 = tick 8
    ]
    rhythm = quantize_notes(notes, est, audio_source="a.wav")
    assert [n.onset_tick for n in rhythm.notes] == [0, 3, 8]
    assert rhythm.notes[0].duration_ticks == 6
    assert rhythm.notes[1].duration_ticks == 3
    assert rhythm.notes[0].deviation_sec == pytest.approx(0.01)
    assert rhythm.tempo_bpm == 120.0
    assert rhythm.tempo_map[0].time_sec == pytest.approx(0.0)
    assert rhythm.audio_source == "a.wav"
    assert rhythm.divisions_per_quarter == DIVISIONS_PER_QUARTER
    # 拍列は格子原点=小節1拍1 の機械的割当
    assert rhythm.beats[0] == Beat(time_sec=pytest.approx(0.0), measure=1, position=1)


def test_quantize_notes_duration_minimum_one_tick():
    est = _estimate(bpm=120.0)
    rhythm = quantize_notes([_note(0.0, dur=0.001)], est)
    assert rhythm.notes[0].duration_ticks == 1


def test_quantize_notes_shifts_origin_for_negative_ticks():
    # 格子原点より 1 拍以上前の音符 → 小節単位で原点を繰り上げて非負 tick に
    est = _estimate(bpm=120.0, origin=1.9)
    notes = [_note(0.4), _note(1.9)]
    rhythm = quantize_notes(notes, est)
    assert all(n.onset_tick >= 0 for n in rhythm.notes)
    # 0.4 秒は原点(1.9)の 3 拍前 = -36 tick → 1 小節(48 tick)繰り上げで 12
    assert rhythm.notes[0].onset_tick == 12
    assert rhythm.notes[1].onset_tick == 48
    # 原点の物理時刻も 1 小節(2 秒)前へ
    assert rhythm.tempo_map[0].time_sec == pytest.approx(-0.1)


def test_quantize_preserves_input_order_and_count():
    est = _estimate(bpm=100.0)
    notes = [_note(t * 0.11) for t in range(20)]
    rhythm = quantize_notes(notes, est)
    assert len(rhythm.notes) == 20
    # onset_sec を復元するとほぼ元の値（deviation 込みで厳密一致）
    for i, note in enumerate(notes):
        assert rhythm.onset_sec_of(i) == pytest.approx(note.onset_sec)


# ---------------------------------------------------------------------------
# estimator（音声なしフォールバック）
# ---------------------------------------------------------------------------


def test_estimator_notes_only_recovers_tempo():
    # 100 BPM・位相 0.3 秒、16 分と 3 連を含むパターン（半分/2 倍と区別可能）
    bpm, phase = 100.0, 0.3
    beat = 60.0 / bpm
    onsets = []
    for k in range(16):
        for frac in (0.0, 0.25, 0.5, 2 / 3, 0.75):
            onsets.append(phase + (k + frac) * beat)
    est = LibrosaConstantTempoEstimator().estimate(onsets)
    assert est.bpm == pytest.approx(bpm, rel=0.01)
    # 音声なし経路は位相を P/4（16 分格子の自己一致周期）の法でのみ決める
    cell = 60.0 / est.bpm / 4
    offset = (est.grid_origin_sec - phase) % cell
    assert min(offset, cell - offset) < 0.01


def test_estimator_notes_only_quantizes_back_to_grid():
    bpm = 100.0
    beat = 60.0 / bpm
    onsets = [k * beat / 4 for k in range(64)]  # 16 分の連続
    est = LibrosaConstantTempoEstimator().estimate(onsets)
    notes = [_note(t, dur=beat / 4) for t in onsets]
    rhythm = quantize_notes(notes, est)
    ticks = [n.onset_tick for n in rhythm.notes]
    diffs = {b - a for a, b in zip(ticks, ticks[1:])}
    assert len(diffs) == 1  # 等間隔が保たれる

def test_estimator_empty_input_returns_default():
    est = LibrosaConstantTempoEstimator().estimate([])
    assert est.bpm == 120.0


# ---------------------------------------------------------------------------
# 音声トラッカー信頼モード（trust_tracker、M4b）
# ---------------------------------------------------------------------------


def _click_wav(tmp_path, bpm=120.0, dur_sec=12.0, sr=22050):
    """一定テンポのクリック音声を合成して wav パスを返す。"""
    import numpy as np
    import soundfile as sf

    y = np.zeros(int(dur_sec * sr), dtype=np.float32)
    n = 256
    click = (
        np.hanning(n) * np.sin(2 * np.pi * 1000.0 * np.arange(n) / sr)
    ).astype(np.float32)
    t = 0.0
    while t < dur_sec - n / sr:
        i = int(t * sr)
        y[i : i + n] += click
        t += 60.0 / bpm
    path = tmp_path / "click.wav"
    sf.write(path, y, sr)
    return path


def test_trust_tracker_default_off_and_recorded_in_params():
    est = LibrosaConstantTempoEstimator()
    assert est.trust_tracker is False
    assert est._params(audio_used=False)["trust_tracker"] is False
    assert (
        LibrosaConstantTempoEstimator(trust_tracker=True)._params(audio_used=True)[
            "trust_tracker"
        ]
        is True
    )


def test_trust_tracker_notes_only_path_identical(tmp_path):
    """音声なしでは信頼モードは従来のノートのみフォールバックと同一結果。"""
    bpm, beat = 100.0, 0.6
    onsets = [k * beat / 4 for k in range(64)]
    base = LibrosaConstantTempoEstimator().estimate(onsets)
    trusted = LibrosaConstantTempoEstimator(trust_tracker=True).estimate(onsets)
    assert trusted.bpm == base.bpm
    assert trusted.grid_origin_sec == base.grid_origin_sec


def test_trust_tracker_keeps_tracker_tempo_level(tmp_path):
    """信頼モードは生 beat_track のテンポレベルを上書きしない。"""
    import librosa
    import numpy as np

    audio = _click_wav(tmp_path, bpm=120.0)
    # 8 分主体のノート列（半テンポ格子も完全適合し、選択層がレベルを誤り得る形）
    onsets = [k * 0.25 for k in range(48)]
    y, _ = librosa.load(audio, sr=22050, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=22050, hop_length=512)
    raw_tempo, _b = librosa.beat.beat_track(
        onset_envelope=env, sr=22050, hop_length=512, units="time"
    )
    raw_tempo = float(np.atleast_1d(raw_tempo)[0])

    est = LibrosaConstantTempoEstimator(trust_tracker=True).estimate(
        onsets, audio_path=audio
    )
    # 生トラッカーのテンポ ±6% 内に収まる（ノート格子による上書きなし）
    assert abs(np.log2(est.bpm / raw_tempo)) <= np.log2(1.06)
    assert est.params["trust_tracker"] is True


def test_fit_beats_bpm_robust_to_missing_beats():
    import numpy as np

    from guitartab.rhythm.estimate import _fit_beats_bpm

    period = 0.5  # 120 BPM
    beats = [0.1 + k * period for k in range(40) if k != 7]  # 1 拍欠落
    bpm = _fit_beats_bpm(np.asarray(beats))
    assert bpm == pytest.approx(120.0, rel=1e-6)
    assert _fit_beats_bpm(np.asarray([1.0])) is None


def test_trust_tracker_bypasses_candidate_selection(tmp_path, monkeypatch):
    """信頼モードはテンポ精密化 _refine を呼ばず（ノートは位相のみ）、
    従来モードは候補族の走査で複数回呼ぶ（stem 経路の従来コードパス証明）。"""
    audio = _click_wav(tmp_path, bpm=120.0)
    onsets = [k * 0.25 for k in range(48)]

    calls = []
    orig_refine = LibrosaConstantTempoEstimator._refine

    def spy_refine(self, o, bpm0):
        calls.append(bpm0)
        return orig_refine(self, o, bpm0)

    monkeypatch.setattr(LibrosaConstantTempoEstimator, "_refine", spy_refine)

    LibrosaConstantTempoEstimator(trust_tracker=True).estimate(
        onsets, audio_path=audio
    )
    assert calls == []  # ノート格子適合でテンポを動かさない

    LibrosaConstantTempoEstimator().estimate(onsets, audio_path=audio)
    assert len(calls) > 1  # 従来モードは候補族を走査する（stem 経路は不変）


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def test_tempo_acc1_and_acc2():
    assert tempo_acc1(119.0, 119.0)
    assert tempo_acc1(119.0 * 1.03, 119.0)
    assert not tempo_acc1(119.0 * 1.05, 119.0)
    assert not tempo_acc1(59.5, 119.0)
    assert tempo_acc2(59.5, 119.0)      # 1/2
    assert tempo_acc2(238.0, 119.0)     # 2
    assert tempo_acc2(357.0, 119.0)     # 3
    assert tempo_acc2(119.0 / 3, 119.0)  # 1/3
    assert not tempo_acc2(119.0 * 1.5, 119.0)  # 3/2 は族に含めない（標準 Acc2）


def test_grid_position_accuracy():
    assert grid_position_accuracy([0, 3, 6], [0, 3, 6]) == 1.0
    assert grid_position_accuracy([0, 4, 6], [0, 3, 6]) == pytest.approx(2 / 3)
    assert grid_position_accuracy([], []) == 1.0
    with pytest.raises(ValueError):
        grid_position_accuracy([0], [0, 3])


def test_displacement_stats():
    stats = displacement_stats([-0.02, 0.01, 0.03, -0.04, 0.0])
    assert stats.median_sec == pytest.approx(0.02)
    assert stats.p90_sec <= 0.04
    assert displacement_stats([]).median_sec == 0.0


def test_evaluate_beats_perfect_and_offset():
    ref = [i * 0.5 for i in range(60)]
    perfect = evaluate_beats(ref, ref)
    assert perfect.f_measure == pytest.approx(1.0)
    shifted = evaluate_beats([t + 0.25 for t in ref], ref)
    assert shifted.f_measure == pytest.approx(0.0)
