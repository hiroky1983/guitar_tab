"""metrics.py を合成データで検証する。"""

import pytest

from guitartab.eval.metrics import evaluate_notes
from guitartab.transcribe.base import NoteEvent


def _note(onset, pitch, dur=0.25):
    return NoteEvent(onset_sec=onset, offset_sec=onset + dur, midi_pitch=pitch)


REF = [_note(0.0, 45), _note(0.5, 50), _note(1.0, 55), _note(1.5, 59)]


def test_perfect_match_gives_f1_1():
    m = evaluate_notes(list(REF), list(REF))
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.n_ref == m.n_est == 4


def test_half_correct_gives_expected_values():
    # 4音中2音は正解、2音はピッチが半音以上ずれている → P=R=F1=0.5
    est = [_note(0.0, 45), _note(0.5, 50), _note(1.0, 57), _note(1.5, 61)]
    m = evaluate_notes(est, list(REF))
    assert m.precision == pytest.approx(0.5)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(0.5)


def test_missing_notes_affect_recall_only():
    # 正解4音のうち2音だけ検出（検出した2音は正確）→ P=1.0, R=0.5, F1=2/3
    est = [_note(0.0, 45), _note(0.5, 50)]
    m = evaluate_notes(est, list(REF))
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(0.5)
    assert m.f1 == pytest.approx(2 / 3)


def test_onset_within_tolerance_matches():
    est = [_note(0.03, 45)]  # 30ms ずれ < 50ms トレランス
    m = evaluate_notes(est, [_note(0.0, 45)])
    assert m.f1 == pytest.approx(1.0)


def test_onset_beyond_tolerance_does_not_match():
    est = [_note(0.2, 45)]  # 200ms ずれ > 50ms トレランス
    m = evaluate_notes(est, [_note(0.0, 45)])
    assert m.f1 == pytest.approx(0.0)


def test_onset_tolerance_is_configurable():
    est = [_note(0.2, 45)]
    m = evaluate_notes(est, [_note(0.0, 45)], onset_tolerance_sec=0.3)
    assert m.f1 == pytest.approx(1.0)


def test_pitch_off_by_semitone_does_not_match():
    m = evaluate_notes([_note(0.0, 46)], [_note(0.0, 45)])
    assert m.f1 == pytest.approx(0.0)


def test_empty_cases():
    assert evaluate_notes([], []).f1 == pytest.approx(1.0)
    assert evaluate_notes([], list(REF)).f1 == pytest.approx(0.0)
    assert evaluate_notes(list(REF), []).f1 == pytest.approx(0.0)
