"""benchmark.py の発見ロジックと比較表出力をダミーエンジンで検証する。"""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from guitartab.eval.benchmark import discover_items, format_table, run_benchmark
from guitartab.eval.loaders import load_ground_truth
from guitartab.transcribe.base import NoteEvent


class GroundTruthEngine:
    """GT をそのまま返すダミーエンジン（F1=1.0 になるはず）。"""

    name = "gt-oracle"

    def __init__(self, gt_path: Path):
        self._gt_path = gt_path

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        return load_ground_truth(self._gt_path)


class SilentEngine:
    """何も検出しないダミーエンジン（F1=0.0 になるはず）。"""

    name = "silent"

    def transcribe(self, audio_path: Path) -> list[NoteEvent]:
        return []


@pytest.fixture
def bench_root(tmp_path):
    item_dir = tmp_path / "items" / "song1"
    item_dir.mkdir(parents=True)
    # 1秒の無音 WAV（ネットワーク・実エンジン不要のダミー音源）
    sf.write(str(item_dir / "audio.wav"), np.zeros(22050, dtype=np.float32), 22050)
    gt = [{"time": 0.0, "string": 5, "fret": 0, "duration": 0.25}]
    (item_dir / "ground_truth.json").write_text(json.dumps(gt))
    return tmp_path


def test_discover_items(bench_root):
    items = discover_items(bench_root)
    assert len(items) == 1
    assert items[0].item_id == "items/song1"
    assert items[0].audio_path.name == "audio.wav"
    assert items[0].gt_path.name == "ground_truth.json"


def test_discover_guitarset_layout(tmp_path):
    # annotations/ と audio/ が並列の GuitarSet 形式
    root = tmp_path / "guitarset"
    (root / "annotations").mkdir(parents=True)
    (root / "audio").mkdir()
    jams_doc = {
        "annotations": [
            {
                "namespace": "note_midi",
                "data": [{"time": 0.0, "duration": 0.5, "value": 45.0, "confidence": None}],
            }
        ]
    }
    (root / "annotations" / "00_BN3-119-G_solo.jams").write_text(json.dumps(jams_doc))
    (root / "annotations" / "01_missing_audio.jams").write_text(json.dumps(jams_doc))
    sf.write(
        str(root / "audio" / "00_BN3-119-G_solo_mic.wav"),
        np.zeros(22050, dtype=np.float32),
        22050,
    )

    items = discover_items(tmp_path)
    # 音声のあるアノテーションだけがアイテムになる
    assert len(items) == 1
    assert items[0].item_id == "guitarset/00_BN3-119-G_solo"
    assert items[0].audio_path.name == "00_BN3-119-G_solo_mic.wav"
    assert items[0].gt_path.name == "00_BN3-119-G_solo.jams"


def test_discover_items_empty_dir(tmp_path):
    assert discover_items(tmp_path) == []
    assert discover_items(tmp_path / "missing") == []


def test_run_benchmark_and_table(bench_root, tmp_path):
    items = discover_items(bench_root)
    gt_path = items[0].gt_path
    out_dir = tmp_path / "bench_out"
    result = run_benchmark(
        [GroundTruthEngine(gt_path), SilentEngine()], items, out_dir=out_dir
    )

    assert result.metrics["gt-oracle"]["items/song1"].f1 == pytest.approx(1.0)
    assert result.metrics["silent"]["items/song1"].f1 == pytest.approx(0.0)
    assert not result.errors["gt-oracle"]
    # 推定 notes.json が out_dir に残る
    assert (out_dir / "gt-oracle" / "items" / "song1.json").exists()

    table = format_table(result)
    assert "gt-oracle" in table
    assert "silent" in table
    assert "mean" in table
