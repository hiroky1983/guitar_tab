"""GuitarSet dev 10トラックから合成歪みベンチ eval_data/distorted_synth/ を生成する。

- ソース audio は読み取りのみ。JAMS はバイト無改変コピー。
- 歪みは pedalboard（Spotify, C++）による決定論的処理。
  タイミング/ピッチを変えない処理のみ使用（波形整形・IIRフィルタ・コンプ。
  時間伸縮/ピッチシフト/ディレイ系は不使用）ため GT をそのまま流用できる。

実行: distenv/bin/python make_distorted.py
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pedalboard as pb
import soundfile as sf

REPO = Path("/Users/yamadahiroki/myspace/guitar_tab")
SRC = REPO / "eval_data" / "guitarset"
DST = REPO / "eval_data" / "distorted_synth"

# プリセット定義（README にもこの値を転記する）
PRESETS = {
    "crunch": [
        ("HighpassFilter", {"cutoff_frequency_hz": 60.0}),
        ("Gain", {"gain_db": 14.0}),                     # プリゲイン
        ("Distortion", {"drive_db": 18.0}),              # tanh 波形整形
        ("PeakFilter", {"cutoff_frequency_hz": 2200.0, "gain_db": 3.0, "q": 0.9}),  # プレゼンス
        ("LowpassFilter", {"cutoff_frequency_hz": 5500.0}),  # キャビネット風ロールオフ
    ],
    "highgain": [
        ("HighpassFilter", {"cutoff_frequency_hz": 60.0}),
        ("Compressor", {"threshold_db": -30.0, "ratio": 4.0,
                        "attack_ms": 2.0, "release_ms": 120.0}),  # サステイン/コンプ感
        ("Gain", {"gain_db": 26.0}),                     # プリゲイン（強）
        ("Distortion", {"drive_db": 32.0}),              # tanh 波形整形（強）
        ("Clipping", {"threshold_db": -3.0}),            # ハードクリップ
        ("PeakFilter", {"cutoff_frequency_hz": 700.0, "gain_db": -4.0, "q": 0.7}),  # ミッドスクープ
        ("LowpassFilter", {"cutoff_frequency_hz": 4500.0}),  # キャビネット風ロールオフ
    ],
}
TARGET_PEAK_DBFS = -1.0  # 処理後にピーク正規化


def build_board(spec):
    return pb.Pedalboard([getattr(pb, name)(**kw) for name, kw in spec])


def main():
    manifest_src = json.loads((SRC / "manifest.json").read_text())
    for preset_name, spec in PRESETS.items():
        out_root = DST / preset_name
        (out_root / "audio").mkdir(parents=True, exist_ok=True)
        (out_root / "annotations").mkdir(parents=True, exist_ok=True)
        board = build_board(spec)
        tracks = []
        for t in manifest_src["tracks"]:
            tid = t["track_id"]
            src_audio = REPO / t["audio"]
            src_jams = REPO / t["annotation"]
            audio, sr = sf.read(src_audio, dtype="float64")
            out = board(audio.astype(np.float32), sr)
            peak = float(np.max(np.abs(out)))
            gain = 10 ** (TARGET_PEAK_DBFS / 20) / peak
            out = out * gain
            dst_audio = out_root / "audio" / f"{tid}_{preset_name}.wav"
            dst_jams = out_root / "annotations" / f"{tid}.jams"
            sf.write(dst_audio, out, sr, subtype="PCM_16")
            shutil.copyfile(src_jams, dst_jams)  # 無改変コピー
            tracks.append(
                {
                    **{k: t[k] for k in ("track_id", "player", "style", "tempo", "key", "mode")},
                    "audio": str(dst_audio.relative_to(REPO)),
                    "annotation": str(dst_jams.relative_to(REPO)),
                    "source_audio": t["audio"],
                    "post_norm_gain_db": round(20 * np.log10(gain), 2),
                }
            )
            print(f"{preset_name}: {tid} peak_before_norm={20*np.log10(peak):+.1f}dBFS")
        manifest = {
            "source": "eval_data/guitarset (GuitarSet Zenodo record 3371780, audio_mono-mic), "
                      "synthetically distorted — NOT a real amplifier recording",
            "generator": "pedalboard 0.9.24 (Spotify), see eval_data/distorted_synth/README.md",
            "preset": preset_name,
            "chain": [{"plugin": n, "params": kw} for n, kw in spec],
            "post_normalization_peak_dbfs": TARGET_PEAK_DBFS,
            "annotations": "GuitarSet JAMS copied byte-identical (distortion is time/pitch preserving)",
            "tracks": tracks,
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()
