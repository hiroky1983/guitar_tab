"""basic-pitch ランナースクリプト（別 venv の Python 3.10 で単独実行される）。

guitartab 本体（Python 3.11）からは import せず、BasicPitchEngine が
サブプロセスとしてファイルパス指定で実行する。したがって:
- guitartab パッケージを import してはならない（basic-pitch venv には入っていない）
- 依存は stdlib + basic_pitch のみ
- 出力 JSON のスキーマは guitartab/transcribe/base.py の notes.json schema 1 と
  手動で同期すること

Usage: python _basicpitch_runner.py <audio.wav> <out_notes.json>
"""

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python _basicpitch_runner.py <audio.wav> <out_notes.json>",
            file=sys.stderr,
        )
        return 2

    audio_path, out_path = sys.argv[1], sys.argv[2]

    from basic_pitch.inference import predict

    # note_events: list of (start_sec, end_sec, midi_pitch, amplitude, pitch_bends)
    _model_output, _midi_data, note_events = predict(audio_path)

    notes = [
        {
            "onset_sec": float(start),
            "offset_sec": float(end),
            "midi_pitch": int(pitch),
            "velocity": float(amplitude),
            # basic-pitch は独立した confidence を出さないため amplitude を流用
            "confidence": float(amplitude),
        }
        for (start, end, pitch, amplitude, *_rest) in note_events
    ]
    notes.sort(key=lambda n: (n["onset_sec"], n["midi_pitch"]))

    with open(out_path, "w") as f:
        json.dump({"schema": 1, "notes": notes}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
