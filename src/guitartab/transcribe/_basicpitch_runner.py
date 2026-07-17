"""basic-pitch ランナースクリプト（別 venv の Python 3.10 で単独実行される）。

guitartab 本体（Python 3.11）からは import せず、BasicPitchEngine が
サブプロセスとしてファイルパス指定で実行する。したがって:
- guitartab パッケージを import してはならない（basic-pitch venv には入っていない）
- 依存は stdlib + basic_pitch のみ
- 出力 JSON のスキーマは guitartab/transcribe/base.py の notes.json schema 1 と
  手動で同期すること

Usage: python _basicpitch_runner.py <audio.wav> <out_notes.json> [params_json]

params_json は basic_pitch.inference.predict() にそのまま渡す推論パラメータの
JSON オブジェクト（例: '{"onset_threshold": 0.7}'）。許可キーは PREDICT_PARAMS。
省略時は predict() のデフォルト（従来動作）。
"""

import json
import sys

# basic_pitch.inference.predict() が受けるネイティブ推論パラメータのみ許可する。
# 自作の後処理フィルタはここに追加しない（docs/DESIGN.md 開発ルール参照）。
PREDICT_PARAMS = frozenset(
    {
        "onset_threshold",
        "frame_threshold",
        "minimum_note_length",
        "minimum_frequency",
        "maximum_frequency",
        "multiple_pitch_bends",
        "melodia_trick",
    }
)


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(
            "usage: python _basicpitch_runner.py <audio.wav> <out_notes.json> "
            "[params_json]",
            file=sys.stderr,
        )
        return 2

    audio_path, out_path = sys.argv[1], sys.argv[2]

    params = {}
    if len(sys.argv) == 4:
        try:
            params = json.loads(sys.argv[3])
        except json.JSONDecodeError as e:
            print(f"invalid params_json: {e}", file=sys.stderr)
            return 2
        if not isinstance(params, dict):
            print("params_json must be a JSON object", file=sys.stderr)
            return 2
        unknown = sorted(set(params) - PREDICT_PARAMS)
        if unknown:
            print(
                f"unknown predict() params: {', '.join(unknown)} "
                f"(allowed: {', '.join(sorted(PREDICT_PARAMS))})",
                file=sys.stderr,
            )
            return 2

    from basic_pitch.inference import predict

    # note_events: list of (start_sec, end_sec, midi_pitch, amplitude, pitch_bends)
    _model_output, _midi_data, note_events = predict(audio_path, **params)

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
