"""MuScriptor ランナースクリプト（別 venv で単独実行される）。

guitartab 本体からは import せず、MuScriptorEngine がサブプロセスとして
ファイルパス指定で実行する。したがって:
- guitartab パッケージを import してはならない（muscriptor venv には入っていない）
- 依存は stdlib + soundfile + muscriptor のみ
- 出力 JSON のスキーマは guitartab/transcribe/base.py の notes.json schema 1 と
  手動で同期すること

重みはゲート付き HF リポジトリ（CC BY-NC・要 HF_TOKEN。親プロセスの環境変数を
そのまま継承する。docs/MUSCRIPTOR_VERIFICATION_2026-07-17.md）。

Usage: python _muscriptor_runner.py <audio.wav> <out_notes.json> [params_json]

params_json の許可キーは RUNNER_PARAMS:
- model: モデルサイズ（デフォルト "small"）
- device: "mps"（デフォルト）| "cpu" | "cuda"。MPS は明示指定が必須のためデフォルト側で指定
- batch_size: チャンクバッチサイズ（デフォルト 4。1 だと CPU より遅い）
- instruments: 生成条件付けプロンプト（非空の楽器名リスト。
  デフォルト ["acoustic_guitar", "distorted_electric_guitar"] = ベンチ採用構成）
- cfg_coef: classifier-free guidance 係数（デフォルト 1.5。**1.6 以上で縮退暴走の実測あり**、
  docs/BENCHMARKS.md 2026-07-18 スイープ）
- use_sampling / temperature: サンプリング生成（デフォルト無効 = greedy。実測で greedy に劣る）
- max_notes_per_sec: 暴走検知閾値（デフォルト 30 notes/sec。0 で無効化）

exit code: 0 成功 / 2 引数エラー / 3 生成暴走検知（EXIT_RUNAWAY、stderr にノート数を出力）
"""

import json
import os
import sys

# スクリプト実行では sys.path[0] が本ファイルのディレクトリになり、隣の
# エンジンモジュール muscriptor.py が muscriptor **パッケージ**を隠してしまう
# （エンジン側は guitartab を import するので即死する）。ここで自ディレクトリを
# sys.path から除去してから muscriptor パッケージを import できるようにする。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [
    p for p in sys.path if os.path.abspath(p or os.getcwd()) != _SCRIPT_DIR
]

RUNNER_PARAMS = frozenset(
    {
        "model",
        "device",
        "batch_size",
        "instruments",
        "cfg_coef",
        "use_sampling",
        "temperature",
        "max_notes_per_sec",
    }
)

EXIT_RUNAWAY = 3

# モデルは velocity を出さないため定数埋め（velocity=100 相当を 0-1 スキーマに正規化）
VELOCITY = 100 / 127.0
# 終端イベントが来なかった音符（dangling）の救済音価
DANGLING_NOTE_SEC = 0.05


def is_runaway(n_notes: int, audio_sec: float, max_notes_per_sec: float) -> bool:
    """音声1秒あたりのノート数が閾値を超えたか（縮退暴走検知）。閾値 0 以下は無効。"""
    if max_notes_per_sec <= 0 or audio_sec <= 0:
        return False
    return n_notes / audio_sec > max_notes_per_sec


def parse_params(argv: list) -> tuple:
    """引数を検証して (audio_path, out_path, params) を返す。エラー時は SystemExit(2)。

    重い import（torch / muscriptor）より前に呼ぶこと。
    """
    if len(argv) not in (3, 4):
        print(
            "usage: python _muscriptor_runner.py <audio.wav> <out_notes.json> "
            "[params_json]",
            file=sys.stderr,
        )
        raise SystemExit(2)

    audio_path, out_path = argv[1], argv[2]

    params = {}
    if len(argv) == 4:
        try:
            params = json.loads(argv[3])
        except json.JSONDecodeError as e:
            print(f"invalid params_json: {e}", file=sys.stderr)
            raise SystemExit(2)
        if not isinstance(params, dict):
            print("params_json must be a JSON object", file=sys.stderr)
            raise SystemExit(2)
        unknown = sorted(set(params) - RUNNER_PARAMS)
        if unknown:
            print(
                f"unknown runner params: {', '.join(unknown)} "
                f"(allowed: {', '.join(sorted(RUNNER_PARAMS))})",
                file=sys.stderr,
            )
            raise SystemExit(2)

    params.setdefault("model", "small")
    params.setdefault("device", "mps")
    params.setdefault("batch_size", 4)
    params.setdefault("instruments", ["acoustic_guitar", "distorted_electric_guitar"])
    params.setdefault("cfg_coef", 1.5)
    params.setdefault("use_sampling", False)
    params.setdefault("temperature", 1.0)
    params.setdefault("max_notes_per_sec", 30.0)

    instruments = params["instruments"]
    if (
        not isinstance(instruments, list)
        or not instruments
        or not all(isinstance(i, str) and i for i in instruments)
    ):
        # 無条件生成（instruments なし）は他楽器も転写して est が爆発するため許可しない
        # （docs/BENCHMARKS.md スイープ #9: est 14061）
        print(
            "instruments must be a non-empty list of instrument names",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if not os.path.exists(audio_path):
        print(f"audio not found: {audio_path}", file=sys.stderr)
        raise SystemExit(2)

    return audio_path, out_path, params


def transcribe(audio_path: str, params: dict) -> tuple:
    """audio → (notes, audio_sec, n_dangling)。notes は schema 1 の notes 相当。"""
    import soundfile as sf
    from muscriptor import TranscriptionModel
    from muscriptor.events import NoteEndEvent, NoteStartEvent

    info = sf.info(audio_path)
    audio_sec = info.frames / info.samplerate

    model = TranscriptionModel.load_model(params["model"], device=params["device"])

    gen_kwargs = {}
    if params["use_sampling"]:
        gen_kwargs["use_sampling"] = True
        gen_kwargs["temperature"] = float(params["temperature"])

    starts = {}  # index -> NoteStartEvent
    notes = []
    for ev in model.transcribe(
        audio_path,
        instruments=params["instruments"],
        batch_size=int(params["batch_size"]),
        cfg_coef=float(params["cfg_coef"]),
        **gen_kwargs,
    ):
        if isinstance(ev, NoteStartEvent):
            starts[ev.index] = ev
        elif isinstance(ev, NoteEndEvent):
            s = ev.start_event
            starts.pop(s.index, None)
            notes.append(
                {
                    "onset_sec": float(s.start_time),
                    "offset_sec": float(ev.end_time),
                    "midi_pitch": int(s.pitch),
                    "velocity": VELOCITY,
                    "confidence": 1.0,
                }
            )
    # 終端イベントが来なかった音符は最小音価で救済
    n_dangling = len(starts)
    for s in starts.values():
        notes.append(
            {
                "onset_sec": float(s.start_time),
                "offset_sec": float(s.start_time) + DANGLING_NOTE_SEC,
                "midi_pitch": int(s.pitch),
                "velocity": VELOCITY,
                "confidence": 1.0,
            }
        )
    notes.sort(key=lambda n: (n["onset_sec"], n["midi_pitch"]))
    return notes, audio_sec, n_dangling


def main() -> int:
    audio_path, out_path, params = parse_params(sys.argv)
    notes, audio_sec, n_dangling = transcribe(audio_path, params)
    if n_dangling:
        print(f"dangling note-start events rescued: {n_dangling}", file=sys.stderr)

    limit = float(params["max_notes_per_sec"])
    if is_runaway(len(notes), audio_sec, limit):
        rate = len(notes) / audio_sec
        print(
            f"runaway generation detected: {len(notes)} notes in "
            f"{audio_sec:.1f}s audio = {rate:.1f} notes/sec "
            f"(limit {limit:g} notes/sec)",
            file=sys.stderr,
        )
        return EXIT_RUNAWAY

    with open(out_path, "w") as f:
        json.dump({"schema": 1, "notes": notes}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
