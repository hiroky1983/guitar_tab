"""Beat This! ランナースクリプト（別 venv で単独実行される）。

guitartab 本体からは import せず、BeatThisTempoEstimator がサブプロセスとして
ファイルパス指定で実行する（transcribe/_muscriptor_runner.py と同型）。したがって:
- guitartab パッケージを import してはならない（beatthis venv には入っていない）
- 依存は stdlib + beat_this（+ torch / soundfile）のみ
- 出力 JSON は beats/downbeats の時刻列（beatthis.py 側と手動同期）

チェックポイントは初回に JKU クラウド（cloud.cp.jku.at、直接 HTTPS）から
torch.hub キャッシュへ自動 DL される（final0 = 78MB。Google Drive 経由ではない）。

Usage: python _beatthis_runner.py <audio.wav> <out_beats.json> [params_json]

params_json の許可キーは RUNNER_PARAMS:
- checkpoint: チェックポイント名 / パス / URL（デフォルト "final0"）
- device: "cpu"（デフォルト）| "mps" | "cuda"。GuitarSet 級の 30 秒クリップでは
  CPU の方が MPS より速い実測（1.3s vs 3.4s）のため CPU をデフォルトにする
- dbn: madmom DBN 後処理（デフォルト False。madmom は本プロジェクト不採用のため
  True にするには venv への追加インストールが必要）

exit code: 0 成功 / 2 引数エラー
"""

import json
import os
import sys
import time

RUNNER_PARAMS = frozenset({"checkpoint", "device", "dbn"})


def parse_params(argv: list) -> tuple:
    """引数を検証して (audio_path, out_path, params) を返す。エラー時は SystemExit(2)。

    重い import（torch / beat_this）より前に呼ぶこと。
    """
    if len(argv) not in (3, 4):
        print(
            "usage: python _beatthis_runner.py <audio.wav> <out_beats.json> "
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

    params.setdefault("checkpoint", "final0")
    params.setdefault("device", "cpu")
    params.setdefault("dbn", False)

    if not os.path.exists(audio_path):
        print(f"audio not found: {audio_path}", file=sys.stderr)
        raise SystemExit(2)

    return audio_path, out_path, params


def main() -> int:
    audio_path, out_path, params = parse_params(sys.argv)

    from beat_this.inference import File2Beats

    file2beats = File2Beats(
        checkpoint_path=params["checkpoint"],
        device=params["device"],
        dbn=bool(params["dbn"]),
    )
    t0 = time.time()
    beats, downbeats = file2beats(audio_path)
    infer_sec = time.time() - t0

    with open(out_path, "w") as f:
        json.dump(
            {
                "schema": 1,
                "beats_sec": [float(b) for b in beats],
                "downbeats_sec": [float(b) for b in downbeats],
                "checkpoint": params["checkpoint"],
                "device": params["device"],
                "dbn": bool(params["dbn"]),
                "infer_sec": infer_sec,
            },
            f,
            indent=1,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
