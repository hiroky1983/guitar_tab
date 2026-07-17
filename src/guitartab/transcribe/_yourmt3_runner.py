"""YourMT3+ ランナースクリプト（別 venv の Python 3.11 で単独実行される）。

guitartab 本体からは import せず、YourMT3Engine がサブプロセスとして
ファイルパス指定で実行する。したがって:
- guitartab パッケージを import してはならない（yourmt3 venv には入っていない）
- 依存は stdlib + torch/torchaudio/soundfile + YourMT3 コード
  （GUITARTAB_YOURMT3_HOME 配下、`amt/src` を sys.path に追加して import）のみ
- 出力 JSON のスキーマは guitartab/transcribe/base.py の notes.json schema 1 と
  手動で同期すること

YourMT3 のコード+チェックポイントは GPL/Apache 混在ライセンスのためリポジトリに
同梱しない（third_party/ は gitignore 済み）。取得方法は README を参照。
モデルは YPTF.MoE+Multi (noPS)（YourMT3+ 論文ベスト、HF Space mimbres/YourMT3 配布）。

Usage: python _yourmt3_runner.py <audio.wav> <out_notes.json> [params_json]

params_json の許可キーは RUNNER_PARAMS:
- home: YourMT3 コード+チェックポイントのディレクトリ（省略時は
  $GUITARTAB_YOURMT3_HOME、それもなければ third_party/yourmt3）
- device: "cpu"（デフォルト）| "mps" | "cuda"。M2 では MPS と CPU がほぼ同速の
  ため安定側の CPU をデフォルトとする（docs/YOURMT3_VERIFICATION_2026-07-17.md）
- batch_size: inference のセグメントバッチサイズ（デフォルト 8）
"""

import json
import os
import sys

RUNNER_PARAMS = frozenset({"home", "device", "batch_size"})

DEFAULT_HOME = os.path.join("third_party", "yourmt3")
HOME_ENV_VAR = "GUITARTAB_YOURMT3_HOME"

# YPTF.MoE+Multi (noPS) の起動引数一式（HF Space app.py デフォルトと同一構成）
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
MODEL_ARGS = [
    CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
    "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
    "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
    "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32",
]


def parse_params(argv: list) -> tuple:
    """引数を検証して (audio_path, out_path, params) を返す。エラー時は SystemExit(2)。

    重い import（torch / YourMT3）より前に呼ぶこと。
    """
    if len(argv) not in (3, 4):
        print(
            "usage: python _yourmt3_runner.py <audio.wav> <out_notes.json> "
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

    home = params.get("home") or os.environ.get(HOME_ENV_VAR) or DEFAULT_HOME
    home = os.path.abspath(home)
    if not os.path.isdir(os.path.join(home, "amt", "src")):
        print(
            f"YourMT3 home not found or invalid (missing amt/src): {home}\n"
            "Download the YourMT3 code + checkpoint (HF Space mimbres/YourMT3) "
            "into third_party/yourmt3, or set GUITARTAB_YOURMT3_HOME.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    params["home"] = home
    params.setdefault("device", "cpu")
    params.setdefault("batch_size", 8)

    if not os.path.exists(audio_path):
        print(f"audio not found: {audio_path}", file=sys.stderr)
        raise SystemExit(2)

    return audio_path, out_path, params


def load_model(device):
    """model_helper.load_model_checkpoint 相当（device 指定可能な最小再実装）。

    cwd = home 前提（config の save_dir "amt/logs" が cwd 相対のため）。
    """
    import argparse

    import torch
    from model.init_train import initialize_trainer, update_config
    from model.ymt3 import YourMT3
    from utils.task_manager import TaskManager
    from utils.utils import str2bool

    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", type=str)
    parser.add_argument("-p", "--project", type=str, default="ymt3")
    parser.add_argument("-ac", "--audio-codec", type=str, default=None)
    parser.add_argument("-hop", "--hop-length", type=int, default=None)
    parser.add_argument("-nmel", "--n-mels", type=int, default=None)
    parser.add_argument("-if", "--input-frames", type=int, default=None)
    parser.add_argument("-sqr", "--sca-use-query-residual", type=str2bool, default=None)
    parser.add_argument("-enc", "--encoder-type", type=str, default=None)
    parser.add_argument("-dec", "--decoder-type", type=str, default=None)
    parser.add_argument("-preenc", "--pre-encoder-type", type=str, default="default")
    parser.add_argument("-predec", "--pre-decoder-type", type=str, default="default")
    parser.add_argument("-cout", "--conv-out-channels", type=int, default=None)
    parser.add_argument("-tenc", "--task-cond-encoder", type=str2bool, default=True)
    parser.add_argument("-tdec", "--task-cond-decoder", type=str2bool, default=True)
    parser.add_argument("-df", "--d-feat", type=int, default=None)
    parser.add_argument("-pt", "--pretrained", type=str2bool, default=False)
    parser.add_argument("-b", "--base-name", type=str, default="google/t5-v1_1-small")
    parser.add_argument("-epe", "--encoder-position-encoding-type", type=str, default="default")
    parser.add_argument("-dpe", "--decoder-position-encoding-type", type=str, default="default")
    parser.add_argument("-twe", "--tie-word-embedding", type=str2bool, default=None)
    parser.add_argument("-el", "--event-length", type=int, default=None)
    parser.add_argument("-dl", "--d-latent", type=int, default=None)
    parser.add_argument("-nl", "--num-latents", type=int, default=None)
    parser.add_argument("-dpm", "--perceiver-tf-d-model", type=int, default=None)
    parser.add_argument("-npb", "--num-perceiver-tf-blocks", type=int, default=None)
    parser.add_argument("-npl", "--num-perceiver-tf-local-transformers-per-block", type=int, default=None)
    parser.add_argument("-npt", "--num-perceiver-tf-temporal-transformers-per-block", type=int, default=None)
    parser.add_argument("-atc", "--attention-to-channel", type=str2bool, default=None)
    parser.add_argument("-ln", "--layer-norm-type", type=str, default=None)
    parser.add_argument("-ff", "--ff-layer-type", type=str, default=None)
    parser.add_argument("-wf", "--ff-widening-factor", type=int, default=None)
    parser.add_argument("-nmoe", "--moe-num-experts", type=int, default=None)
    parser.add_argument("-kmoe", "--moe-topk", type=int, default=None)
    parser.add_argument("-act", "--hidden-act", type=str, default=None)
    parser.add_argument("-rt", "--rotary-type", type=str, default=None)
    parser.add_argument("-rk", "--rope-apply-to-keys", type=str2bool, default=None)
    parser.add_argument("-rp", "--rope-partial-pe", type=str2bool, default=None)
    parser.add_argument("-dff", "--decoder-ff-layer-type", type=str, default=None)
    parser.add_argument("-dwf", "--decoder-ff-widening-factor", type=int, default=None)
    parser.add_argument("-tk", "--task", type=str, default="mt3_full_plus")
    parser.add_argument("-epv", "--eval-program-vocab", type=str, default=None)
    parser.add_argument("-edv", "--eval-drum-vocab", type=str, default=None)
    parser.add_argument("-etk", "--eval-subtask-key", type=str, default="default")
    parser.add_argument("-t", "--onset-tolerance", type=float, default=0.05)
    parser.add_argument("-os", "--test-octave-shift", type=str2bool, default=False)
    parser.add_argument("-w", "--write-model-output", type=str2bool, default=False)
    parser.add_argument("-pr", "--precision", type=str, default="32")
    parser.add_argument("-st", "--strategy", type=str, default="auto")
    parser.add_argument("-n", "--num-nodes", type=int, default=1)
    parser.add_argument("-g", "--num-gpus", type=str, default="auto")
    parser.add_argument("-wb", "--wandb-mode", type=str, default="disabled")
    parser.add_argument("-debug", "--debug-mode", type=str2bool, default=False)
    parser.add_argument("-tps", "--test-pitch-shift", type=int, default=None)
    args = parser.parse_args(MODEL_ARGS)
    if torch.__version__ >= "1.13":
        torch.set_float32_matmul_precision("high")
    args.epochs = None

    _, _, dir_info, shared_cfg = initialize_trainer(args, stage="test")
    shared_cfg, audio_cfg, model_cfg = update_config(args, shared_cfg, stage="test")

    tm = TaskManager(
        task_name=args.task,
        max_shift_steps=int(shared_cfg["TOKENIZER"]["max_shift_steps"]),
        debug_mode=False,
    )
    model = YourMT3(
        audio_cfg=audio_cfg,
        model_cfg=model_cfg,
        shared_cfg=shared_cfg,
        optimizer=None,
        task_manager=tm,
        eval_subtask_key=args.eval_subtask_key,
        write_output_dir=None,
    ).to(device)
    ckpt = torch.load(dir_info["last_ckpt_path"], map_location=device, weights_only=False)
    state_dict = {k: v for k, v in ckpt["state_dict"].items() if "pitchshift" not in k}
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


def transcribe(audio_path: str, params: dict) -> list:
    """audio → NoteEvent dict のリスト（notes.json schema 1 の notes 相当）。"""
    from collections import Counter

    import soundfile as sf
    import torch
    import torchaudio
    from utils.audio import slice_padded_array
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes

    device = torch.device(params["device"])
    model = load_model(device)

    # torchaudio.load は torchcodec 必須化のため soundfile で読む
    # （docs/YOURMT3_VERIFICATION_2026-07-17.md）
    wav, sr = sf.read(audio_path, dtype="float32", always_2d=True)  # (n, ch)
    audio = torch.from_numpy(wav.T)  # (ch, n)
    audio = torch.mean(audio, dim=0).unsqueeze(0)  # mono (1, n)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    segs = slice_padded_array(
        audio, model.audio_cfg["input_frames"], model.audio_cfg["input_frames"]
    )
    segs = torch.from_numpy(segs.astype("float32")).to(device).unsqueeze(1)

    pred_token_arr, _ = model.inference_file(
        bsz=int(params["batch_size"]), audio_segments=segs
    )

    start_secs_file = [
        model.audio_cfg["input_frames"] * i / model.audio_cfg["sample_rate"]
        for i in range(segs.shape[0])
    ]
    pred_notes_in_file = []
    n_err_cnt = Counter()
    for ch in range(model.task_manager.num_decoding_channels):
        pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped, _, _ = model.task_manager.detokenize_list_batches(
            pred_token_arr_ch, start_secs_file, return_events=True
        )
        pred_notes_ch, n_err = merge_zipped_note_events_and_ties_to_notes(zipped)
        pred_notes_in_file.append(pred_notes_ch)
        n_err_cnt += n_err
    pred_notes = mix_notes(pred_notes_in_file)
    if n_err_cnt:
        print(f"decode errors: {dict(n_err_cnt)}", file=sys.stderr)

    # NoteEvent schema 1 へ変換（ドラムは除外。velocity は 0-1 に正規化、
    # YourMT3 は独立した confidence を出さないため 1.0 固定）
    notes = [
        {
            "onset_sec": float(n.onset),
            "offset_sec": float(n.offset),
            "midi_pitch": int(n.pitch),
            "velocity": float(n.velocity) if n.velocity <= 1 else float(n.velocity) / 127.0,
            "confidence": 1.0,
        }
        for n in pred_notes
        if not n.is_drum
    ]
    notes.sort(key=lambda d: (d["onset_sec"], d["midi_pitch"]))
    return notes


def main() -> int:
    audio_path, out_path, params = parse_params(sys.argv)
    audio_path = os.path.abspath(audio_path)
    out_path = os.path.abspath(out_path)

    # config の save_dir "amt/logs" が cwd 相対のため home へ移動し、
    # YourMT3 のソースを import パスに追加する
    os.chdir(params["home"])
    sys.path.insert(0, os.path.join(params["home"], "amt", "src"))

    notes = transcribe(audio_path, params)

    with open(out_path, "w") as f:
        json.dump({"schema": 1, "notes": notes}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
