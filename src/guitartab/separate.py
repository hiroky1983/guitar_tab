"""Demucs htdemucs_6s によるギターステム分離。

~/myspace/sound_stems/stems.py（動作実績あり）の run_separation を移植したもの。
出力: work/{id}/stems/guitar.wav（他のステムも同じディレクトリに保存する）。

demucs / torch / torchaudio は本パッケージの依存には含めない（外部ツール扱い）。
利用時は実行環境に別途インストールすること:

    uv pip install demucs

インストールされていない場合は transcribe 実行時に手順付きのエラーを出す。
"""

from __future__ import annotations

import sys
from pathlib import Path

DEFAULT_MODEL = "htdemucs_6s"  # 6ステム: drums / bass / guitar / piano / other / vocals
GUITAR_STEM = "guitar"

_INSTALL_HINT = (
    "demucs (and torch/torchaudio) is not installed in this environment.\n"
    "Install it separately, e.g.:\n"
    "    uv pip install demucs\n"
    "demucs is intentionally not a dependency of guitartab (see docs/DESIGN.md)."
)


def separate_stems(
    audio_path: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
) -> dict[str, Path]:
    """audio_path を Demucs で分離し、各ステムを 24bit WAV で out_dir に保存する。

    戻り値: {stem_name: wav_path}。全ステムが既に存在すれば分離をスキップする。
    """
    audio_path = Path(audio_path)
    out_dir = Path(out_dir)

    try:
        import soundfile as sf
        import torch
        import torchaudio
        from demucs.apply import apply_model as demucs_apply
        from demucs.pretrained import get_model
    except ImportError as e:
        raise RuntimeError(_INSTALL_HINT) from e

    print(f"loading model: {model_name}", file=sys.stderr)
    model = get_model(model_name)
    model.eval()

    expected = {name: out_dir / f"{name}.wav" for name in model.sources}
    if not force and all(p.exists() for p in expected.values()):
        print(f"cached: {out_dir}", file=sys.stderr)
        return expected

    print(f"loading audio: {audio_path.name}", file=sys.stderr)
    wav, sr = torchaudio.load(str(audio_path))

    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # モノラル → ステレオ

    print("separating ...", file=sys.stderr)
    with torch.no_grad():
        # shape: (batch, sources, channels, samples) → [0] でバッチ次元を落とす
        sources = demucs_apply(model, wav[None], progress=True)[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, stem_name in enumerate(model.sources):
        path = expected[stem_name]
        # (channels, samples) → (samples, channels) に転置して 24bit WAV で保存
        audio_np = sources[i].numpy().T
        sf.write(str(path), audio_np, model.samplerate, subtype="PCM_24")
        duration = sources[i].shape[-1] / model.samplerate
        print(f"  {stem_name}.wav  ({duration:.1f}s)", file=sys.stderr)

    return expected


def separate_guitar(
    audio_path: Path,
    out_dir: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    force: bool = False,
) -> Path:
    """ギターステムを抽出して out_dir/guitar.wav のパスを返す。

    guitar.wav が既に存在する場合は分離自体をスキップする（force=True で再実行）。
    """
    out_dir = Path(out_dir)
    guitar_path = out_dir / f"{GUITAR_STEM}.wav"
    if guitar_path.exists() and not force:
        print(f"cached: {guitar_path}", file=sys.stderr)
        return guitar_path

    stems = separate_stems(audio_path, out_dir, model_name=model_name, force=force)
    if GUITAR_STEM not in stems:
        raise RuntimeError(
            f"model {model_name} has no '{GUITAR_STEM}' stem "
            f"(sources: {sorted(stems)}). Use a 6-stem model such as {DEFAULT_MODEL}."
        )
    return stems[GUITAR_STEM]
