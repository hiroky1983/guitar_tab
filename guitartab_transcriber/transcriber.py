import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, List

import librosa

from .tab_format import TabResult
from .types import Note
from .youtube import download_youtube_audio


@dataclass
class TranscriptionConfig:
    tuning: Literal["E_standard", "Drop_D"] = "E_standard"
    sample_rate: int = 44100
    min_pitch: int = 40
    max_pitch: int = 88


class Transcriber:
    def __init__(self, config: Optional[TranscriptionConfig] = None):
        self.config = config or TranscriptionConfig()

    # === 公開API ===

    def transcribe(self, audio_path: str | Path) -> TabResult:
        audio_path = Path(audio_path)
        notes = self._transcribe_to_notes(audio_path)
        tab_events = self._notes_to_guitar_positions(notes)
        return TabResult.from_tab_events(tab_events)

    def transcribe_from_youtube(self, url: str) -> TabResult:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_youtube_audio(url, Path(tmpdir))
            return self.transcribe(audio_path)

    # === 内部実装 ===

    def _load_audio(self, audio_path: Path):
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        return y, sr

    def _transcribe_to_notes(self, audio_path: Path) -> List[Note]:
        """
        ここが「AI部分」。
        Basic Pitchなどのモデルで音声→ノート列に変換する。
        """
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # predict returns: model_output, midi_data, note_events
        # note_events is a list of (start, end, pitch, amplitude, pitch_bends)
        #
        # basic_pitch can emit verbose debug information to stdout/stderr. Redirect
        # both streams during inference to keep CLI output focused on the tab
        # results.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            _, _, note_events = predict(
                str(audio_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
            )

        notes: List[Note] = []
        for start, end, pitch, velocity, _ in note_events:
            if pitch < self.config.min_pitch or pitch > self.config.max_pitch:
                continue
            notes.append(
                Note(
                    start=float(start),
                    end=float(end),
                    pitch=int(pitch),
                    velocity=float(velocity),
                )
            )
        return notes

    def _notes_to_guitar_positions(self, notes: List[Note]) -> list[dict]:
        """
        note列をギターの弦・フレットに割り当てるロジック。
        ここはMVP用に「一番低い弦で弾けるポジションを選ぶ」だけの簡易版。
        チューニングや運指最適化は今後拡張。
        """
        # E標準の開放弦のMIDI: 6弦E2=40, 5弦A2=45, 4弦D3=50, 3弦G3=55, 2弦B3=59, 1弦E4=64
        open_strings = {
            6: 40,
            5: 45,
            4: 50,
            3: 55,
            2: 59,
            1: 64,
        }

        tab_events: list[dict] = []
        
        # 運指決定のための状態変数
        # 初期位置はローポジション（例: 5フレット付近）を想定、あるいは0
        current_hand_pos = 0

        for n in notes:
            possible_positions = []

            # 1. この音が弾けるすべてのポジションを列挙
            for s, open_pitch in open_strings.items():
                fret = n.pitch - open_pitch
                if 0 <= fret <= 20:  # 20フレットまで
                    possible_positions.append({"string": s, "fret": fret})

            if not possible_positions:
                continue

            # 2. 最適なポジションを選択
            # コスト関数: 人間が弾きやすい運指を選ぶ
            
            def calculate_cost(pos):
                fret = pos["fret"]
                
                # 1. フレット移動コスト
                # 開放弦(0)はどこからでもアクセスしやすいので移動コストを低く（0）みなす
                if fret == 0:
                    fret_dist = 0
                else:
                    # 現在の手の位置との距離
                    # ただし、current_hand_pos が 0 の場合（直前が開放弦だった場合）、
                    # そのさらに前の「押弦していた位置」を基準にするのが理想だが、
                    # ここではシンプルに「0からの距離」になってしまうのを防ぐため、
                    # 0の場合は「移動なし」とみなすか、あるいはデフォルト位置（例えば5フレット）との距離にする
                    if current_hand_pos == 0:
                        fret_dist = 0 # 簡易的にコスト0とする
                    else:
                        fret_dist = abs(fret - current_hand_pos)

                # 2. ハイフレットペナルティ
                # 基本的にローポジション〜ミドルポジションを優先する
                # 12フレットを超えるとペナルティを付与
                high_fret_penalty = 0
                if fret > 12:
                    high_fret_penalty = (fret - 12) * 2
                
                # 3. 弦の優先度（オプション）
                # 低音弦（太い弦）の方が太い音がしてギターらしい場合が多いが、
                # ここではフレット移動のしやすさを最優先する
                
                return fret_dist + high_fret_penalty

            best_pos = min(possible_positions, key=calculate_cost)

            # 選んだポジションを採用
            tab_events.append(
                {
                    "string": best_pos["string"],
                    "fret": best_pos["fret"],
                    "start": n.start,
                    "end": n.end,
                }
            )
            
            # 手の位置を更新
            # 開放弦の場合は手の位置（ポジション）を変えない
            if best_pos["fret"] > 0:
                current_hand_pos = best_pos["fret"]

        return tab_events
