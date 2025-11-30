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
            # コスト関数: abs(フレット - 現在の手の位置) が小さいものを選ぶ
            # 同じフレット距離なら、弦の移動が少ない方や、特定の弦を優先するなどの重み付けも可能だが、
            # まずはシンプルに「フレット移動距離の最小化」を行う。
            
            def calculate_cost(pos):
                fret_distance = abs(pos["fret"] - current_hand_pos)
                # オプション: 開放弦(0フレット)は移動コストを低く見積もるなどの調整も可
                return fret_distance

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
            
            # 手の位置を更新（ただし開放弦の場合は手の位置を変えない、などの工夫もアリだが一旦単純更新）
            if best_pos["fret"] > 0:
                current_hand_pos = best_pos["fret"]

        return tab_events
