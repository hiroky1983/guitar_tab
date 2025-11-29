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

        for n in notes:
            best_string = None
            best_fret = None

            for s, open_pitch in open_strings.items():
                fret = n.pitch - open_pitch
                if 0 <= fret <= 20:  # 一旦20フレットまでとしておく
                    # 「できるだけ太い弦（番号が大きい）を優先」する例
                    if best_string is None or s > best_string:
                        best_string = s
                        best_fret = fret

            if best_string is None:
                # どの弦でも弾けない → スキップ
                continue

            tab_events.append(
                {
                    "string": best_string,
                    "fret": best_fret,
                    "start": n.start,
                    "end": n.end,
                }
            )

        return tab_events
