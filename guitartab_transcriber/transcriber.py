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

    def transcribe(self, audio_path: str | Path, bpm: Optional[float] = None) -> TabResult:
        audio_path = Path(audio_path)
        
        # 音源分離（ギターパートの抽出）
        print("Separating audio sources (this may take a while)...")
        guitar_audio_path = self._separate_audio(audio_path)
        print(f"Using separated audio: {guitar_audio_path}")
        
        # デバッグ用に分離された音声を保存
        debug_path = Path("debug_guitar.wav")
        import shutil
        shutil.copy(guitar_audio_path, debug_path)
        print(f"Saved separated guitar audio to: {debug_path.absolute()}")

        notes, estimated_bpm = self._transcribe_to_notes(guitar_audio_path)
        
        # 指定されたBPMがあれば優先、なければ推定値を使用
        final_bpm = bpm if bpm is not None else estimated_bpm
        print(f"Final BPM: {final_bpm}")

        # ノイズ除去（最低限のフィルタのみ残す）
        notes = self._filter_notes(notes)
        events = self._notes_to_guitar_positions(notes)
        return TabResult.from_tab_events(events, bpm=final_bpm)

    def transcribe_from_youtube(self, url: str, bpm: Optional[float] = None) -> TabResult:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_youtube_audio(url, Path(tmpdir))
            return self.transcribe(audio_path, bpm=bpm)

    # === 内部実装 ===
    
    def _separate_audio(self, audio_path: Path) -> Path:
        """
        Demucsを使って音源分離を行い、ギターが含まれる 'other' トラックのパスを返す。
        """
        import subprocess
        import shutil
        
        # 出力ディレクトリ
        out_dir = audio_path.parent / "separated"
        
        # demucsコマンドの実行
        # -n htdemucs: 高性能モデル
        # --two-stems=other: other（ギター含む）とそれ以外に分ける（高速化）
        cmd = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems", "other",
            "-o", str(out_dir),
            str(audio_path)
        ]
        
        # demucsがインストールされているか確認
        if shutil.which("demucs") is None:
            print("Warning: 'demucs' command not found. Skipping separation.")
            return audio_path

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Demucs failed: {e.stderr.decode()}")
            print("Skipping separation and using original audio.")
            return audio_path
            
        # 生成されたファイルのパス
        # separated/htdemucs/{filename}/other.wav
        track_name = audio_path.stem
        separated_path = out_dir / "htdemucs" / track_name / "other.wav"
        
        if separated_path.exists():
            return separated_path
        else:
            print(f"Separated file not found at {separated_path}. Using original.")
            return audio_path

    def _load_audio(self, audio_path: Path):
        y, sr = librosa.load(audio_path, sr=self.config.sample_rate, mono=True)
        return y, sr

    def _transcribe_to_notes(self, audio_path: Path) -> tuple[List[Note], float]:
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
        # basic_pitch can emit verbose debug information to stdout/stderr. Redirect
        # both streams during inference to keep CLI output focused on the tab
        # results.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(
            devnull
        ), contextlib.redirect_stderr(devnull):
            _, _, note_events = predict(
                str(audio_path),
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.5,       # 0.6 -> 0.5: 標準に戻す（拾い漏れ防止）
                frame_threshold=0.3,       # 0.4 -> 0.3: 標準に戻す
                minimum_note_length=50.0,  # 80ms -> 50ms: 速いパッセージに対応
            )

        # BPM推定 (librosa)
        import librosa
        y, sr = librosa.load(str(audio_path), sr=self.config.sample_rate)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        estimated_bpm = int(round(float(tempo)))
        print(f"Estimated BPM: {estimated_bpm}")

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
            
        # 時間順にソート
        notes.sort(key=lambda n: n.start)

        # デバッグ: 最初の10音を表示
        print("\n--- First 10 detected notes (Sorted) ---")
        for i, n in enumerate(notes[:10]):
            print(f"Note {i}: Start={n.start:.3f}, Pitch={n.pitch}, Vel={n.velocity:.2f}")
        print("-------------------------------------\n")
            
        return notes, float(estimated_bpm)

    def _filter_notes(self, notes: List[Note]) -> List[Note]:
        """
        AIが検出したノートからノイズを除去し、ギターらしい演奏に整理する。
        特に「倍音ノイズ」の除去に注力する。
        """
        # 1. 時間順にソート
        notes.sort(key=lambda n: n.start)
        
        # 2. グループ化（同時発音）
        groups = []
        if not notes:
            return []
            
        current_group = [notes[0]]
        for i in range(1, len(notes)):
            if abs(notes[i].start - current_group[0].start) < 0.05:
                current_group.append(notes[i])
            else:
                groups.append(current_group)
                current_group = [notes[i]]
        groups.append(current_group)
        
        filtered_notes = []
        
        for group in groups:
            if len(group) == 1:
                filtered_notes.append(group[0])
                continue
                
            # 倍音除去ロジック
            # 低い音順にソート
            group.sort(key=lambda n: n.pitch)
            
            kept_notes = []
            # 一番低い音は（ベース音として）必ず残す
            root = group[0]
            kept_notes.append(root)
            
            for i in range(1, len(group)):
                note = group[i]
                is_harmonic = False
                
                # ルート音との比較
                interval = note.pitch - root.pitch
                
                # オクターブ (12, 24) や 完全5度 (7, 19) は倍音の可能性が高い
                # 特に音量がルートより小さい場合はノイズとみなす
                if interval in [12, 24, 7, 19]:
                    if note.velocity < root.velocity * 0.8: # ルートより明らかに弱い
                        is_harmonic = True
                
                # 3度 (4, 16) も歪みで出やすいが、和音の構成音かもしれないので慎重に
                # ここでは「非常に弱い」場合のみ消す
                if interval in [4, 16]:
                    if note.velocity < root.velocity * 0.5:
                        is_harmonic = True

                if not is_harmonic:
                    kept_notes.append(note)
            
            filtered_notes.extend(kept_notes)

        # 3. 最終的なゴミ掃除
        final_result = []
        for n in filtered_notes:
            duration = n.end - n.start
            if duration < 0.05: continue
            if n.pitch > 75 and n.velocity < 0.3: continue # 超高音ノイズ
            final_result.append(n)
            
        return final_result

    def _notes_to_guitar_positions(self, notes: List[Note]) -> list[dict]:
        """
        note列をギターの弦・フレットに割り当てるロジック。
        ここはMVP用に「一番低い弦で弾けるポジションを選ぶ」だけの簡易版。
        チューニングや運指最適化は今後拡張。
        """
        if not notes:
            return []

        # リズム補正: 最初の音を 0.0秒（小節の頭）に合わせる
        # これにより、曲の開始位置によるズレを解消する
        first_start = notes[0].start
        print(f"Shifting all notes by -{first_start:.3f}s to align start.")
        
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
            # 時間シフト
            shifted_start = n.start - first_start
            shifted_end = n.end - first_start
            
            if shifted_start < 0: shifted_start = 0
            if shifted_end < 0: shifted_end = 0.1

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
                
                return fret_dist + high_fret_penalty

            best_pos = min(possible_positions, key=calculate_cost)

            # 選んだポジションを採用
            tab_events.append(
                {
                    "string": best_pos["string"],
                    "fret": best_pos["fret"],
                    "start": shifted_start,
                    "end": shifted_end,
                }
            )
            
            # 手の位置を更新
            # 開放弦の場合は手の位置（ポジション）を変えない
            if best_pos["fret"] > 0:
                current_hand_pos = best_pos["fret"]

        return tab_events
