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
        
        # 1. 音源分離（全パート）
        print("Separating audio sources (this may take a while)...")
        separated_paths = self._separate_audio(audio_path)
        guitar_path = separated_paths["guitar"]
        drums_path = separated_paths["drums"]
        print(f"Using separated audio: Guitar={guitar_path}, Drums={drums_path}")
        
        # デバッグ用に保存
        import shutil
        shutil.copy(guitar_path, "debug_guitar.wav")
        shutil.copy(drums_path, "debug_drums.wav")

        # 2. リズム解析（ドラム）
        print("Analyzing rhythm from drums...")
        beat_times, estimated_bpm = self._analyze_rhythm_from_drums(drums_path)
        
        # 指定されたBPMがあれば優先
        final_bpm = bpm if bpm is not None else estimated_bpm
        print(f"Final BPM: {final_bpm}")

        # 3. 音程解析（ギター）
        print("Transcribing guitar notes...")
        notes, _ = self._transcribe_to_notes(guitar_path)

        # 歪みギターのアタック遅れ補正 (-120ms)
        print("Applying latency correction (-0.12s)...")
        for n in notes:
            n.start -= 0.12
            n.end -= 0.12

        # 【強力補正 v3】
        # 倍音補正を改善: より精密なゾーン分けと条件付き補正
        # Zone 1: 5度倍音 (B2~Eb3) -> -7 (Root) ただし、ベロシティが低い場合のみ
        # Zone 2: オクターブ倍音 (E3~E4) -> -12 (Root) より広範囲に適用
        print("Applying harmonic correction v3...")
        for n in notes:
            original = n.pitch
            # 5度倍音の補正 - ベロシティが低い場合のみ適用
            if 46 <= n.pitch <= 51 and n.velocity < 0.7:
                n.pitch -= 7
                print(f"Shifted note {original} -> {n.pitch} (5th -> Root, vel={n.velocity:.2f})")
            # オクターブ倍音の補正 - より積極的に適用
            elif 52 <= n.pitch <= 65:
                n.pitch -= 12
                print(f"Shifted note {original} -> {n.pitch} (Octave -> Root)")

        # 4. 統合（スナップ）
        # ギターのノートを、ドラムのビート（グリッド）に合わせる
        print("Snapping notes to drum beats...")
        snapped_notes = self._snap_notes_to_grid(notes, beat_times, final_bpm)

        # 時間シフト: 最初のドラムビートを 0.0秒（基準）にする
        if beat_times:
            first_beat_time = beat_times[0]
            print(f"Shifting notes by -{first_beat_time:.3f}s (First beat)")
            shifted_notes = []
            for n in snapped_notes:
                shifted_notes.append(Note(
                    start=n.start - first_beat_time,
                    end=n.end - first_beat_time,
                    pitch=n.pitch,
                    velocity=n.velocity
                ))
            snapped_notes = shifted_notes

        # ノイズ除去
        filtered_notes = self._filter_notes(snapped_notes)
        
        print(f"\n--- First 10 Filtered Notes ---")
        for i, n in enumerate(filtered_notes[:10]):
            print(f"Note {i}: Start={n.start:.3f}, Pitch={n.pitch}, Vel={n.velocity:.2f}")
        print("-------------------------------\n")
        
        events = self._notes_to_guitar_positions(filtered_notes)
        return TabResult.from_tab_events(events, bpm=final_bpm)

    def _analyze_rhythm_from_drums(self, drums_path: Path) -> tuple[List[float], float]:
        """
        ドラムトラックからビート（拍）の時刻を検出する。
        """
        import librosa
        y, sr = librosa.load(str(drums_path), sr=self.config.sample_rate)
        
        # オンセット検出（発音タイミング）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # ビートトラッキング
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        print(f"Detected {len(beat_times)} beats. First 10: {beat_times[:10]}")
        
        return list(beat_times), float(tempo)

    def _snap_notes_to_grid(self, notes: List[Note], beat_times: List[float], bpm: float) -> List[Note]:
        """
        ノートの開始時刻を、最も近いビート（またはその分割点）に吸着させる。
        """
        if not beat_times:
            return notes
            
        # ビート間隔（秒）
        beat_interval = 60.0 / bpm
        
        # グリッドの作成: ビート位置だけでなく、16分音符単位のサブグリッドも作る
        grid_points = []
        
        # 最初のビートより前も埋める
        start_time = beat_times[0]
        while start_time > 0:
            start_time -= beat_interval
            grid_points.append(start_time)
            
        for t in beat_times:
            grid_points.append(t)
            # ビート間を4分割（16分音符）
            for i in range(1, 4):
                grid_points.append(t + beat_interval * (i / 4.0))
                
        # 最後のビート以降も少し埋める
        last_time = beat_times[-1]
        for i in range(1, 20): # 5小節分くらい余分に
            grid_points.append(last_time + beat_interval * i)
            for j in range(1, 4):
                grid_points.append(last_time + beat_interval * i + beat_interval * (j / 4.0))
                
        grid_points.sort()
        
        snapped = []
        for n in notes:
            # 最も近いグリッド点を探す
            closest_grid = min(grid_points, key=lambda g: abs(g - n.start))
            
            # 音の長さもグリッド単位に調整
            duration = n.end - n.start
            # 16分音符単位に丸める
            sixteenth_note = beat_interval / 4.0
            quantized_duration = round(duration / sixteenth_note) * sixteenth_note
            if quantized_duration < sixteenth_note:
                quantized_duration = sixteenth_note
                
            snapped.append(Note(
                start=closest_grid,
                end=closest_grid + quantized_duration,
                pitch=n.pitch,
                velocity=n.velocity
            ))
            
        return snapped


    def transcribe_from_youtube(self, url: str, bpm: Optional[float] = None) -> TabResult:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = download_youtube_audio(url, Path(tmpdir))
            return self.transcribe(audio_path, bpm=bpm)

    # === 内部実装 ===
    
    def _separate_audio(self, audio_path: Path) -> dict[str, Path]:
        """
        Demucsを使って音源分離を行い、各トラックのパスを返す。
        リズム解析用にドラム、音程解析用にギター(other)を使用する。
        """
        import subprocess
        import shutil
        
        # 出力ディレクトリ
        out_dir = audio_path.parent / "separated"
        
        # demucsコマンドの実行
        # -n htdemucs: 高性能モデル
        # --two-stems オプションを削除し、全パート（drums, bass, other, vocals）を分離する
        cmd = [
            "demucs",
            "-n", "htdemucs",
            "-o", str(out_dir),
            str(audio_path)
        ]
        
        # demucsがインストールされているか確認
        if shutil.which("demucs") is None:
            print("Warning: 'demucs' command not found. Skipping separation.")
            return {"guitar": audio_path, "drums": audio_path}

        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Demucs failed: {e.stderr.decode()}")
            print("Skipping separation and using original audio.")
            return {"guitar": audio_path, "drums": audio_path}
            
        # 生成されたファイルのパス
        # separated/htdemucs/{filename}/{stem}.wav
        track_name = audio_path.stem
        base_dir = out_dir / "htdemucs" / track_name
        
        guitar_path = base_dir / "other.wav"
        drums_path = base_dir / "drums.wav"
        
        if guitar_path.exists() and drums_path.exists():
            return {"guitar": guitar_path, "drums": drums_path}
        else:
            print(f"Separated files not found at {base_dir}. Using original.")
            return {"guitar": audio_path, "drums": audio_path}

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
                onset_threshold=0.35,      # 0.4 -> 0.35: さらに感度を上げて、ミュート音などを拾いやすくする
                frame_threshold=0.25,      # 0.3 -> 0.25: フレーム検出の感度を上げる
                minimum_note_length=25.0,  # 30ms -> 25ms: より細かい刻みを拾う
                minimum_frequency=38.0,    # 40Hz -> 38Hz: より低音を拾う(E1=41Hz周辺)
                maximum_frequency=950.0,   # 1000Hz -> 950Hz: 倍音対策を強化
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

        # 3. 最終的なゴミ掃除 - 改善版
        final_result = []
        for n in filtered_notes:
            duration = n.end - n.start
            # 極端に短い音符を除外（ただし、Palm Muteの可能性も考慮）
            if duration < 0.03: continue
            # 超高音ノイズの除外基準を緩和（ハーモニクスの可能性）
            if n.pitch > 77 and n.velocity < 0.25: continue
            # 極端に低い音量の音符を除外
            if n.velocity < 0.15: continue
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
                string_num = pos["string"]

                # 1. 開放弦ボーナス（ロック曲では開放弦を積極的に使う）
                open_string_bonus = -3 if fret == 0 else 0

                # 2. フレット移動コスト
                if fret == 0:
                    fret_dist = 0  # 開放弦は移動コストゼロ
                else:
                    # 現在の手の位置との距離
                    if current_hand_pos == 0:
                        # 開放弦から押さえ弦へ: ローポジション(1-5フレット)を優先
                        fret_dist = abs(fret - 3) * 0.5
                    else:
                        fret_dist = abs(fret - current_hand_pos)

                # 3. ハイフレットペナルティ
                # 12フレットを超えると大きなペナルティ
                high_fret_penalty = 0
                if fret > 12:
                    high_fret_penalty = (fret - 12) * 3
                elif fret > 7:
                    # 7フレット以上も少しペナルティ（ローポジション優先）
                    high_fret_penalty = (fret - 7) * 0.5

                # 4. 弦の優先度（低い弦を優先するロック/メタルスタイル）
                string_preference = (7 - string_num) * 0.3  # 6弦が最優先

                return fret_dist + high_fret_penalty + string_preference + open_string_bonus

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
