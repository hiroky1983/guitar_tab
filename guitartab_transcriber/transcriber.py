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
    min_pitch: int = 45  # A2 (5弦開放) - 40から45に変更してベース音を除外
    max_pitch: int = 84  # C6 (1弦20フレット相当) - 88から84に変更して倍音ノイズを削減


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

        # BPM検証と補正
        # 不自然なBPM値を補正（倍テンポ/半テンポの誤検出対策）
        if estimated_bpm > 180:
            print(f"BPM {estimated_bpm} is too high, halving to {estimated_bpm/2}")
            estimated_bpm = estimated_bpm / 2
        elif estimated_bpm < 60:
            print(f"BPM {estimated_bpm} is too low, doubling to {estimated_bpm*2}")
            estimated_bpm = estimated_bpm * 2

        # 指定されたBPMがあれば優先
        final_bpm = bpm if bpm is not None else estimated_bpm
        print(f"Final BPM: {final_bpm:.1f}")

        # 3. 音程解析（ギター）
        print("Transcribing guitar notes...")
        notes, _ = self._transcribe_to_notes(guitar_path)

        # 歪みギターのアタック遅れ補正 (-100ms)
        # 120ms から 100ms に調整: より正確なタイミング補正
        print("Applying latency correction (-0.10s)...")
        for n in notes:
            n.start -= 0.10
            n.end -= 0.10

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

        # 4. パームミュート検出
        # ロック/メタルで頻出するパームミュートを検出してマーク
        print("Detecting palm muted notes...")
        notes = self._detect_palm_mutes(notes)

        # 5. 統合（スナップ）
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

        # 6. ベロシティ正規化（音量のばらつきを整える）
        print("Normalizing note velocities...")
        filtered_notes = self._normalize_velocities(filtered_notes)

        # 7. 音符長の精密化（短すぎる/長すぎる音符を調整）
        print("Refining note durations...")
        filtered_notes = self._refine_durations(filtered_notes, final_bpm)

        print(f"\n--- First 10 Processed Notes ---")
        for i, n in enumerate(filtered_notes[:10]):
            print(f"Note {i}: Start={n.start:.3f}, Pitch={n.pitch}, Vel={n.velocity:.2f}, Dur={n.end-n.start:.3f}")
        print("-------------------------------\n")

        events = self._notes_to_guitar_positions(filtered_notes)

        # 8. ポジションシフトの最適化（後処理）
        print("Optimizing position shifts...")
        events = self._optimize_position_shifts(events)

        return TabResult.from_tab_events(events, bpm=final_bpm)

    def _normalize_velocities(self, notes: List[Note]) -> List[Note]:
        """
        ベロシティ（音量）の正規化
        - 極端に大きい/小さい値を調整
        - より一貫した音量表現にする
        """
        if not notes:
            return notes

        velocities = [n.velocity for n in notes]
        if not velocities:
            return notes

        # 統計値を計算
        mean_vel = sum(velocities) / len(velocities)
        max_vel = max(velocities)
        min_vel = min(velocities)

        # 正規化範囲: 0.3 - 0.9 (極端な値を避ける)
        target_min = 0.3
        target_max = 0.9

        normalized_notes = []
        for n in notes:
            # 線形正規化
            if max_vel > min_vel:
                normalized_vel = target_min + (n.velocity - min_vel) / (max_vel - min_vel) * (target_max - target_min)
            else:
                normalized_vel = (target_min + target_max) / 2

            # クリップ
            normalized_vel = max(target_min, min(target_max, normalized_vel))

            normalized_notes.append(Note(
                start=n.start,
                end=n.end,
                pitch=n.pitch,
                velocity=normalized_vel
            ))

        return normalized_notes

    def _refine_durations(self, notes: List[Note], bpm: float) -> List[Note]:
        """
        音符の長さを精密化
        - 極端に短い/長い音符を調整
        - 音楽的に意味のある長さにする
        """
        if not notes:
            return notes

        beat_interval = 60.0 / bpm
        # 最小長: 32分音符
        min_duration = beat_interval / 8.0
        # 最大長: 全音符
        max_duration = beat_interval * 4.0

        refined_notes = []
        for i, n in enumerate(notes):
            duration = n.end - n.start

            # 次の音符との間隔をチェック
            if i < len(notes) - 1:
                next_note = notes[i + 1]
                gap_to_next = next_note.start - n.end

                # 次の音符まで余裕がない場合、音符を短くする
                if gap_to_next < 0.01:
                    # 重なっている場合は短くする
                    new_duration = max(min_duration, n.end - n.start - 0.02)
                    refined_notes.append(Note(
                        start=n.start,
                        end=n.start + new_duration,
                        pitch=n.pitch,
                        velocity=n.velocity
                    ))
                    continue

            # 長さの調整
            if duration < min_duration:
                new_duration = min_duration
            elif duration > max_duration:
                new_duration = max_duration
            else:
                new_duration = duration

            refined_notes.append(Note(
                start=n.start,
                end=n.start + new_duration,
                pitch=n.pitch,
                velocity=n.velocity
            ))

        return refined_notes

    def _detect_palm_mutes(self, notes: List[Note]) -> List[Note]:
        """
        パームミュート（P.M.）の検出
        特徴:
        - 短い音符 (duration < 0.2s)
        - 中程度〜低いベロシティ (0.2 < velocity < 0.6)
        - 低音弦のピッチ (MIDI 40-55)
        - 連続して出現することが多い
        """
        if not notes:
            return notes

        palm_mute_count = 0

        for i, note in enumerate(notes):
            duration = note.end - note.start

            # パームミュートの条件判定
            is_pm = (
                duration < 0.2 and                    # 短い音符
                0.2 < note.velocity < 0.6 and         # 中程度の音量
                40 <= note.pitch <= 55                 # 低音弦（E2〜G3）
            )

            # 連続性チェック: 前後の音符も同様の特徴を持つか
            if is_pm and i > 0:
                prev_note = notes[i-1]
                time_diff = note.start - prev_note.end
                # 前の音符が近い位置にあり、同様の特徴を持つ場合、確信度を上げる
                if time_diff < 0.3 and 40 <= prev_note.pitch <= 55:
                    is_pm = True

            if is_pm:
                palm_mute_count += 1
                # Note: 将来的にはNote型にpalm_muteフラグを追加する
                # 現時点では検出だけを行い、velocityを若干調整
                note.velocity = max(0.15, note.velocity * 0.9)

        print(f"Detected {palm_mute_count} potential palm muted notes")
        return notes

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
        改善版: 32分音符も含めたより細かいグリッドと、適応的なスナップ閾値
        """
        if not beat_times:
            return notes

        # ビート間隔（秒）
        beat_interval = 60.0 / bpm

        # グリッドの作成: 32分音符単位のサブグリッド
        # ロック曲の高速フレーズに対応するため、より細かいグリッドを使用
        grid_points = []

        # 最初のビートより前も埋める
        start_time = beat_times[0]
        while start_time > 0:
            start_time -= beat_interval
            grid_points.append(start_time)

        for t in beat_times:
            grid_points.append(t)
            # ビート間を8分割（32分音符）
            for i in range(1, 8):
                grid_points.append(t + beat_interval * (i / 8.0))

        # 最後のビート以降も少し埋める
        last_time = beat_times[-1]
        for i in range(1, 20): # 5小節分くらい余分に
            grid_points.append(last_time + beat_interval * i)
            for j in range(1, 8):
                grid_points.append(last_time + beat_interval * i + beat_interval * (j / 8.0))

        grid_points.sort()

        # スナップ閾値: これより離れている場合はスナップしない
        snap_threshold = beat_interval / 16.0  # 16分音符の半分

        snapped = []
        for n in notes:
            # 最も近いグリッド点を探す
            closest_grid = min(grid_points, key=lambda g: abs(g - n.start))

            # 距離をチェック
            distance = abs(closest_grid - n.start)

            # スナップ閾値内であればスナップ、そうでなければ元の時刻を使用
            if distance < snap_threshold:
                snapped_start = closest_grid
            else:
                snapped_start = n.start

            # 音の長さもグリッド単位に調整
            duration = n.end - n.start
            # 32分音符単位に丸める
            thirtysecond_note = beat_interval / 8.0
            quantized_duration = round(duration / thirtysecond_note) * thirtysecond_note
            if quantized_duration < thirtysecond_note:
                quantized_duration = thirtysecond_note

            snapped.append(Note(
                start=snapped_start,
                end=snapped_start + quantized_duration,
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
                onset_threshold=0.5,       # 0.35 -> 0.5: 閾値を上げて確実な音のみ検出（過検出を防ぐ）
                frame_threshold=0.4,       # 0.25 -> 0.4: フレーム検出も厳しく
                minimum_note_length=50.0,  # 25ms -> 50ms: 極端に短い音を除外
                minimum_frequency=100.0,   # 38Hz -> 100Hz: A2(110Hz)付近から検出、ベース音を完全除外
                maximum_frequency=880.0,   # 950Hz -> 880Hz: A5まで、さらに倍音を制限
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
                
            # 倍音除去ロジック - 強化版
            # 低い音順にソート
            group.sort(key=lambda n: n.pitch)

            kept_notes = []
            # 一番低い音は（ベース音として）必ず残す
            root = group[0]
            kept_notes.append(root)

            # 単音の場合が多いので、グループサイズが小さい場合は全て残す
            if len(group) <= 2:
                filtered_notes.extend(group)
                continue

            for i in range(1, len(group)):
                note = group[i]
                is_harmonic = False

                # ルート音との比較
                interval = note.pitch - root.pitch

                # 倍音判定を厳しく - より多くの倍音を除外
                # オクターブ (12, 24) や 完全5度 (7, 19) は倍音の可能性が高い
                if interval in [12, 24, 7, 19]:
                    if note.velocity < root.velocity * 0.9: # 基準を0.8→0.9に厳しく
                        is_harmonic = True

                # 3度 (4, 16) も積極的に除外
                if interval in [4, 16]:
                    if note.velocity < root.velocity * 0.7: # 0.5→0.7に厳しく
                        is_harmonic = True

                # 2度 (2, 14) や6度 (9, 21) など不協和音程も倍音ノイズの可能性
                if interval in [2, 14, 9, 21]:
                    if note.velocity < root.velocity * 0.8:
                        is_harmonic = True

                if not is_harmonic:
                    kept_notes.append(note)

            filtered_notes.extend(kept_notes)

        # 3. 最終的なゴミ掃除 - 改善版 v2
        final_result = []
        for n in filtered_notes:
            duration = n.end - n.start

            # パームミュートの可能性がある音符は緩い基準を適用
            is_potential_palm_mute = (
                0.05 < duration < 0.25 and
                0.2 < n.velocity < 0.6 and
                40 <= n.pitch <= 55
            )

            # 極端に短い音符を除外（ただし、Palm Muteは保護）
            if duration < 0.03 and not is_potential_palm_mute:
                continue

            # 超高音ノイズの除外（ハーモニクスは保護）
            if n.pitch > 77 and n.velocity < 0.25:
                continue

            # 極端に低い音量の音符を除外（パームミュートは保護）
            if n.velocity < 0.15 and not is_potential_palm_mute:
                continue

            final_result.append(n)

        return final_result

    def _detect_simultaneous_notes(self, notes: List[Note], time_window: float = 0.02) -> List[List[Note]]:
        """
        同時発音している音符をグループ化する（和音検出）
        時間窓を20msに縮小 - 過剰な和音検出を防ぐ
        """
        if not notes:
            return []

        groups = []
        current_group = [notes[0]]

        for i in range(1, len(notes)):
            # 前の音符との時間差 - より厳密に判定
            if abs(notes[i].start - current_group[0].start) < time_window:
                current_group.append(notes[i])
            else:
                groups.append(current_group)
                current_group = [notes[i]]

        groups.append(current_group)
        return groups

    def _notes_to_guitar_positions(self, notes: List[Note]) -> list[dict]:
        """
        note列をギターの弦・フレットに割り当てるロジック。
        改善版: 和音検出と運指の連続性を考慮
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
        current_hand_pos = 0
        used_strings_in_chord = set()  # 和音で使用中の弦を記録

        # 同時発音グループを検出
        note_groups = self._detect_simultaneous_notes(notes)

        for group in note_groups:
            used_strings_in_chord.clear()

            for n in group:
                possible_positions = []

                # 1. この音が弾けるすべてのポジションを列挙
                for s, open_pitch in open_strings.items():
                    # 和音内で既に使用されている弦は除外
                    if s in used_strings_in_chord:
                        continue

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
                    # ボーナスを-3→-5に強化して開放弦を最優先
                    open_string_bonus = -5 if fret == 0 else 0

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

                    # 4. 弦の優先度（中音弦を優先 - 5,4,3弦を好む）
                    # 6弦にペナルティ（ベース音を避ける）
                    if string_num == 6:
                        string_preference = 5.0  # 6弦に大きなペナルティ
                    elif string_num in [5, 4, 3]:
                        string_preference = -1.0  # 5,4,3弦にボーナス
                    else:
                        string_preference = 0.0

                    # 5. 和音内での弦の重複を避けるボーナス
                    string_conflict_penalty = 0 if s not in used_strings_in_chord else 100

                    return fret_dist + high_fret_penalty + string_preference + open_string_bonus + string_conflict_penalty

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

                # 和音内で使用した弦を記録
                used_strings_in_chord.add(best_pos["string"])

                # 手の位置を更新
                # 開放弦の場合は手の位置（ポジション）を変えない
                if best_pos["fret"] > 0:
                    current_hand_pos = best_pos["fret"]

        return tab_events

    def _optimize_position_shifts(self, events: list[dict]) -> list[dict]:
        """
        ポジションシフト（手の移動）を最適化
        - 不要な大きなシフトを削減
        - 連続する音符で同じポジションを維持できる場合は維持
        - より滑らかな運指を実現
        """
        if len(events) <= 1:
            return events

        optimized = []

        # 開放弦のMIDIピッチ
        open_strings = {6: 40, 5: 45, 4: 50, 3: 55, 2: 59, 1: 64}

        for i, event in enumerate(events):
            if i == 0:
                optimized.append(event)
                continue

            prev_event = optimized[-1]

            # 同じ音符が連続する場合、同じ弦を使う（トレモロピッキング対応）
            if event["string"] != prev_event["string"]:
                # 前の音符と音程が近い場合、同じポジション付近を維持できるか確認
                pitch_diff = abs((event["fret"] + open_strings[event["string"]]) -
                                (prev_event["fret"] + open_strings[prev_event["string"]]))

                # 音程差が小さい（半音〜全音）場合
                if pitch_diff <= 2:
                    # 同じ弦で弾けるか確認
                    for string, open_pitch in open_strings.items():
                        alt_fret = (event["fret"] + open_strings[event["string"]]) - open_pitch

                        # 前の音符と同じ弦で、かつフレット範囲内
                        if string == prev_event["string"] and 0 <= alt_fret <= 20:
                            # ポジション移動が小さくなる場合は変更
                            if abs(alt_fret - prev_event["fret"]) < abs(event["fret"] - prev_event["fret"]):
                                event = {
                                    "string": string,
                                    "fret": alt_fret,
                                    "start": event["start"],
                                    "end": event["end"]
                                }
                                break

            # 大きなポジションシフト（5フレット以上）を検出して警告
            if event["string"] == prev_event["string"]:
                fret_shift = abs(event["fret"] - prev_event["fret"])
                if fret_shift > 5:
                    # 開放弦を使えば避けられないかチェック
                    for string, open_pitch in open_strings.items():
                        alt_fret = (event["fret"] + open_strings[event["string"]]) - open_pitch
                        if string != event["string"] and alt_fret == 0:
                            # 開放弦で同じ音が出せる
                            event = {
                                "string": string,
                                "fret": 0,
                                "start": event["start"],
                                "end": event["end"]
                            }
                            break

            optimized.append(event)

        return optimized
