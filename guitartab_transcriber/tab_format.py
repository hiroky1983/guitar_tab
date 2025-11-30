import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

@dataclass
class TabEvent:
    string: int   # 1〜6
    fret: int
    start: float  # 秒
    end: float    # 秒

@dataclass
class TabResult:
    events: List[TabEvent]
    bpm: Optional[float] = None

    @classmethod
    def from_tab_events(cls, events: list[dict], bpm: Optional[float] = None) -> "TabResult":
        return cls(
            events=[
                TabEvent(
                    string=e["string"],
                    fret=e["fret"],
                    start=e["start"],
                    end=e["end"],
                )
                for e in events
            ],
            bpm=bpm,
        )

    def to_text(self) -> str:
        """
        超簡易ASCIIタブ（一旦「時間方向は雑に均等」でOKな例）
        """
        if not self.events:
            return "(no notes)"

        # 弦ごとにソートして並べる（簡易版）
        lines = {s: [] for s in range(1, 7)}
        # 時間順に並べる
        sorted_events = sorted(self.events, key=lambda e: e.start)

        for e in sorted_events:
            for s in range(1, 7):
                if s == e.string:
                    lines[s].append(str(e.fret).rjust(2, "-"))
                else:
                    lines[s].append("--")

        # ASCIIタブ生成（上が1弦になるように）
        result_lines = []
        for s in sorted(lines.keys()):
            row = "".join(lines[s])
            result_lines.append(f"{s}|{row}")
        return "\n".join(result_lines)

    def to_json(self) -> dict:
        return {
            "events": [
                {
                    "string": e.string,
                    "fret": e.fret,
                    "start": e.start,
                    "end": e.end,
                }
                for e in self.events
            ],
            "bpm": self.bpm,
        }

    def to_matplotlib(self, save_path: Optional[str] = None):
        """
        簡易TAB描画: 6本の弦にフレット番号をテキスト描画
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))

        strings = [1, 2, 3, 4, 5, 6]

        # 弦の線
        for s in strings:
            ax.hlines(y=s, xmin=0, xmax=1, linewidth=1)

        if self.events:
            max_time = max(e.start for e in self.events) or 1.0
        else:
            max_time = 1.0

        for e in self.events:
            x = e.start / max_time
            y = e.string
            ax.text(x, y, str(e.fret), ha="center", va="center")

        ax.set_ylim(0.5, 6.5)
        ax.set_yticks(strings)
        ax.set_yticklabels([f"Str {s}" for s in strings])
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_title("Guitar TAB (simplified)")

        plt.tight_layout()

        if save_path:
            file_suffix = Path(save_path).suffix.lower()
            # matplotlib は拡張子で自動判別するが、明示的な format 指定も許可
            fmt = file_suffix.lstrip(".") if file_suffix else None
            plt.savefig(save_path, format=fmt)
            plt.close(fig)
        else:
            plt.show()

    def to_svg(self, save_path: str = "result.svg"):
        """
        PNG ではなく SVG 形式でTABを出力したい場合のヘルパー。
        """
        self.to_matplotlib(save_path=save_path)

    # === LilyPond 出力 ===

    def to_lilypond(
        self,
        ly_path: str | Path = "result.ly",
        *,
        title: str = "Guitar TAB",
        compile_output: Optional[str | Path] = None,
        lilypond_executable: str = "lilypond",
    ) -> Path:
        """
        TAB から LilyPond 記法（.ly）を生成する。`compile_output` を指定した場合のみ、
        外部ツールの LilyPond CLI を呼び出して PDF / SVG / PNG をビルドする（外部依存）。

        Parameters
        ----------
        ly_path: str | Path
            出力する .ly ファイルのパス。
        title: str
            スコアのタイトル。
        compile_output: str | Path, optional
            `None` の場合は .ly ファイルを書き出すだけ（ライブラリの責務）。
            `"score.svg"` のようにファイル名を指定すると、拡張子に応じて LilyPond を実行し
            SVG / PNG / PDF を生成する（LilyPond 本体が担当）。
        lilypond_executable: str
            実行する lilypond コマンド名。
        """

        ly_path = Path(ly_path)

        if not self.events:
            raise ValueError("No events to export as LilyPond score.")

        lilypond_source = self._build_lilypond_source(title=title)
        ly_path.write_text(lilypond_source, encoding="utf-8")

        if compile_output is None:
            return ly_path

        compile_path = Path(compile_output)
        output_format = compile_path.suffix.lower().lstrip(".")

        format_flag: list[str]
        if output_format == "svg":
            format_flag = ["--svg"]
        elif output_format == "png":
            format_flag = ["--png"]
        elif output_format == "pdf":
            # PDF はデフォルト
            format_flag = []
        else:
            raise ValueError("compile_output must end with .svg, .png, or .pdf")

        output_stem = str(compile_path.with_suffix(""))

        if shutil.which(lilypond_executable) is None:
            raise FileNotFoundError(
                f"LilyPond executable '{lilypond_executable}' not found in PATH; "
                "install LilyPond or update the executable path."
            )

        subprocess.run(
            [lilypond_executable, *format_flag, "-o", output_stem, str(ly_path)],
            check=True,
        )

        return compile_path

    # === 内部ヘルパー ===

    def _build_lilypond_source(self, *, title: str) -> str:
        events = sorted(self.events, key=lambda e: e.start)
        bpm = self.bpm or 120
        beats_per_second = bpm / 60

        open_strings = {
            6: 40,
            5: 45,
            4: 50,
            3: 55,
            2: 59,
            1: 64,
        }

        duration_table = [
            (4.0, "1"),
            (2.0, "2"),
            (1.0, "4"),
            (0.5, "8"),
            (0.25, "16"),
            (0.125, "32"),
        ]

        def quantize_duration(beats: float) -> str:
            beats = max(beats, 0.125)
            return min(duration_table, key=lambda d: abs(d[0] - beats))[1]

        def midi_to_pitch(midi: int) -> str:
            note_names = ["c", "cis", "d", "ees", "e", "f", "fis", "g", "gis", "a", "bes", "b"]
            name = note_names[midi % 12]
            # LilyPondの 'c' は C3 (MIDI 48)
            octave = (midi - 48) // 12
            if octave > 0:
                return name + "'" * octave
            if octave < 0:
                return name + "," * abs(octave)
            return name

        tokens: list[str] = []
        # 量子化の最小単位（1拍を何分割するか）
        # 4: 16分音符, 3: 3連符, 6: 6連符
        # ここでは 12 (3と4の公倍数) を基準に考えると計算しやすいが、
        # シンプルに「最も近いグリッド」を選ぶ方式にする。
        
        def quantize_beats(beats: float) -> float:
            """
            拍数を音楽的なグリッドに吸着させる。
            リズムを安定させるため、強制的に「16分音符 (0.25)」グリッドのみを使用する。
            3連符などは一旦除外。
            """
            grid = 0.25
            return round(beats / grid) * grid

        # 1. 全イベントを量子化し、開始時刻でグループ化（和音対応）
        quantized_events = []
        for event in events:
            start_beats = event.start * beats_per_second
            end_beats = event.end * beats_per_second
            
            q_start = quantize_beats(start_beats)
            q_end = quantize_beats(end_beats)
            
            if q_end <= q_start:
                q_end = q_start + 0.125 # 最低長
            
            quantized_events.append({
                "start": q_start,
                "end": q_end,
                "event": event
            })
            
        # 開始時刻でソート
        quantized_events.sort(key=lambda x: x["start"])
        
        # 隙間埋め（Legato化）
        # ロックのリフでは音を繋げて弾くことが多いので、短い隙間は埋める
        for i in range(len(quantized_events) - 1):
            curr = quantized_events[i]
            next_evt = quantized_events[i+1]
            
            gap = next_evt["start"] - curr["end"]
            
            # 隙間があり、かつそれが大きすぎない（1拍未満）場合
            if 0 < gap < 1.0:
                # 前の音を伸ばす
                curr["end"] = next_evt["start"]
        
        # 同じ開始時刻のイベントをまとめる
        grouped_events = []
        if quantized_events:
            current_group = [quantized_events[0]]
            for i in range(1, len(quantized_events)):
                prev = current_group[-1]
                curr = quantized_events[i]
                
                # 開始時刻がほぼ同じなら同じグループ（和音）とみなす
                if abs(curr["start"] - prev["start"]) < 0.01:
                    current_group.append(curr)
                else:
                    grouped_events.append(current_group)
                    current_group = [curr]
            grouped_events.append(current_group)

        tokens: list[str] = []
        previous_end_beats = 0.0

        for group in grouped_events:
            # グループ内の代表時刻（すべて同じはず）
            start_beats = group[0]["start"]
            # 終了時刻はグループ内で最大のものを選ぶ（和音全体の長さ）
            end_beats = max(e["end"] for e in group)
            
            # 前の音との隙間（休符）
            gap = start_beats - previous_end_beats
            
            # ロックのリフでは、短い休符はノイズや検出漏れの可能性が高いので、
            # 前の音を伸ばして埋めてしまう（レガート化）
            # ただし、長すぎる休符（1拍以上など）は意図的なブレイクとして残す
            if 0 < gap < 1.0:
                # 前のトークン（音符）を探して、長さを修正するのは難しいので、
                # ここでは「休符を出力しない」ことで対応するが、
                # LilyPondは時間を厳密に管理するので、gapを埋めるための「見えない音」か
                # あるいは前の音の長さを計算し直す必要がある。
                
                # 簡易的なアプローチ:
                # gapが小さいなら、今回の音の開始位置を前にずらす（食わせる）ことはできない（順序が変わるため）
                # なので、「休符トークンを追加しない」だけでは時間がズレてしまう。
                
                # 正しいアプローチ:
                # quantized_events の時点で、前の音の end を次の音の start まで伸ばしておくべき。
                pass
            
            if gap > 0.05:
                # ここで休符を入れるかどうか判定
                # 16分音符(0.25)以上の隙間なら休符を入れるが、
                # 今回は「休符NG」という要望なので、極力埋めたい。
                # しかし、ここで埋める処理をするより、前段の quantized_events 作成時に
                # 「隙間を埋める」処理をした方が安全。
                tokens.append(f"r{quantize_duration(gap)}")
            
            # 音価
            duration = end_beats - start_beats
            dur_str = quantize_duration(duration)
            
            # 音符の生成
            # 和音かどうかで分岐
            valid_notes = []
            for item in group:
                e = item["event"]
                pitch = open_strings[e.string] + e.fret
                if 40 <= pitch <= 88:
                    valid_notes.append((pitch, e.string))
            
            if not valid_notes:
                continue
                
            if len(valid_notes) == 1:
                # 単音
                p, s = valid_notes[0]
                tokens.append(f"{midi_to_pitch(p)}{dur_str}\\{s}")
            else:
                # 和音 < c e g >4 のような形式
                # TAB譜では弦指定が必要: < c\5 e\4 g\3 >4
                chord_content = " ".join([f"{midi_to_pitch(p)}\\{s}" for p, s in valid_notes])
                tokens.append(f"<{chord_content}>{dur_str}")

            previous_end_beats = end_beats

        token_lines: list[str] = []
        line: list[str] = []
        for token in tokens:
            line.append(token)
            if len(line) >= 8:
                token_lines.append(" ".join(line))
                line = []
        if line:
            token_lines.append(" ".join(line))

        music_block = "\n  ".join(token_lines)

        return (
            "\\version \"2.24.0\"\n"
            f"\\header {{ title = \"{title}\" }}\n\n"
            "music = {\n"
            f"  \\tempo 4 = {int(round(bpm))}\n  "
            f"{music_block}\n"
            "}\n\n"
            "\\score {\n"
            "  <<\n"
            "    \\new Staff { \\clef \"treble\" \\music }\n"
            "    \\new TabStaff { \\clef \"moderntab\" \\tabFullNotation \\music }\n"
            "  >>\n"
            "  \\layout { }\n"
            "  \\midi { }\n"
            "}\n"
        )
