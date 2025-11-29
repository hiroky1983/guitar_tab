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
        previous_end = 0.0

        for event in events:
            gap_seconds = max(event.start - previous_end, 0.0)
            gap_beats = gap_seconds * beats_per_second
            if gap_beats > 0.05:
                tokens.append(f"r{quantize_duration(gap_beats)}")

            pitch = open_strings[event.string] + event.fret
            
            # ギター音域外の音をスキップ（MIDI 40-88）
            if pitch < 40 or pitch > 88:
                continue
                
            duration_beats = max(event.end - event.start, 0.0) * beats_per_second
            tokens.append(f"{midi_to_pitch(pitch)}{quantize_duration(duration_beats)}\\{event.string}")

            previous_end = max(previous_end, event.end)

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
            f"  \\tempo 4 = {bpm}\n  "
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
