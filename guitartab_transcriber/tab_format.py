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
