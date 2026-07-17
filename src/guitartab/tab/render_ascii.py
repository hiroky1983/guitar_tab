"""ASCII tab レンダラ。

v1 `guitartab_transcriber/tab_format.py` の to_text() を叩き台に再実装。
6 行（e|B|G|D|A|E|）形式で、時間軸を `time_step_sec` ごとの等間隔カラムに
離散化して並べる（小節割り・リズム量子化は M4 スコープでここでは行わない）。

- 同時発音（onset が同一カラムに落ちる音）は同一カラムに縦に並ぶ。
- カラム幅は「そのカラムの最大フレット桁数 + 1」で可変（2 桁フレットでも
  行間で桁がずれない）。空セルは '-' で埋める。
- `line_width`（デフォルト 80 桁）を超える場合は複数段に折り返す
  （段の間は空行）。
- 同じ弦の複数音が同一カラムに落ちた場合は先の音を残し後の音を捨てる
  （既知の限界。M4 のリズム量子化で解消予定）。
"""

from __future__ import annotations

from guitartab.tab.fingering import TabNote

DEFAULT_TIME_STEP_SEC = 0.125
DEFAULT_LINE_WIDTH = 80

# 弦番号 → 行ラベル（上が 1 弦 = 高音 e）
STRING_LABELS = {1: "e", 2: "B", 3: "G", 4: "D", 5: "A", 6: "E"}


def render_ascii(
    tab: list[TabNote],
    *,
    time_step_sec: float = DEFAULT_TIME_STEP_SEC,
    line_width: int = DEFAULT_LINE_WIDTH,
) -> str:
    """TabNote 列を ASCII tab 文字列にする（末尾改行なし）。"""
    if not tab:
        return "(no notes)"
    if time_step_sec <= 0:
        raise ValueError(f"time_step_sec must be positive: {time_step_sec}")

    # カラム番号 → {弦: フレット}
    grid: dict[int, dict[int, int]] = {}
    for note in sorted(tab, key=lambda t: (t.onset_sec, t.string)):
        col = int(round(note.onset_sec / time_step_sec))
        cells = grid.setdefault(col, {})
        if note.string not in cells:  # 衝突は先勝ち（モジュール docstring 参照）
            cells[note.string] = note.fret

    n_cols = max(grid) + 1
    col_widths = [
        max((len(str(f)) for f in grid.get(c, {}).values()), default=1) + 1
        for c in range(n_cols)
    ]

    # 折り返し: プレフィックス "e|" (2) + カラム + 末尾 "|" (1)
    budget = max(line_width - 3, max(col_widths))
    blocks: list[list[int]] = [[]]  # 各段のカラム番号リスト
    used = 0
    for c in range(n_cols):
        if blocks[-1] and used + col_widths[c] > budget:
            blocks.append([])
            used = 0
        blocks[-1].append(c)
        used += col_widths[c]

    rendered_blocks: list[str] = []
    for cols in blocks:
        lines = []
        for string in range(1, 7):
            cells = []
            for c in cols:
                fret = grid.get(c, {}).get(string)
                text = "" if fret is None else str(fret)
                cells.append(text.ljust(col_widths[c], "-"))
            lines.append(f"{STRING_LABELS[string]}|{''.join(cells)}|")
        rendered_blocks.append("\n".join(lines))
    return "\n\n".join(rendered_blocks)


__all__ = ["render_ascii", "DEFAULT_TIME_STEP_SEC", "DEFAULT_LINE_WIDTH", "STRING_LABELS"]
