"""notes → tab 運指割当（コスト最小化 DP）。

NoteEvent 列（notes.json）を標準チューニング EADGBE・カポなし・フレット 0-22 の
（弦, フレット）に割り当てる。v1 `guitartab_transcriber/transcriber.py` の
貪欲割当（開放弦ボーナス・ポジション移動ペナルティ・弦競合回避）を叩き台に、
gtrsnipe のコスト設計（フレット幅・手の移動距離・弦切替ペナルティ）を参考に
時系列 DP として再実装したもの。

アルゴリズム:
1. onset が `chord_window_sec` 以内の音符を和音グループにまとめる。
2. 各グループについて「弦競合なし・フレット幅 <= max_chord_span」の割当候補を
   全列挙し、グループ内コスト（開放弦ボーナス / ハイフレット / フレット幅）の
   小さい順に上位 `beam_width` 件へ絞る。
3. グループ列に対して DP。状態 = 直前グループの割当候補（+ そこまでに確定した
   手のポジション）。遷移コスト = ポジション移動距離 + 弦跳び。
   全開放弦のグループは手のポジションを動かさない（直前のポジションを引き継ぐ）。

音域外ノートの方針（クランプ、除外しない）:
    音域（E2=40 〜 1弦22フレット=86）の外のピッチは、音域に入るまでオクターブ
    単位で移調（クランプ）し、`UserWarning` を発して割当を続行する。
    除外よりリズム・フレーズ構造が保たれるため。TabNote.midi_pitch には
    クランプ後（実際に鳴らす）ピッチが入る。

その他の縮退ケース:
- 和音内で同一ピッチの重複はそのまま別弦への割当を試みる（実際のユニゾンを
  許容）。弦が足りず割り当て不能な音は UserWarning 付きで除外する。

既知の限界（M4 送り）: 遷移コストは音符間の時間差を考慮しない（ゆっくりした
パッセージでも大きなポジション移動に同じペナルティがかかる）。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

from guitartab.transcribe.base import NoteEvent

TAB_SCHEMA_VERSION = 1

# 標準チューニング: 弦番号(1=高音E) → 開放弦の MIDI ノート番号
STANDARD_TUNING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
MAX_FRET = 22
MIN_PITCH = STANDARD_TUNING_MIDI[6]  # E2 = 40
MAX_PITCH = STANDARD_TUNING_MIDI[1] + MAX_FRET  # 86

# コスト重み（グループ内）
OPEN_STRING_BONUS = -1.0  # 開放弦 1 本あたりのボーナス（負 = 優遇）
HIGH_FRET_PENALTY_PER_FRET = 0.3  # 12 フレット超 1 フレットごと
CHORD_SPAN_WEIGHT = 0.7  # 押弦フレットの幅 1 フレットごと
# コスト重み（遷移）
POSITION_MOVE_WEIGHT = 1.0  # 手のポジション移動 1 フレットごと
STRING_JUMP_WEIGHT = 0.15  # 平均弦番号の差 1 本ごと

DEFAULT_CHORD_WINDOW_SEC = 0.05
DEFAULT_MAX_CHORD_SPAN = 4
DEFAULT_BEAM_WIDTH = 12


@dataclass(frozen=True)
class TabNote:
    """tab.json の 1 要素。string は 1(高音e)〜6(低音E)、fret は 0-22。

    midi_pitch は実際に鳴らすピッチ（= 開放弦 MIDI + fret。音域外入力は
    クランプ後の値）。
    """

    onset_sec: float
    duration_sec: float
    string: int
    fret: int
    midi_pitch: int


# ---------------------------------------------------------------------------
# tab.json 保存 / 読込
# ---------------------------------------------------------------------------


def sort_tab(tab: list[TabNote]) -> list[TabNote]:
    """onset → 弦番号の順で安定ソートしたコピーを返す。"""
    return sorted(tab, key=lambda t: (t.onset_sec, t.string))


def save_tab(tab: list[TabNote], path: Path) -> Path:
    """TabNote リストを tab.json 形式で保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": TAB_SCHEMA_VERSION, "tab": [asdict(t) for t in sort_tab(tab)]}
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return path


def load_tab(path: Path) -> list[TabNote]:
    """tab.json を読む。"""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "tab" not in data:
        raise ValueError(f"unrecognized tab.json structure: {path}")
    schema = data.get("schema", TAB_SCHEMA_VERSION)
    if schema != TAB_SCHEMA_VERSION:
        raise ValueError(f"unsupported tab.json schema: {schema} ({path})")
    return [
        TabNote(
            onset_sec=float(d["onset_sec"]),
            duration_sec=float(d["duration_sec"]),
            string=int(d["string"]),
            fret=int(d["fret"]),
            midi_pitch=int(d["midi_pitch"]),
        )
        for d in data["tab"]
    ]


# ---------------------------------------------------------------------------
# 運指割当 DP
# ---------------------------------------------------------------------------

# 割当候補: 音符ごとの (string, fret) タプル列
_Assignment = tuple[tuple[int, int], ...]


def _clamp_pitch(pitch: int) -> int:
    """音域外ピッチをオクターブ移調で音域内にクランプする（警告付き）。"""
    clamped = pitch
    while clamped < MIN_PITCH:
        clamped += 12
    while clamped > MAX_PITCH:
        clamped -= 12
    if clamped != pitch:
        warnings.warn(
            f"midi_pitch {pitch} is outside guitar range "
            f"[{MIN_PITCH}, {MAX_PITCH}]; clamped to {clamped}",
            stacklevel=2,
        )
    return clamped


def _positions_for_pitch(pitch: int) -> list[tuple[int, int]]:
    """ピッチを鳴らせる (string, fret) 候補を列挙する。"""
    positions = []
    for string, open_pitch in STANDARD_TUNING_MIDI.items():
        fret = pitch - open_pitch
        if 0 <= fret <= MAX_FRET:
            positions.append((string, fret))
    return positions


def _intra_cost(assignment: _Assignment) -> float:
    """和音グループ内のコスト（開放弦ボーナス + ハイフレット + フレット幅）。"""
    cost = 0.0
    fretted = [f for _, f in assignment if f > 0]
    for _, fret in assignment:
        if fret == 0:
            cost += OPEN_STRING_BONUS
        elif fret > 12:
            cost += (fret - 12) * HIGH_FRET_PENALTY_PER_FRET
    if fretted:
        cost += (max(fretted) - min(fretted)) * CHORD_SPAN_WEIGHT
    return cost


def _hand_position(assignment: _Assignment) -> float | None:
    """割当の手のポジション（押弦フレットの最小値）。全開放弦なら None。"""
    fretted = [f for _, f in assignment if f > 0]
    return float(min(fretted)) if fretted else None


def _mean_string(assignment: _Assignment) -> float:
    return sum(s for s, _ in assignment) / len(assignment)


def _enumerate_chord_candidates(
    pitches: list[int],
    max_span: int | None,
    beam_width: int,
) -> list[_Assignment]:
    """弦競合のない和音割当候補を列挙し、グループ内コスト昇順に beam_width 件返す。

    max_span を課すと候補ゼロになる場合は制約なしで再試行する。
    """
    options = [_positions_for_pitch(p) for p in pitches]
    results: list[_Assignment] = []

    def dfs(i: int, used: set[int], acc: list[tuple[int, int]]) -> None:
        if i == len(options):
            results.append(tuple(acc))
            return
        for string, fret in options[i]:
            if string in used:
                continue
            if max_span is not None and fret > 0:
                fretted = [f for _, f in acc if f > 0] + [fret]
                if max(fretted) - min(fretted) > max_span:
                    continue
            used.add(string)
            acc.append((string, fret))
            dfs(i + 1, used, acc)
            acc.pop()
            used.discard(string)

    dfs(0, set(), [])
    if not results and max_span is not None:
        return _enumerate_chord_candidates(pitches, None, beam_width)
    results.sort(key=_intra_cost)
    return results[:beam_width]


def _group_chords(
    notes: list[NoteEvent], chord_window_sec: float
) -> list[list[NoteEvent]]:
    """onset が近接する音符を和音グループにまとめる（グループ先頭 onset 基準）。"""
    ordered = sorted(notes, key=lambda n: (n.onset_sec, n.midi_pitch))
    groups: list[list[NoteEvent]] = []
    for note in ordered:
        if groups and note.onset_sec - groups[-1][0].onset_sec <= chord_window_sec:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def _drop_unassignable(
    group: list[NoteEvent], pitches: list[int]
) -> tuple[list[NoteEvent], list[int]]:
    """弦数を超える等で割当不能なグループから、末尾の音を警告付きで除外する。"""
    kept, kept_pitches = list(group), list(pitches)
    while len(kept) > 1 and not _enumerate_chord_candidates(kept_pitches, None, 1):
        dropped = kept.pop()
        kept_pitches.pop()
        warnings.warn(
            f"dropped unassignable note (pitch={dropped.midi_pitch}, "
            f"onset={dropped.onset_sec:.3f}s): no free string in chord",
            stacklevel=2,
        )
    return kept, kept_pitches


def assign_fingering(
    notes: list[NoteEvent],
    *,
    chord_window_sec: float = DEFAULT_CHORD_WINDOW_SEC,
    max_chord_span: int = DEFAULT_MAX_CHORD_SPAN,
    beam_width: int = DEFAULT_BEAM_WIDTH,
) -> list[TabNote]:
    """NoteEvent 列に弦/フレットを割り当てて TabNote 列を返す（時系列 DP）。

    音域外ピッチはオクターブ移調でクランプする（モジュール docstring 参照）。
    """
    if not notes:
        return []

    groups = _group_chords(notes, chord_window_sec)

    # 各グループの割当候補を列挙
    group_notes: list[list[NoteEvent]] = []
    group_candidates: list[list[_Assignment]] = []
    for group in groups:
        pitches = [_clamp_pitch(n.midi_pitch) for n in group]
        candidates = _enumerate_chord_candidates(pitches, max_chord_span, beam_width)
        if not candidates:
            group, pitches = _drop_unassignable(group, pitches)
            candidates = _enumerate_chord_candidates(pitches, max_chord_span, beam_width)
            if not candidates:
                continue
        group_notes.append(group)
        group_candidates.append(candidates)

    if not group_candidates:
        return []

    # DP: 状態 = (割当候補インデックス, 引き継いだ手のポジション)。
    # 全開放弦のグループは手を動かさないため、carried_pos を状態に含めることで
    # 開放弦を跨いだポジション維持もコスト最小に扱える（beam 内で厳密）。
    # layers[g]: {(j, carried_pos): (累積コスト, 前層の状態キー)}
    _State = tuple[int, float | None]
    layer: dict[_State, tuple[float, _State | None]] = {}
    for j, cand in enumerate(group_candidates[0]):
        key = (j, _hand_position(cand))
        cost = _intra_cost(cand)
        if key not in layer or cost < layer[key][0]:
            layer[key] = (cost, None)
    layers: list[dict[_State, tuple[float, _State | None]]] = [layer]

    for g in range(1, len(group_candidates)):
        prev_cands = group_candidates[g - 1]
        prev_layer = layers[-1]
        layer = {}
        for j, cand in enumerate(group_candidates[g]):
            intra = _intra_cost(cand)
            pos = _hand_position(cand)
            mean_s = _mean_string(cand)
            for (pj, prev_pos), (prev_cost, _) in prev_layer.items():
                trans = STRING_JUMP_WEIGHT * abs(mean_s - _mean_string(prev_cands[pj]))
                if pos is not None and prev_pos is not None:
                    trans += POSITION_MOVE_WEIGHT * abs(pos - prev_pos)
                total = prev_cost + trans + intra
                key = (j, pos if pos is not None else prev_pos)
                if key not in layer or total < layer[key][0]:
                    layer[key] = (total, (pj, prev_pos))
        layers.append(layer)

    # バックトラック
    state: _State | None = min(layers[-1], key=lambda k: layers[-1][k][0])
    chosen: list[int] = []
    for g in range(len(layers) - 1, -1, -1):
        assert state is not None
        chosen.append(state[0])
        state = layers[g][state][1]
    chosen.reverse()

    tab: list[TabNote] = []
    for group, candidates, j in zip(group_notes, group_candidates, chosen):
        for note, (string, fret) in zip(group, candidates[j]):
            tab.append(
                TabNote(
                    onset_sec=note.onset_sec,
                    duration_sec=max(0.0, note.offset_sec - note.onset_sec),
                    string=string,
                    fret=fret,
                    midi_pitch=STANDARD_TUNING_MIDI[string] + fret,
                )
            )
    return sort_tab(tab)


__all__ = [
    "TAB_SCHEMA_VERSION",
    "STANDARD_TUNING_MIDI",
    "MAX_FRET",
    "MIN_PITCH",
    "MAX_PITCH",
    "TabNote",
    "assign_fingering",
    "save_tab",
    "load_tab",
    "sort_tab",
]
