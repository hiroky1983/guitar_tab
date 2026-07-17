"""プロジェクト直下の .env を環境変数に読み込む（依存ライブラリなし）。

HF_TOKEN 等のシークレットをシェル設定に書かずリポジトリ内 .env（gitignore 済み）で
管理するためのもの。既に設定済みの環境変数は上書きしない。
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """path（デフォルト: カレントディレクトリの .env）を読み os.environ に反映する。

    形式: `KEY=VALUE`（行頭 #・空行は無視、値の前後の引用符は除去）。
    既存の環境変数は上書きしない。読み込んだ値の dict を返す。
    """
    env_path = path if path is not None else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded
