"""guitartab.env.load_dotenv のテスト。"""

import os

from guitartab.env import load_dotenv


def test_load_dotenv_basic(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "\n"
        "HF_TOKEN=hf_dummy123\n"
        'QUOTED="hello world"\n'
        "invalid-line-without-equals\n"
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    loaded = load_dotenv(env_file)

    assert loaded == {"HF_TOKEN": "hf_dummy123", "QUOTED": "hello world"}
    assert os.environ["HF_TOKEN"] == "hf_dummy123"
    assert os.environ["QUOTED"] == "hello world"


def test_load_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from_file\n")
    monkeypatch.setenv("HF_TOKEN", "from_shell")

    load_dotenv(env_file)

    assert os.environ["HF_TOKEN"] == "from_shell"


def test_load_dotenv_missing_file(tmp_path):
    assert load_dotenv(tmp_path / "no_such.env") == {}
