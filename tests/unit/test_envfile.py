import os
from vmr.core.envfile import load_env_file


def test_env_file_fills_missing_variables_without_overriding(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "\n".join(
            [
                "# comment",
                "AGENT_MODEL=from-file",
                "export AGENT_API_KEY='secret'",
                'AGENT_BASE_URL="https://example.test/v1"',
                "ALREADY=from-file",
                "not a line",
                "1BAD=no",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ALREADY", "from-env")
    keys = ("AGENT_MODEL", "AGENT_API_KEY", "AGENT_BASE_URL")
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        load_env_file(path)
        assert os.environ["AGENT_MODEL"] == "from-file"
        assert os.environ["AGENT_API_KEY"] == "secret"
        assert os.environ["AGENT_BASE_URL"] == "https://example.test/v1"
        assert os.environ["ALREADY"] == "from-env"
        assert "1BAD" not in os.environ
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_missing_env_file_is_ignored(tmp_path):
    load_env_file(tmp_path / ".env")


def test_cli_loads_dotenv_from_its_working_directory(tmp_path, monkeypatch):
    import pytest
    from vmr.cli import main

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("AGENT_MODEL=cli-fixture\n")
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    assert os.environ["AGENT_MODEL"] == "cli-fixture"
    # The autouse fixture restores direct os.environ writes after this test.
