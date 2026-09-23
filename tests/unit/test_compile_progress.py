from types import SimpleNamespace

import pytest

from vmr.compiler.batch import compile_dataset
from vmr.core.errors import HarnessError


def _config(tmp_path):
    return SimpleNamespace(
        compile=SimpleNamespace(consecutive_failure_limit=3),
        storage=SimpleNamespace(artifacts=tmp_path / "artifacts", templates=tmp_path),
    )


def _patch_dataset(monkeypatch, videos):
    monkeypatch.setattr(
        "vmr.compiler.batch.load_dataset", lambda path: {"name": "monitor"}
    )
    monkeypatch.setattr("vmr.compiler.batch.select_split", lambda meta, split: "dev")
    monkeypatch.setattr("vmr.compiler.batch.load_videos", lambda *args: dict(videos))
    monkeypatch.setattr(
        "vmr.compiler.batch.write_wikiset",
        lambda *args, **kwargs: SimpleNamespace(
            data={"artifacts": kwargs["artifacts"]}
        ),
    )


def test_compile_prints_a_line_for_each_sealed_video(tmp_path, monkeypatch, capsys):
    _patch_dataset(
        monkeypatch,
        {"v1": {"video_path": "v1.mp4"}, "v2": {"video_path": "v2.mp4"}},
    )
    monkeypatch.setattr(
        "vmr.compiler.batch.compile_video", lambda *args, **kwargs: object()
    )
    result = compile_dataset(_config(tmp_path), tmp_path, "dev", tmp_path / "set.json")
    output = capsys.readouterr().out
    assert list(result.data["artifacts"]) == ["v1", "v2"]
    assert "v1: sealed" in output
    assert "v2: sealed" in output
    assert "Compiling videos:" in output
    assert "2/2" in output


def test_compile_counts_a_failed_video_before_stopping(tmp_path, monkeypatch, capsys):
    _patch_dataset(monkeypatch, {"v1": {"video_path": "v1.mp4"}})

    def fail(*args, **kwargs):
        raise HarnessError("agent timed out")

    monkeypatch.setattr("vmr.compiler.batch.compile_video", fail)
    config = _config(tmp_path)
    config.compile.consecutive_failure_limit = 1
    with pytest.raises(HarnessError, match="circuit stopped"):
        compile_dataset(config, tmp_path, "dev", tmp_path / "set.json")
    output = capsys.readouterr().out
    assert "v1: failed: agent timed out" in output
    assert "1/1" in output
