"""No model API calls. Opt in with VMR_TEST_DOCKER=1 after building the image."""
import os
import subprocess
from pathlib import Path

import pytest

from agents.runner import DockerRunner


pytestmark = pytest.mark.skipif(os.environ.get("VMR_TEST_DOCKER") != "1", reason="Docker integration is opt-in")


class ContainerProbeRunner(DockerRunner):
    def agent_command(self):
        return ["python3", "-c", '''
import os
from pathlib import Path
assert os.getuid() == 1000
assert not Path("/var/run/docker.sock").exists()
assert sorted(p.name for p in Path("/home/node").iterdir()) == []
for path in ("/workspace/wiki/wiki.md", "/workspace/task.json", "/workspace/forbidden", "/root/forbidden"):
    try:
        Path(path).write_text("forbidden")
    except OSError:
        pass
    else:
        raise AssertionError("unexpected write allowed: " + path)
Path("/workspace/output/prediction.json").write_text("{}")
''']


def test_actual_container_readonly_mounts_fresh_home_and_cli_flags(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "fake-key-no-api-call")
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o755)
    (workspace / "wiki").mkdir()
    (workspace / "wiki" / "wiki.md").write_text("immutable wiki")
    (workspace / "task.json").write_text("{}")
    (workspace / "output").mkdir()
    (workspace / "output").chmod(0o777)
    runner = ContainerProbeRunner(cfg)
    result = runner.run(workspace, "probe", tmp_path / "stdout", tmp_path / "stderr")
    assert result.exit_code == 0, (tmp_path / "stderr").read_text()
    assert (workspace / "output" / "prediction.json").read_text() == "{}"
    assert (workspace / "wiki" / "wiki.md").read_text() == "immutable wiki"
    # Both production command vectors must parse with this exact image; --help
    # exits before any model request. Each invocation still has its own home.
    for agent in ("codex", "claude_code"):
        cfg["query"]["agent"] = agent
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-no-api-call")
        adapter = DockerRunner(cfg)
        command = adapter.docker_command(workspace.resolve(), f"vmr-cli-{agent}") + ["--help"]
        proc = subprocess.run(command, env=adapter.env, capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0, proc.stderr


def test_actual_timeout_removes_container(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "fake-key-no-api-call")
    cfg["query"]["timeout_sec"] = 1
    (tmp_path / "output").mkdir()
    runner = ContainerProbeRunner(cfg)
    monkeypatch.setattr(runner, "agent_command", lambda: ["python3", "-c", "import time; time.sleep(90)"])
    names = []
    original_command = runner.docker_command
    def tracked_command(workspace, name):
        names.append(name)
        return original_command(workspace, name)
    monkeypatch.setattr(runner, "docker_command", tracked_command)
    result = runner.run(tmp_path, "", tmp_path / "stdout", tmp_path / "stderr")
    assert result.timed_out
    # run() waits for docker rm --force, so no delayed output writer survives.
    assert result.exit_code is None
    probe = subprocess.run(["docker", "ps", "-aq", "--filter", f"name=^/{names[0]}$"],
                           check=True, capture_output=True, text=True, timeout=10)
    assert probe.stdout.strip() == ""
