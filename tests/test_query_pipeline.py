import copy
import subprocess
import sys
from pathlib import Path

import pytest

from agents.runner import AgentResult, DockerRunner
from harness.aggregate import aggregate
from harness.common import HarnessError, read_json, read_jsonl, write_json
from harness.evaluate import evaluate
from harness.run_query import Experiment


class ProcessFixtureRunner:
    """Test-only subprocess, never exposed as a production agent/backend."""
    provenance = {"runtime": "test-subprocess"}

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.workspaces = []
        self.pids = []

    def run(self, workspace, prompt, stdout, stderr):
        self.workspaces.append(workspace)
        assert set(p.name for p in workspace.iterdir()) == {"AGENTS.md", "task.json", "wiki", "output"}
        assert set(p.name for p in (workspace / "wiki").iterdir()) == {"wiki.md", "frames.jsonl", "frames"}
        assert list((workspace / "output").iterdir()) == []
        assert "task.json" in prompt
        script = '''
import json, os
from pathlib import Path
task = json.loads(Path("task.json").read_text())
print(os.getpid())
result = {"query_id": task["query_id"], "video_id": task["video_id"], "split": task["split"], "moments": [
    {"start_sec": 0, "end_sec": 1, "score": 0.9, "evidence": "red scene"},
    {"start_sec": 2, "end_sec": 3, "score": 0.8, "evidence": "red scene"}]}
Path("output/prediction.json").write_text(json.dumps(result))
'''
        process = subprocess.run([sys.executable, "-c", script], cwd=workspace, capture_output=True)
        self.pids.append(int(process.stdout))
        stdout.write_bytes(process.stdout)
        stderr.write_bytes(process.stderr)
        prediction = workspace / "output" / "prediction.json"
        if self.behavior == "missing":
            prediction.unlink()
        elif self.behavior == "invalid":
            prediction.write_text('{"bad":NaN}')
        elif self.behavior == "extra":
            (workspace / "output" / "extra.txt").write_text("extra")
        elif self.behavior == "symlink":
            prediction.unlink()
            prediction.symlink_to(workspace / "task.json")
        elif self.behavior == "empty":
            row = read_json(prediction)
            row["moments"] = []
            write_json(prediction, row)
        elif self.behavior == "mutate":
            wiki = workspace / "wiki" / "wiki.md"
            wiki.chmod(0o644)
            wiki.write_text("changed")
        return AgentResult(7 if self.behavior == "nonzero" else process.returncode,
                           self.behavior == "timeout")


def test_end_to_end_fresh_processes_cleanup_and_no_gt_reads(frozen):
    cfg, captioner = frozen
    truth = Path(cfg["paths"]["datasets"]) / "qvhighlights" / "ground_truth.jsonl"
    original_gt = truth.read_text()
    truth.write_text("INVALID GT: query execution must never read this")
    runner = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=runner) as experiment:
        results = [experiment.run(query) for query in experiment.queries]
        assert all(r["status"] == "success" for r in results)
        assert len({r["wiki_hash"] for r in results}) == 1
        assert len(set(runner.workspaces)) == 3
        assert len(set(runner.pids)) == 3
        assert all(not w.exists() for w in runner.workspaces)
        assert list(Path(cfg["paths"]["runs"]).iterdir()) == []
    with Experiment(cfg, "test", runner=runner) as experiment:
        assert experiment.run(experiment.queries[0])["status"] == "success"
    assert len(runner.workspaces) == 3
    assert captioner.calls == 3
    truth.write_text(original_gt)
    root = Path(cfg["paths"]["results"]) / "test"
    rows = aggregate(root / "predictions", root / "predictions.jsonl")
    assert len(rows) == 3 and all(len(r["moments"]) == 2 for r in rows)
    metrics = evaluate(root / "predictions.jsonl", truth, split="train", evaluator="qvhighlights",
                       metadata_dir=root / "run_metadata")
    assert metrics["failed_runs"] == 0
    assert metrics["primary_score"] == 100.0
    assert metrics["average_predictions_per_query"] == 2.0
    assert (root / "config.yaml").exists()


@pytest.mark.parametrize("behavior", ["missing", "invalid", "extra", "nonzero", "timeout", "symlink", "mutate"])
def test_failures_never_retried_or_published(frozen, behavior):
    cfg, _ = frozen
    runner = ProcessFixtureRunner(behavior)
    with Experiment(cfg, "test", runner=runner) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["status"] == "failed"
        assert result["finished_at"] is not None
        assert experiment.run(experiment.queries[0]) == result
        assert not (experiment.root / "predictions" / "1.json").exists()
        assert len(runner.workspaces) == 1
        assert not runner.workspaces[0].exists()


def test_abstention_is_success(frozen):
    cfg, _ = frozen
    with Experiment(cfg, "test", runner=ProcessFixtureRunner("empty")) as experiment:
        assert experiment.run(experiment.queries[0])["status"] == "success"


def test_interrupted_attempt_is_finalized_without_second_process(frozen):
    cfg, _ = frozen
    runner = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=runner) as experiment:
        query = experiment.queries[0]
        write_json(experiment.root / "run_metadata" / "1.json", {
            "query_id": "1", "video_id": "video", "dataset": cfg["dataset"]["name"],
            "split": cfg["dataset"]["split"], "status": "running",
            "started_at": "2026-09-08T00:00:00+00:00", "finished_at": None,
        })
        result = experiment.run(query)
        assert result["status"] == "failed"
        assert result["finished_at"] is not None
        assert runner.workspaces == []


def test_mixed_agent_and_modified_inputs_refused(frozen):
    cfg, _ = frozen
    runner = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=runner) as experiment:
        experiment.run(experiment.queries[0])
        with pytest.raises(HarnessError, match="already running"):
            with Experiment(cfg, "test", runner=runner):
                pass
    modified = copy.deepcopy(cfg)
    modified["query"]["agent"] = "claude_code"
    with pytest.raises(HarnessError, match="changed"):
        with Experiment(modified, "test", runner=runner):
            pass


def test_docker_mount_boundary_and_credentials_not_in_command(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    command = runner.docker_command(tmp_path, "test-container", "isolated-network")
    assert "test-secret" not in " ".join(command)
    assert "CODEX_API_KEY" in command
    assert "--read-only" in command and "--cap-drop=ALL" in command
    assert command[command.index("--network") + 1] == "isolated-network"
    assert command[command.index("--dns") + 1] == "127.0.0.1"
    assert "HTTPS_PROXY=http://egress-proxy:8080" in command
    assert runner.provenance["egress"] == {
        "mode": "allowlist-proxy", "hosts": ["api.openai.com", "chatgpt.com"]}
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert mounts == [f"type=bind,src={tmp_path},dst=/workspace,readonly",
                      f"type=bind,src={tmp_path / 'output'},dst=/workspace/output"]
    assert "--ephemeral" in command
    assert "resume" not in command
    assert "sha256:fixture" in command
    cfg["query"]["agent"] = "claude_code"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "another-secret")
    command = DockerRunner(cfg).agent_command()
    assert "--no-session-persistence" in command
    assert "--append-system-prompt-file" in command
    assert "--strict-mcp-config" in command
