import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from agents.runner import AgentResult, DockerRunner
from harness.aggregate import aggregate
from harness.common import HarnessError, read_json, write_json
from harness.config import load_config
from harness.evaluate import evaluate
from harness.freeze import freeze_dataset
from harness.ingest_all import ingest_all
from harness.run_query import Experiment
from harness.workspace import require_anonymous_wiki


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


def test_experiment_requires_frozen_split(prepared):
    cfg, captioner = prepared
    with pytest.raises(HarnessError, match="not ingested"):
        Experiment(cfg, "test", runner=ProcessFixtureRunner())
    ingest_all(cfg, captioner=captioner)
    with pytest.raises(HarnessError, match="ingested but not frozen") as exc:
        Experiment(cfg, "test", runner=ProcessFixtureRunner())
    assert "python harness/freeze.py --dataset qvhighlights --split train" in str(exc.value)


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


def test_workspace_hides_public_identifiers_and_saves_real_ones(frozen):
    """The agent is given aliases; the saved prediction carries real identifiers."""
    cfg, _ = frozen
    seen = []
    class Spy(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            seen.append(read_json(workspace / "task.json"))
            return super().run(workspace, prompt, stdout, stderr)
    with Experiment(cfg, "test", runner=Spy()) as experiment:
        query = experiment.queries[0]
        result = experiment.run(query)
    task = seen[0]
    assert task["query_id"] != query["query_id"] and task["video_id"] != query["video_id"]
    assert task["split"] != cfg["dataset"]["split"]
    assert task["query"] == query["query"]  # the task input itself cannot be obscured
    assert set(task) == {"query_id", "video_id", "split", "query", "max_predictions"}
    saved = read_json(Path(cfg["paths"]["results"]) / "test" / "predictions" / f"{query['query_id']}.json")
    assert (saved["query_id"], saved["video_id"], saved["split"]) == (
        query["query_id"], query["video_id"], cfg["dataset"]["split"])
    assert result["task_query_id"] == task["query_id"]


def test_aliases_are_stable_within_and_distinct_across_experiments(frozen):
    cfg, _ = frozen
    tasks = {}
    class Spy(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            tasks.setdefault(self.behavior, []).append(read_json(workspace / "task.json")["query_id"])
            return super().run(workspace, prompt, stdout, stderr)
    for name in ("one", "one", "two"):
        runner = Spy(name)
        with Experiment(cfg, name, runner=runner) as experiment:
            experiment.run(experiment.queries[0])
    # Resuming "one" reuses the minted secret, so a rerun cannot change aliases;
    # a different experiment must not produce a correlatable alias.
    assert len(tasks["one"]) == 1 and tasks["one"][0] != tasks["two"][0]
    with Experiment(cfg, "one", runner=Spy("one")) as experiment:
        assert experiment.aliases.query[experiment.queries[0]["query_id"]] == tasks["one"][0]


def test_prediction_echoing_the_real_identifier_is_rejected(frozen):
    cfg, _ = frozen
    class RealIdRunner(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            result = super().run(workspace, prompt, stdout, stderr)
            write_json(workspace / "output" / "prediction.json",
                       {"query_id": "1", "video_id": "video", "split": "train",
                        "moments": [{"start_sec": 0, "end_sec": 1, "score": 0.9}]})
            return result
    with Experiment(cfg, "test", runner=RealIdRunner()) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["failure_kind"] == "invalid_output"


def test_wiki_naming_its_video_is_refused(frozen, tmp_path):
    """A frozen wiki cannot be edited, so legacy titles are caught on the way in."""
    cfg, _ = frozen
    current = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video" / "wiki.md"
    require_anonymous_wiki(current)
    legacy = tmp_path / "legacy.md"
    legacy.write_text("# Video: video\n" + current.read_text().split("\n", 1)[1], encoding="utf-8")
    with pytest.raises(HarnessError, match="names its own video"):
        require_anonymous_wiki(legacy)


def test_experiment_without_alias_secret_is_refused(frozen):
    cfg, _ = frozen
    with Experiment(cfg, "test", runner=ProcessFixtureRunner()):
        pass
    manifest = Path(cfg["paths"]["results"]) / "test" / "experiment.json"
    saved = read_json(manifest)
    del saved["alias_secret"]
    write_json(manifest, saved)
    with pytest.raises(HarnessError, match="predates workspace identifier aliasing"):
        with Experiment(cfg, "test", runner=ProcessFixtureRunner()):
            pass


def test_abstention_is_success(frozen):
    cfg, _ = frozen
    with Experiment(cfg, "test", runner=ProcessFixtureRunner("empty")) as experiment:
        assert experiment.run(experiment.queries[0])["status"] == "success"


@pytest.mark.parametrize("behavior,kind", [
    ("missing", "invalid_output"), ("invalid", "invalid_output"), ("extra", "invalid_output"),
    ("symlink", "invalid_output"), ("nonzero", "agent_error"), ("timeout", "timeout"),
    ("mutate", "tampered")])
def test_agent_failures_are_classified_by_cause(frozen, behavior, kind):
    cfg, _ = frozen
    with Experiment(cfg, "test", runner=ProcessFixtureRunner(behavior)) as experiment:
        result = experiment.run(experiment.queries[0])
        assert (result["status"], result["failure_kind"]) == ("failed", kind)
        assert result["attempts"] == 1


def test_harness_failure_is_recorded_raised_and_retried(frozen):
    """An infrastructure fault is not the agent's score, so it never stands as one."""
    cfg, _ = frozen
    class BrokenRunner(ProcessFixtureRunner):
        def run(self, *args):
            raise OSError("docker daemon is not running")
    with Experiment(cfg, "test", runner=BrokenRunner()) as experiment:
        query = experiment.queries[0]
        with pytest.raises(OSError):
            experiment.run(query)
        recorded = read_json(experiment.root / "run_metadata" / "1.json")
        assert recorded["failure_kind"] == "harness_error"
        assert "docker daemon" in recorded["error"]
    runner = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=runner) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["status"] == "success"
        assert result["attempts"] == 2
        assert [f["kind"] for f in result["superseded_failures"]] == ["harness_error"]
        assert len(runner.workspaces) == 1


def test_interrupted_attempt_is_retried_not_charged_to_the_agent(frozen):
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
        assert result["status"] == "success"
        assert [f["kind"] for f in result["superseded_failures"]] == ["interrupted"]
        assert len(runner.workspaces) == 1


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
        "mode": "allowlist-proxy",
        "hosts": cfg["query"]["egress_allowed_hosts"]["codex"],
    }
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


def test_claude_tool_surface_allows_writing_and_excludes_network_tools(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    cfg["query"]["agent"] = "claude_code"
    command = DockerRunner(cfg).agent_command()
    declared = command[command.index("--tools") + 1].split(",")
    assert "Write" in declared, "without Write the agent cannot produce prediction.json"
    # --bare pins the tools to Bash, Edit and Read whatever --tools says.
    assert "--bare" not in command
    denied = command[command.index("--disallowedTools") + 1:command.index("--append-system-prompt-file")]
    assert set(denied) == {"Monitor", "PushNotification"}
    for tool in ("WebSearch", "WebFetch", "Agent", "TaskCreate", "CronCreate"):
        assert tool not in declared


def test_default_base_url_is_not_passed_into_the_container(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    command = runner.docker_command(tmp_path, "test-container", "isolated-network")
    assert not [arg for arg in command if "BASE_URL" in arg]
    assert runner.provenance["api_base_url"] is None


def test_gateway_base_url_reaches_the_agent_and_provenance(cfg, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    cfg["query"]["agent"] = "claude_code"
    cfg["query"]["base_url"]["claude_code"] = "https://gateway.example.net/v1"
    cfg["query"]["egress_allowed_hosts"]["claude_code"] = ["gateway.example.net"]
    runner = DockerRunner(cfg)
    command = runner.docker_command(tmp_path, "test-container", "isolated-network")
    assert "ANTHROPIC_BASE_URL=https://gateway.example.net/v1" in command
    assert "OPENAI_BASE_URL" not in " ".join(command)
    assert runner.provenance["api_base_url"] == "https://gateway.example.net/v1"


def written_config(tmp_path, **query) -> Path:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    raw["query"].update(query)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_base_url_defaults_to_the_vendor_endpoint(tmp_path):
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    del raw["query"]["base_url"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path)["query"]["base_url"] == {"codex": None, "claude_code": None}


@pytest.mark.parametrize("url, reason", [
    # The proxy would answer each of these with an opaque 403 or never see them.
    ("https://gateway.example.net/v1", "not in query.egress_allowed_hosts"),
    ("http://api.example.net/v1", "must be an https URL"),
    ("https://api.example.net:8443/v1", "must use port 443"),
    ("https://user:pw@api.example.net/v1", "must not carry credentials"),
])
def test_unreachable_base_url_is_rejected_at_load(tmp_path, url, reason):
    path = written_config(tmp_path, base_url={"codex": None, "claude_code": url})
    with pytest.raises(HarnessError, match=reason):
        load_config(path)


def test_base_url_matching_the_allowlist_is_accepted(tmp_path):
    path = written_config(tmp_path, base_url={"codex": None, "claude_code": "https://api.example.net/v1"})
    assert load_config(path)["query"]["base_url"]["claude_code"] == "https://api.example.net/v1"
