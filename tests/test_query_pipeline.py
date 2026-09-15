import copy
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from agents.runner import AgentCancelled, AgentResult, DockerRunner, readable_trace, trace_path
from harness.aggregate import aggregate
from harness.common import HarnessError, read_json, write_json
from harness.config import load_config
from harness.evaluate import evaluate
from harness.freeze import freeze_dataset
from harness.ingest_all import ingest_all
from harness.run_query import Experiment, clamp_prediction_ends, run_status
from harness.workspace import require_anonymous_wiki


class ProcessFixtureRunner:
    """Test-only subprocess, never exposed as a production agent/backend."""
    provenance = {"runtime": "test-subprocess"}

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.workspaces = []
        self.pids = []

    def run(self, workspace, prompt, stdout, stderr, **kwargs):
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


def test_structured_trace_keeps_readable_final_answers(tmp_path):
    assert trace_path(tmp_path / "q1.stdout.log") == tmp_path / "q1.trace.jsonl"
    assert trace_path(tmp_path / "stdout") == tmp_path / "stdout.trace.jsonl"

    claude = b'\n'.join([
        b'{"type":"harness.input","prompt":"secret task"}',
        b'{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read"}]}}',
        b'{"type":"result","result":"claude final"}',
    ])
    assert readable_trace("claude_code", claude) == b"claude final\n"

    claude_timeout = b'\n'.join([
        b'{"type":"harness.input","prompt":"secret task"}',
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"best partial diagnosis"}]}}',
        b'{"type":"tool_result","content":"large omitted payload"}',
    ])
    assert readable_trace("claude_code", claude_timeout) == b"best partial diagnosis\n"
    assert readable_trace("claude_code", b'{"type":"tool_result"}\n') == (
        b"[no final answer event; last agent event: tool_result]\n")

    codex = b'\n'.join([
        b'{"type":"item.completed","item":{"type":"command_execution","command":"pwd"}}',
        b'{"type":"item.completed","item":{"type":"agent_message","text":"codex interim"}}',
        b'{"type":"item.completed","item":{"type":"agent_message","text":"codex final"}}',
    ])
    assert readable_trace("codex", codex) == b"codex final\n"
    assert readable_trace("claude_code", b"legacy plain output\n") == b"legacy plain output\n"
    trace = tmp_path / "stream.trace.jsonl"
    trace.write_bytes(codex)
    assert readable_trace("codex", trace) == b"codex final\n"


def test_image_id_falls_back_to_exact_image_listing(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[2] == "inspect":
            raise subprocess.CalledProcessError(1, command, stderr="No such image")
        return subprocess.CompletedProcess(command, 0, "sha256:fixture-id\n", "")

    monkeypatch.setattr("agents.runner.subprocess.run", run)
    assert DockerRunner._inspect("vmr-wiki-agents:local") == "sha256:fixture-id"
    assert calls == [
        ["docker", "image", "inspect", "--format", "{{.Id}}", "vmr-wiki-agents:local"],
        ["docker", "image", "ls", "--no-trunc", "--quiet", "vmr-wiki-agents:local"],
    ]


def test_egress_proxy_is_probed_from_the_agent_network():
    runner = object.__new__(DockerRunner)
    runner.image = "sha256:fixture"
    runner.allowed_hosts = ("api.example.test",)
    calls = []
    runner._docker = lambda command, message, **kwargs: calls.append(
        (command, message, kwargs))

    runner._start_egress_proxy("vmr-test-internal", "vmr-test-proxy")

    assert calls[0][0][:6] == [
        "docker", "run", "--detach", "--rm", "--name", "vmr-test-proxy"]
    assert calls[1][0] == [
        "docker", "network", "connect", "--alias", "egress-proxy",
        "vmr-test-internal", "vmr-test-proxy"]
    probe = calls[2][0]
    assert probe[:3] == ["docker", "run", "--rm"]
    assert probe[probe.index("--network") + 1] == "vmr-test-internal"
    assert probe[-3:-1] == ["python3", "-c"]
    assert "egress-proxy" in probe[-1]
    assert calls[2][2] == {"timeout": 30}


def test_readable_stdout_is_written_even_when_container_cleanup_fails(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    monkeypatch.setattr(runner, "_docker", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_start_egress_proxy", lambda *args: None)
    event = ('{"type":"item.completed","item":'
             '{"type":"agent_message","text":"finished answer"}}')
    monkeypatch.setattr(
        runner, "docker_command",
        lambda *args, **kwargs: [sys.executable, "-c", f"print({event!r})"])
    removals = []

    def remove(name):
        removals.append(name)
        if len(removals) == 1:
            raise HarnessError("cleanup failed")

    monkeypatch.setattr(runner, "_remove_container", remove)
    monkeypatch.setattr("agents.runner.subprocess.run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 0, "", ""))
    with pytest.raises(HarnessError, match="cleanup failed"):
        runner.run(tmp_path, "prompt", tmp_path / "stdout", tmp_path / "stderr")
    assert (tmp_path / "stdout").read_text() == "finished answer\n"


def test_trace_and_stdout_exist_when_agent_setup_fails(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    monkeypatch.setattr(runner, "_docker", lambda *args, **kwargs:
                        (_ for _ in ()).throw(HarnessError("network failed")))
    monkeypatch.setattr(runner, "_remove_container", lambda *args: None)
    with pytest.raises(HarnessError, match="network failed"):
        runner.run(tmp_path, "prompt", tmp_path / "stdout", tmp_path / "stderr")
    assert trace_path(tmp_path / "stdout").exists()
    assert (tmp_path / "stderr").exists()
    assert (tmp_path / "stdout").read_text() == "[no agent events were emitted]\n"


def test_container_runner_wait_observes_cooperative_cancellation(cfg, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.PIPE)
    cancelled = threading.Event()
    timer = threading.Timer(0.1, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(AgentCancelled):
            runner._wait_for_agent(process, "prompt", cancelled)
    finally:
        timer.cancel()
        process.kill()
        process.wait(timeout=2)
    assert time.monotonic() - started < 2


def test_container_runner_timeout_override_is_per_call(cfg, monkeypatch):
    monkeypatch.setenv("CODEX_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = DockerRunner(cfg)
    configured_timeout = runner.timeout_sec
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], stdin=subprocess.PIPE)
    started = time.monotonic()
    try:
        result = runner._wait_for_agent(process, "prompt", None, timeout_sec=0.05)
    finally:
        process.kill()
        process.wait(timeout=2)
    assert result.timed_out
    assert time.monotonic() - started < 2
    assert runner.timeout_sec == configured_timeout


def test_all_prediction_ends_over_duration_are_clamped():
    prediction = {"moments": [
        {"start_sec": 41.0, "end_sec": 43.933333, "score": 0.9},
        {"start_sec": 10.0, "end_sec": 43.94, "score": 0.8},
        {"start_sec": 43.929, "end_sec": 43.932, "score": 0.7},
        {"start_sec": 43.0, "end_sec": 43.93000000000001, "score": 0.6},
        {"start_sec": 10.0, "end_sec": 44.0, "score": 0.5},
    ]}
    adjustments = clamp_prediction_ends(prediction, duration=43.93)
    assert prediction["moments"][0]["end_sec"] == 43.93
    assert prediction["moments"][1]["end_sec"] == 43.93
    assert prediction["moments"][2]["end_sec"] == 43.93
    assert prediction["moments"][3]["end_sec"] == 43.93
    assert prediction["moments"][4]["end_sec"] == 44.0
    assert adjustments[0] == {
        "kind": "clamp_end_to_effective_duration",
        "moment_index": 0,
        "original_end_sec": 43.933333,
        "adjusted_end_sec": 43.93,
        "delta_sec": 0.003333,
        "reason": "end_sec exceeded the authoritative prediction duration",
    }
    assert [row["moment_index"] for row in adjustments] == [0, 1, 2]
    status = run_status({"query_id": "q1", "status": "success",
                         "output_adjustments": adjustments}, Path("results/test"))
    assert "success [adjusted]" in status


def test_endpoint_clamp_that_collapses_interval_is_invalid_output(frozen):
    cfg, _ = frozen

    class CollapsedRunner(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr, **kwargs):
            result = super().run(workspace, prompt, stdout, stderr, **kwargs)
            prediction = read_json(workspace / "output/prediction.json")
            prediction["moments"] = [
                {"start_sec": 3.001, "end_sec": 3.002, "score": 0.9}
            ]
            write_json(workspace / "output/prediction.json", prediction)
            return result

    with Experiment(cfg, "collapsed", runner=CollapsedRunner()) as experiment:
        result = experiment.run(experiment.queries[0])
    assert result["failure_kind"] == "invalid_output"
    assert result["output_adjustments"] == []


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


def test_text_only_query_omits_frames_and_records_effective_instructions(frozen):
    cfg, _ = frozen
    cfg["query"]["text_only"] = True
    seen = []

    class TextOnlyRunner(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            wiki_files = set(p.name for p in (workspace / "wiki").iterdir())
            assert wiki_files == {"wiki.md", "frames.jsonl"}
            assert not (workspace / "wiki" / "frames").exists()
            assert "本次不提供图片" in prompt
            assert "wiki/frames/" in prompt
            instructions = (workspace / "AGENTS.md").read_text()
            assert "纯文本 Video Moment Retrieval" in instructions
            assert "最多读取 24 张" not in instructions
            assert "读取 `task.json`、`wiki/wiki.md`、`wiki/frames.jsonl` 和 `wiki/frames/`" not in instructions
            seen.append(workspace)

            task = read_json(workspace / "task.json")
            write_json(workspace / "output" / "prediction.json", {
                "query_id": task["query_id"], "video_id": task["video_id"],
                "split": task["split"], "moments": [
                    {"start_sec": 0, "end_sec": 1, "score": 0.9}
                ]})
            stdout.write_text("text-only fixture\n")
            stderr.write_text("")
            return AgentResult(0)

    with Experiment(cfg, "text-only", runner=TextOnlyRunner()) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["status"] == "success"
        saved_agents = (experiment.root / "templates" / "AGENTS.md").read_text()
        saved_prompt = (experiment.root / "templates" / "query_prompt.md").read_text()
        assert "纯文本 Video Moment Retrieval" in saved_agents
        assert "本次不提供图片" in saved_prompt
    assert len(seen) == 1 and not seen[0].exists()


def test_experiment_requires_frozen_split(prepared):
    cfg, captioner = prepared
    with pytest.raises(HarnessError, match="not ingested"):
        Experiment(cfg, "test", runner=ProcessFixtureRunner())
    ingest_all(cfg, captioner=captioner)
    with pytest.raises(HarnessError, match="ingested but not frozen") as exc:
        Experiment(cfg, "test", runner=ProcessFixtureRunner())
    assert "python harness/freeze.py --dataset qvhighlights --split train" in str(exc.value)


@pytest.mark.parametrize("behavior", ["missing", "invalid", "extra", "nonzero", "timeout", "symlink", "mutate"])
def test_failed_cases_are_retried_then_success_is_skipped(frozen, behavior):
    cfg, _ = frozen
    runner = ProcessFixtureRunner(behavior)
    with Experiment(cfg, "test", runner=runner) as experiment:
        query = experiment.queries[0]
        result = experiment.run(query)
        assert result["status"] == "failed"
        assert result["finished_at"] is not None
        assert not (experiment.root / "predictions" / "1.json").exists()
        assert len(runner.workspaces) == 1
        assert not runner.workspaces[0].exists()

    retry = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=retry) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["status"] == "success"
        assert result["attempts"] == 2
        assert len(result["superseded_failures"]) == 1
        assert len(retry.workspaces) == 1

    skipped = ProcessFixtureRunner()
    with Experiment(cfg, "test", runner=skipped) as experiment:
        assert experiment.run(experiment.queries[0]) == result
        assert skipped.workspaces == []


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
    assert set(task) == {"query_id", "video_id", "split", "query", "max_predictions", "duration"}
    assert task["duration"] == 3.0
    saved = read_json(Path(cfg["paths"]["results"]) / "test" / "predictions" / f"{query['query_id']}.json")
    assert (saved["query_id"], saved["video_id"], saved["split"]) == (
        query["query_id"], query["video_id"], cfg["dataset"]["split"])
    assert result["task_query_id"] == task["query_id"]


def test_query_clamps_wiki_annotation_duration_gap_and_records_it(frozen):
    cfg, _ = frozen
    seen = []

    class RoundingGapRunner(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            seen.append(read_json(workspace / "task.json"))
            return super().run(workspace, prompt, stdout, stderr)

    with Experiment(cfg, "rounding-gap", runner=RoundingGapRunner()) as experiment:
        query = experiment.queries[0]
        # The frozen media is 3.0s; emulate a dataset annotation rounded down.
        experiment.videos[query["video_id"]]["duration"] = 2.99
        result = experiment.run(query)

    assert seen[0]["duration"] == 2.99
    assert result["status"] == "success"
    assert result["output_adjustments"] == [{
        "kind": "clamp_end_to_effective_duration",
        "moment_index": 1,
        "original_end_sec": 3,
        "adjusted_end_sec": 2.99,
        "delta_sec": 0.01,
        "reason": "end_sec exceeded the authoritative prediction duration",
    }]
    saved = read_json(Path(cfg["paths"]["results"]) / "rounding-gap" /
                      "predictions" / f"{query['query_id']}.json")
    assert saved["moments"][1]["end_sec"] == 2.99


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


def test_internal_prediction_processing_error_stays_a_harness_failure(frozen, monkeypatch):
    cfg, _ = frozen
    monkeypatch.setattr("harness.run_query.clamp_prediction_ends", lambda *args, **kwargs:
                        (_ for _ in ()).throw(HarnessError("internal clamp invariant")))
    with Experiment(cfg, "internal-error", runner=ProcessFixtureRunner()) as experiment:
        query = experiment.queries[0]
        with pytest.raises(HarnessError, match="internal clamp invariant"):
            experiment.run(query)
        metadata = read_json(experiment.root / "run_metadata" / f"{query['query_id']}.json")
    assert metadata["failure_kind"] == "harness_error"


def test_discarded_tampered_prediction_does_not_claim_adjustments(frozen):
    cfg, _ = frozen

    class OverrunAndTamperRunner(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr, **kwargs):
            result = super().run(workspace, prompt, stdout, stderr, **kwargs)
            path = workspace / "output/prediction.json"
            prediction = read_json(path)
            prediction["moments"][1]["end_sec"] = 3.01
            write_json(path, prediction)
            wiki = workspace / "wiki/wiki.md"
            wiki.chmod(0o644)
            wiki.write_text("changed")
            return result

    with Experiment(cfg, "tampered-adjustment", runner=OverrunAndTamperRunner()) as experiment:
        result = experiment.run(experiment.queries[0])
    assert result["failure_kind"] == "tampered"
    assert result["output_adjustments"] == []


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
    assert "--json" in command
    assert "resume" not in command
    assert "sha256:fixture" in command
    cfg["query"].update(load_config(
        Path(__file__).resolve().parents[1] / "config.yaml",
        query_agent_profile="claude_default")["query"])
    cfg["query"]["model"] = "fixture-agent"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "another-secret")
    command = DockerRunner(cfg).agent_command()
    assert "--no-session-persistence" in command
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in command
    assert "--append-system-prompt-file" in command
    assert "--strict-mcp-config" in command


def test_claude_tool_surface_allows_writing_and_excludes_network_tools(cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setattr(DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    cfg["query"].update(load_config(
        Path(__file__).resolve().parents[1] / "config.yaml",
        query_agent_profile="claude_default")["query"])
    cfg["query"]["model"] = "fixture-agent"
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
    cfg["query"].update(load_config(
        Path(__file__).resolve().parents[1] / "config.yaml",
        query_agent_profile="claude_default")["query"])
    cfg["query"]["model"] = "fixture-agent"
    cfg["query"]["base_url"]["claude_code"] = "https://gateway.example.net/v1"
    cfg["query"]["egress_allowed_hosts"]["claude_code"] = ["gateway.example.net"]
    runner = DockerRunner(cfg)
    command = runner.docker_command(tmp_path, "test-container", "isolated-network")
    assert "ANTHROPIC_BASE_URL=https://gateway.example.net/v1" in command
    assert "OPENAI_BASE_URL" not in " ".join(command)
    assert runner.provenance["api_base_url"] == "https://gateway.example.net/v1"


def written_config(tmp_path, **profile) -> Path:
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    raw["query"]["agent_profile"] = "claude_default"
    raw["profiles"]["agents"]["claude_default"].update(profile)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_text_only_defaults_off_and_rejects_non_boolean(tmp_path):
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    del raw["query"]["input_mode"]
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path)["query"]["text_only"] is False

    raw["query"]["input_mode"] = "images_maybe"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="query.input_mode must be text or multimodal"):
        load_config(path)


def test_base_url_defaults_to_the_vendor_endpoint(tmp_path):
    raw = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    del raw["profiles"]["agents"]["codex_default"]["base_url"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path)["query"]["base_url"] == {"codex": None}


@pytest.mark.parametrize("url, reason", [
    # The proxy would answer each of these with an opaque 403 or never see them.
    ("https://gateway.example.net/v1", "not in profiles.agents.claude_default.egress_allowed_hosts"),
    ("http://api.example.net/v1", "must be an https URL"),
    ("https://api.example.net:8443/v1", "must use port 443"),
    ("https://user:pw@api.example.net/v1", "must not carry credentials"),
])
def test_unreachable_base_url_is_rejected_at_load(tmp_path, url, reason):
    path = written_config(tmp_path, base_url=url)
    with pytest.raises(HarnessError, match=reason):
        load_config(path)


def test_base_url_matching_the_allowlist_is_accepted(tmp_path):
    path = written_config(tmp_path, base_url="https://api.example.net/v1")
    assert load_config(path)["query"]["base_url"]["claude_code"] == "https://api.example.net/v1"
