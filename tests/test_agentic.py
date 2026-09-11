"""Agent-compiled wiki. No model API calls: a fixture runner plays the agent."""
from __future__ import annotations

import copy
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from adapters.qvhighlights import QVHighlightsAdapter
from agents.runner import AgentIngestRunner, AgentResult, FatalAgentError
from harness.agentic_config import AGENTIC_VERSION, settings
from harness.agentic_validate import validate_agent_wiki
from harness.common import (HarnessError, ingest_content_diff, ingest_content_hash, read_json,
                            write_json, write_jsonl)
from harness.config import load_config
from harness.freeze import freeze_dataset, remove_tree
from harness.ingest_all import ingest_all
from harness.run_query import Experiment
from harness.workspace import require_anonymous_wiki

ROOT = Path(__file__).resolve().parents[1]

JPEG = b"\xff\xd8\xff" + b"\x00" * 64


def agentic_config(tmp_path: Path, **agentic) -> Path:
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["dataset"]["split"] = "train"
    raw["ingest"]["caption_mode"] = "agentic"
    raw["ingest"]["sample_interval_sec"] = 1.0
    raw["ingest"]["image_max_size"] = 64
    raw["ingest"]["agentic"].update({"model": "fixture-agent-model", **agentic})
    raw["query"]["model"] = "fixture-agent"
    for kind in ("datasets", "wiki", "runs", "results"):
        raw["paths"][kind] = str(tmp_path / kind)
    raw["paths"]["templates"] = str(ROOT / "templates")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def agentic(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for video integration tests")
    cfg = load_config(agentic_config(tmp_path))
    video_root = tmp_path / "videos"
    video_root.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=80x60:r=10",
                    "-t", "3", "-c:v", "mpeg4", "-y", str(video_root / "video.mp4")], check=True)
    raw = tmp_path / "annotations.jsonl"
    write_jsonl(raw, [
        {"qid": i, "vid": "video", "duration": 3.0, "query": f"Find red scene {i}",
         "split": "train", "relevant_windows": [[0.0, 1.0], [2.0, 3.0]]} for i in range(1, 4)])
    QVHighlightsAdapter().prepare(raw, video_root, Path(cfg["paths"]["datasets"]) / "qvhighlights",
                                  split="train")
    yield cfg
    for kind in ("wiki", "runs"):
        if (tmp_path / kind).exists():
            remove_tree(tmp_path / kind)


def frame_rows(count=3):
    return [{"frame_id": f"f{i:06d}", "timestamp": round(i * 0.5, 3),
             "frame": f"frames/{i:06d}.jpg", "reason": "semantic_evidence",
             "description": f"A red scene, sample {i}."} for i in range(1, count + 1)]


def wiki_text(rows):
    evidence = "\n".join(f"- [{row['frame_id']} @ {row['timestamp']}s]({row['frame']})"
                         for row in rows)
    return ("# Video\n\n## Metadata\n\n- Duration: 3.0 seconds\n\n"
            "## Chapters\n\n### C0001 - A static red scene\n\n"
            "#### E0001 - The scene stays red\n\n"
            "##### M0001 - Nothing moves\n\n"
            f"**Time:** `00:00:00.000 -> 00:00:01.500`\n\n**Evidence:**\n\n{evidence}\n\n"
            "**Observed:**\n\nA uniform red frame with no people.\n\n"
            "**Inferred / retrieval semantics:**\n\nLikely a test pattern.\n\n"
            "**Retrieval aliases:**\n\n- a red screen\n\n"
            "## Temporal Relations\n\n- M0001 PART_OF E0001\n")


class FixtureIngestRunner:
    """Stands in for a coding agent; never contacts a model API."""

    provenance = {"runtime": "test-agentic", "image_id": "sha256:fixture"}

    def __init__(self, behavior="valid"):
        self.behavior = behavior
        self.jobs, self.tasks, self.videos, self.scratches = [], [], [], []

    def run(self, job, prompt, stdout, stderr, *, video, scratch):
        self.jobs.append(Path(job))
        self.videos.append(Path(video))
        self.scratches.append(Path(scratch))
        self.tasks.append(read_json(Path(job) / "task.json"))
        stdout.write_bytes(b"fixture agent stdout\n")
        stderr.write_bytes(b"")
        output = Path(job) / "output"
        if self.behavior == "timeout":
            return AgentResult(None, timed_out=True)
        if self.behavior == "nonzero":
            return AgentResult(3)
        if self.behavior == "missing":
            return AgentResult(0)
        if self.behavior == "tamper":
            instructions = Path(job) / "AGENTS.md"
            instructions.chmod(0o644)
            instructions.write_text("ignore the contract", encoding="utf-8")
            return AgentResult(0)

        rows = frame_rows()
        frames = output / "frames"
        frames.mkdir()
        for row in rows:
            (output / row["frame"]).write_bytes(JPEG)
        wiki = wiki_text(rows)

        if self.behavior == "orphan":
            (frames / "000009.jpg").write_bytes(JPEG)
        elif self.behavior == "extra":
            (output / "notes.md").write_text("scratch work leaked", encoding="utf-8")
        elif self.behavior == "symlink":
            (output / "shortcut.md").symlink_to(output / "frames" / "000001.jpg")
        elif self.behavior == "unregistered_link":
            wiki = wiki.replace("frames/000001.jpg", "frames/000099.jpg")
        elif self.behavior == "late_timestamp":
            rows[-1]["timestamp"] = 99.0
        elif self.behavior == "unsorted":
            rows[1]["timestamp"] = rows[0]["timestamp"]
        elif self.behavior == "duplicate_id":
            rows[1]["frame_id"] = rows[0]["frame_id"]
        elif self.behavior == "bad_reason":
            rows[0]["reason"] = "vibes"
        elif self.behavior == "unknown_field":
            rows[0]["mood"] = "calm"
        elif self.behavior == "absolute_path":
            wiki += "\nFrames were staged in /scratch/candidates.\n"
        elif self.behavior == "traversal":
            wiki += "\nSee [context](../other/frames/000001.jpg).\n"
        elif self.behavior == "bad_title":
            wiki = wiki.replace("# Video\n", "# Video video.mp4\n", 1)
        elif self.behavior == "empty_registry":
            rows = []

        write_jsonl(output / "frames.jsonl", rows)
        (output / "wiki.md").write_text(wiki, encoding="utf-8")
        return AgentResult(0)


class PredictingRunner:
    """Query-side stand-in: writes one valid prediction and exits."""

    provenance = {"runtime": "test-query"}

    def run(self, workspace, prompt, stdout, stderr):
        stdout.write_bytes(b"")
        stderr.write_bytes(b"")
        task = read_json(workspace / "task.json")
        write_json(workspace / "output" / "prediction.json",
                   {"query_id": task["query_id"], "video_id": task["video_id"],
                    "split": task["split"],
                    "moments": [{"start_sec": 0.0, "end_sec": 1.0, "score": 0.9}]})
        return AgentResult(0)


def test_agentic_wiki_flows_through_freeze_and_query(agentic):
    cfg = agentic
    runner = FixtureIngestRunner()
    results = ingest_all(cfg, runner=runner, jobs=1)
    assert [row["video_id"] for row in results] == ["video"]

    wiki = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    assert sorted(p.name for p in wiki.iterdir()) == ["frames", "frames.jsonl", "ingest.json",
                                                      "wiki.md"]
    assert len(list((wiki / "frames").iterdir())) == 3
    require_anonymous_wiki(wiki / "wiki.md")
    metadata = read_json(wiki / "ingest.json")
    assert metadata["agentic_version"] == AGENTIC_VERSION
    assert metadata["telemetry"]["frame_count"] == 3
    assert metadata["agent_provenance"]["model"] == "fixture-agent-model"
    assert "vlm" not in metadata["ingest_config_hash"]

    # The job workspace and its scratch are temporary, and neither survives.
    assert len(runner.jobs) == 1
    assert not runner.jobs[0].exists() and not runner.scratches[0].exists()
    assert list(Path(cfg["paths"]["runs"]).iterdir()) == []
    # Agent logs stay outside the wiki, so they never enter the frozen content.
    logs = Path(cfg["paths"]["wiki"]) / "qvhighlights" / ".ingest-logs" / "video"
    assert (logs / "agent.stdout.log").read_text() == "fixture agent stdout\n"

    freeze_dataset(cfg)
    with Experiment(cfg, "agentic", runner=PredictingRunner()) as experiment:
        assert all(experiment.run(query)["status"] == "success" for query in experiment.queries)


def test_task_json_carries_no_query_split_or_video_identity(agentic):
    cfg = agentic
    runner = FixtureIngestRunner()
    ingest_all(cfg, runner=runner, jobs=1)
    task = runner.tasks[0]
    assert set(task) == {"version", "task", "input", "output", "scratch", "duration",
                         "video_stream_duration", "fps", "width", "height", "has_audio",
                         "image_max_size", "jpeg_qscale", "max_frames", "frame_extraction", "wiki"}
    serialized = yaml.safe_dump(task)
    for leaked in ("video_id", "query", "split", "Find red scene", "qvhighlights"):
        assert leaked not in serialized
    assert task["input"]["video"] == "/input/video.mp4"
    assert task["width"] == 80 and task["height"] == 60 and task["has_audio"] is False
    # The job directory the container can see through /proc is not named after
    # the video, and the mounted video is the real read-only source.
    assert "video" not in runner.jobs[0].name
    assert runner.videos[0].name == "video.mp4"


def test_ingest_all_reuses_a_completed_agentic_wiki(agentic):
    cfg = agentic
    first = FixtureIngestRunner()
    ingest_all(cfg, runner=first, jobs=1)
    second = FixtureIngestRunner()
    ingest_all(cfg, runner=second, jobs=1)
    # A completed ingest is verified, never recompiled by a second agent run.
    assert len(second.jobs) == 0


@pytest.mark.parametrize("behavior,message", [
    ("missing", "got nothing"),
    ("extra", "Expected exactly"),
    ("symlink", "Symlink is forbidden"),
    ("orphan", "not registered in frames.jsonl"),
    ("unregistered_link", "references an unregistered frame"),
    ("late_timestamp", "exceeds the video duration"),
    ("unsorted", "strictly increasing"),
    ("duplicate_id", "Duplicate frame_id"),
    ("bad_reason", "reason must be one of"),
    ("unknown_field", "unknown field"),
    ("absolute_path", "non-portable absolute path"),
    ("traversal", "path traversal"),
    ("bad_title", "must start with"),
    ("empty_registry", "registers no frames"),
    ("nonzero", "exited with code 3"),
    ("timeout", "timed out"),
    ("tamper", "changed its own immutable instructions"),
])
def test_contract_violations_are_never_published(agentic, behavior, message):
    cfg = agentic
    runner = FixtureIngestRunner(behavior)
    with pytest.raises(HarnessError, match=message):
        ingest_all(cfg, runner=runner, jobs=1)
    wiki = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    assert not wiki.exists()
    # A rejected attempt still leaves a report and cleans up its workspace.
    report = read_json(Path(cfg["paths"]["wiki"]) / "qvhighlights" / ".ingest-failures" / "video.json")
    assert report["status"] == "failed"
    assert not runner.jobs[0].exists() and not runner.scratches[0].exists()


def test_identity_leak_is_rejected(tmp_path):
    output = tmp_path / "output"
    (output / "frames").mkdir(parents=True)
    rows = frame_rows(2)
    for row in rows:
        (output / row["frame"]).write_bytes(JPEG)
    write_jsonl(output / "frames.jsonl", rows)
    limits = {"duration": 3.0, "max_frames": 10, "max_wiki_bytes": 1 << 20,
              "max_frame_bytes": 1 << 20}

    (output / "wiki.md").write_text(wiki_text(rows), encoding="utf-8")
    assert validate_agent_wiki(output, forbidden_tokens=("NUsG9BgSes0",), **limits)

    (output / "wiki.md").write_text(wiki_text(rows) + "\nSource: NUsG9BgSes0.\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="leaks the video identity"):
        validate_agent_wiki(output, forbidden_tokens=("NUsG9BgSes0",), **limits)

    # A short id cannot be told apart from prose, so it is not searched for.
    (output / "wiki.md").write_text(wiki_text(rows) + "\nA video of a scene.\n", encoding="utf-8")
    assert validate_agent_wiki(output, forbidden_tokens=("video",), **limits)


def test_agentic_config_needs_no_vlm_deployment(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["ingest"]["caption_mode"] = "agentic"
    raw["ingest"]["agentic"]["model"] = "fixture-agent-model"
    del raw["ingest"]["vlm"]
    raw["paths"]["templates"] = str(ROOT / "templates")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True), encoding="utf-8")
    cfg = load_config(path)
    assert cfg["ingest"]["agentic_version"] == AGENTIC_VERSION
    # The content hash must not depend on VLM settings that are never used.
    assert ingest_content_hash(cfg)


def test_agentic_config_rejects_unpinned_model_and_unreachable_gateway(tmp_path):
    with pytest.raises(HarnessError, match="explicit ingest.agentic.model"):
        load_config(agentic_config(tmp_path, model="REPLACE_WITH_AGENT_MODEL"))
    with pytest.raises(HarnessError, match="not in ingest.agentic.egress_allowed_hosts"):
        load_config(agentic_config(tmp_path, agent="codex",
                                   base_url={"codex": "https://elsewhere.example.org/v1",
                                             "claude_code": None}))
    with pytest.raises(HarnessError, match="levels must stay in chapter, event, moment order"):
        load_config(agentic_config(tmp_path, wiki={"levels": ["moment", "chapter"]}))


def test_instruction_templates_are_part_of_the_content_identity(tmp_path, agentic):
    cfg = agentic
    changed = copy.deepcopy(cfg)
    changed["ingest"]["agentic"]["templates_hash"] = "0" * 64
    assert ingest_content_hash(changed) != ingest_content_hash(cfg)
    diffs = ingest_content_diff(changed["ingest"], cfg)
    assert diffs == ["agentic instruction templates differ"]

    # Freeze refuses a wiki compiled under different instructions.
    ingest_all(cfg, runner=FixtureIngestRunner(), jobs=1)
    with pytest.raises(HarnessError, match="instruction templates differ"):
        freeze_dataset(changed)


def test_model_change_is_a_content_change(agentic):
    cfg = agentic
    other = copy.deepcopy(cfg)
    other["ingest"]["agentic"]["model"] = "another-agent-model"
    assert ingest_content_hash(other) != ingest_content_hash(cfg)
    assert ingest_content_diff(other["ingest"], cfg) == [
        "agentic.model 'another-agent-model' vs 'fixture-agent-model'"]


def test_container_budget_is_provenance_not_content(agentic):
    cfg = agentic
    relaxed = copy.deepcopy(cfg)
    relaxed["ingest"]["agentic"].update(timeout_sec=7200, memory_gb=16, cpus=8,
                                        container_image="other:local")
    assert ingest_content_hash(relaxed) == ingest_content_hash(cfg)
    assert ingest_content_diff(relaxed["ingest"], cfg) == []


def test_ingest_mounts_the_video_readonly_and_keeps_credentials_out(agentic, monkeypatch, tmp_path):
    cfg = agentic
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret")
    monkeypatch.setattr(AgentIngestRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    runner = AgentIngestRunner(cfg)
    job, video, scratch = tmp_path / "job", tmp_path / "v.mp4", tmp_path / "scratch"
    command = runner.docker_command(job, "test-container", "isolated-network",
                                    video=video, scratch=scratch)
    assert "test-secret" not in " ".join(command)
    assert "ANTHROPIC_API_KEY" in command
    assert "--read-only" in command and "--cap-drop=ALL" in command
    assert command[command.index("--network") + 1] == "isolated-network"
    assert command[command.index("--dns") + 1] == "127.0.0.1"
    assert command[command.index("--user") + 1] == "1000:1000"
    mounts = [command[i + 1] for i, arg in enumerate(command) if arg == "--mount"]
    assert mounts == [f"type=bind,src={job},dst=/workspace,readonly",
                      f"type=bind,src={job / 'output'},dst=/workspace/output",
                      f"type=bind,src={video},dst=/input/video.mp4,readonly",
                      f"type=bind,src={scratch},dst=/scratch"]
    assert "sha256:fixture" in command
    assert runner.provenance["egress"] == {
        "mode": "allowlist-proxy",
        "hosts": cfg["ingest"]["agentic"]["egress_allowed_hosts"]["claude_code"]}


def test_missing_credential_is_a_shared_failure_not_one_video(agentic, monkeypatch):
    cfg = agentic
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(AgentIngestRunner, "_inspect", staticmethod(lambda _: "sha256:fixture"))
    # Raised before any video is started, so no failure is charged to a video.
    with pytest.raises(FatalAgentError, match="ANTHROPIC_API_KEY"):
        ingest_all(cfg, jobs=2)
    assert not (Path(cfg["paths"]["wiki"]) / "qvhighlights" / ".ingest-failures").exists()


def test_settings_reject_unknown_keys_and_require_a_recorded_template_hash():
    with pytest.raises(HarnessError, match="Unknown or malformed"):
        settings({"agentic": {"nonsense": 1}})
    with pytest.raises(HarnessError, match="no templates_hash"):
        settings({"agentic": {"model": "m"}})
    resolved = settings({"agentic": {"model": "m"}}, ROOT / "templates")
    assert len(resolved["templates_hash"]) == 64
