"""Raw-video queries use the same prediction and trace contract as Wiki queries."""

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from agents.runner import DockerRunner
from vmr.config.models import QueryConfig
from vmr.config.query import load_query_config
from vmr.core.errors import HarnessError, RunFailure
from vmr.core.jsonio import read_json, write_json, write_jsonl
from vmr.core.hashing import file_hash
from vmr.media.probe import probe_durations as real_probe_durations
from vmr.query.experiment import Experiment
from vmr.query.templates import load_query_templates
from vmr.query.video import VideoSource, video_sources
from vmr.query.workspace import video_query_workspace
from vmr.runtime.trace import trace_path
from vmr.runtime.types import AgentResult


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def fixture_probe(monkeypatch):
    monkeypatch.setattr("vmr.media.probe.probe_durations", lambda _: (3.0, 3.0))


class VideoRuntime:
    provenance = {"runtime": "fixture"}

    def run(self, workspace, prompt, stdout, stderr):
        assert {p.name for p in workspace.iterdir()} == {
            "AGENTS.md",
            "task.json",
            "output",
            "video.mp4",
        }
        staged = workspace / "video.mp4"
        assert staged.is_file() and not staged.is_symlink()
        assert staged.read_bytes() == b"video fixture"
        assert staged.stat().st_mode & 0o222 == 0
        assert not (workspace / "wiki").exists()
        assert workspace.name.startswith("vmr-video-")
        assert workspace.parent.name.startswith("vmr-video-private-")
        assert workspace.parent.stat().st_mode & 0o077 == 0
        assert workspace.is_relative_to(Path("/tmp").resolve())
        assert "video.mp4" in prompt
        assert "没有 Wiki" in (workspace / "AGENTS.md").read_text()
        task = read_json(workspace / "task.json")
        write_json(
            workspace / "output/prediction.json",
            {
                "query_id": task["query_id"],
                "video_id": task["video_id"],
                "split": task["split"],
                "moments": [
                    {
                        "start_sec": 1.0,
                        "end_sec": min(2.0, task["duration"]),
                        "score": 0.8,
                    }
                ],
            },
        )
        trace_path(stdout).write_text('{"type":"harness.input"}\n')
        stdout.write_text("done\n")
        stderr.write_text("")
        return AgentResult(0)


def query_config():
    return QueryConfig(
        agent="codex",
        model="fixture",
        container_image="fixture",
        api_key_env="FIXTURE_KEY",
        egress_allowed_hosts=("api.example.com",),
        type="video-only",
        agents_template="query_agents.video_only.md",
        prompt_template="query_prompt.video_only.md",
    )


def test_yaml_selects_video_only_templates():
    config = load_query_config(ROOT / "configs/query/video-only.yaml").query
    assert config.type == "video-only"
    assert config.agents_template == "query_agents.video_only.md"
    assert config.prompt_template == "query_prompt.video_only.md"


def test_video_only_templates_require_type_declaration(tmp_path):
    (tmp_path / "agents.md").write_text("Read wiki/wiki.md")
    (tmp_path / "prompt.md").write_text(
        "<!-- vmr-query-type: video-only -->\nRead video.mp4"
    )
    with pytest.raises(HarnessError, match="must declare its type"):
        load_query_templates(
            tmp_path,
            query_type="video-only",
            templates={"agents": "agents.md", "prompt": "prompt.md"},
        )
    (tmp_path / "agents.md").write_text(
        "<!-- vmr-query-type: video-only -->\nRead wiki/wiki.md"
    )
    with pytest.raises(HarnessError, match="refer to video.mp4"):
        load_query_templates(
            tmp_path,
            query_type="video-only",
            templates={"agents": "agents.md", "prompt": "prompt.md"},
        )


def setup_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    write_json(
        dataset / "dataset.json",
        {"name": "fixture", "splits": {"dev": {"has_ground_truth": False}}},
    )
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video fixture")
    write_jsonl(
        dataset / "videos.jsonl",
        [
            {
                "video_id": "real-video",
                "video_path": str(video),
                "duration": 3.0,
                "split": "dev",
            }
        ],
    )
    write_jsonl(
        dataset / "queries.jsonl",
        [
            {
                "query_id": "real-query",
                "video_id": "real-video",
                "query": "red scene",
                "split": "dev",
            }
        ],
    )
    return dataset, video


def test_video_query_without_wikiset_preserves_output_and_trace(tmp_path):
    dataset, video = setup_dataset(tmp_path)
    options = dict(
        dataset=dataset,
        split="dev",
        video_root=tmp_path,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=VideoRuntime(),
    )
    with Experiment(query_config(), **options) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["status"] == "success"
        assert result["video_hash"] == experiment.metadata["video_hashes"]["real-video"]
        assert "artifact_id" not in result
        assert (experiment.root / result["trace_path"]).exists()
        prediction = read_json(experiment.root / "predictions/real-query.json")
        assert prediction == {
            "query_id": "real-query",
            "video_id": "real-video",
            "split": "dev",
            "moments": [{"start_sec": 1.0, "end_sec": 2.0, "score": 0.8}],
        }
        assert not experiment.runs.exists()
    with Experiment(query_config(), **options) as experiment:
        assert experiment.run(experiment.queries[0])["attempts"] == 1
    video.write_bytes(b"changed")
    with pytest.raises(HarnessError, match="Video changed|Experiment inputs"):
        with Experiment(query_config(), **options):
            pass


@pytest.mark.parametrize("change", ["content", "missing", "symlink", "directory"])
def test_video_workspace_rejects_changed_staged_video(tmp_path, change):
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    video = tmp_path / "source.mp4"
    video.write_bytes(b"video fixture")
    source = VideoSource(video, file_hash(video), 3.0)
    with pytest.raises(RunFailure) as error:
        with video_query_workspace(
            {"query_id": "fixture"}, {"AGENTS.md": "fixture"}, source
        ) as workspace:
            placeholder = workspace / "video.mp4"
            assert placeholder.is_file()
            assert placeholder.read_bytes() == b"video fixture"
            if change == "content":
                placeholder.chmod(0o644)
                placeholder.write_bytes(b"changed")
            else:
                workspace.chmod(0o755)
                placeholder.unlink()
                if change == "symlink":
                    placeholder.symlink_to(empty_file)
                elif change == "directory":
                    placeholder.mkdir()
    assert error.value.kind == "tampered"
    assert not workspace.exists()
    assert empty_file.read_bytes() == b""


def test_video_and_wiki_arguments_are_exclusive(tmp_path):
    dataset, _ = setup_dataset(tmp_path)
    options = dict(
        dataset=dataset,
        split="dev",
        video_root=tmp_path,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=VideoRuntime(),
    )
    with pytest.raises(HarnessError, match="does not use --wiki-set"):
        Experiment(query_config(), wiki_set=tmp_path / "set.json", **options)
    with pytest.raises(HarnessError, match="requires --wiki-set"):
        Experiment(replace(query_config(), type="video-wiki"), **options)
    with pytest.raises(HarnessError, match="requires --video-root"):
        Experiment(
            query_config(), **{k: v for k, v in options.items() if k != "video_root"}
        )


def test_video_source_must_stay_within_explicit_root(tmp_path):
    dataset, video = setup_dataset(tmp_path)
    root = tmp_path / "videos"
    root.mkdir()
    rows = {"real-video": {"video_path": str(video)}}
    with pytest.raises(HarnessError, match="outside --video-root"):
        video_sources(dataset, rows, root)
    rows["real-video"]["video_path"] = "../source.mp4"
    with pytest.raises(HarnessError, match="outside --video-root"):
        video_sources(dataset, rows, root)


@pytest.mark.parametrize("container, stream", [(2.0, 1.5), (1.5, 2.0)])
def test_video_source_uses_container_duration(tmp_path, monkeypatch, container, stream):
    dataset, video = setup_dataset(tmp_path)
    monkeypatch.setattr(
        "vmr.media.probe.probe_durations", lambda _: (container, stream)
    )
    sources = video_sources(
        dataset, {"real-video": {"video_path": str(video)}}, tmp_path
    )
    assert sources["real-video"].duration == container


@pytest.mark.parametrize("container, stream", [(2.0, 1.5), (1.5, 2.0)])
def test_query_duration_is_bounded_by_container(
    tmp_path, monkeypatch, container, stream
):
    dataset, _ = setup_dataset(tmp_path)
    monkeypatch.setattr(
        "vmr.media.probe.probe_durations", lambda _: (container, stream)
    )
    with Experiment(
        query_config(),
        dataset=dataset,
        split="dev",
        video_root=tmp_path,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=VideoRuntime(),
    ) as experiment:
        result = experiment.run(experiment.queries[0])
        assert result["effective_duration"] == container
        prediction = read_json(experiment.root / "predictions/real-query.json")
        assert prediction["moments"][0]["end_sec"] == container


@pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffprobe unavailable")
def test_non_video_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("vmr.media.probe.probe_durations", real_probe_durations)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    secret = tmp_path / ".env"
    secret.write_text("SECRET=value")
    with pytest.raises(HarnessError):
        video_sources(dataset, {"not-video": {"video_path": str(secret)}}, tmp_path)


def test_video_docker_mount_has_no_separate_source_bind(tmp_path, monkeypatch):
    monkeypatch.setenv("FIXTURE_KEY", "fake")
    monkeypatch.setattr(
        DockerRunner, "_inspect", staticmethod(lambda _: "sha256:fixture")
    )
    cfg = query_config().runtime_config()
    runner = DockerRunner({"query": cfg})
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video fixture")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "output").mkdir()
    command = runner.docker_command(workspace, "fixture", "network")
    mounts = [
        command[i + 1] for i, part in enumerate(command[:-1]) if part == "--mount"
    ]
    assert any(
        f"src={workspace},dst=/workspace,readonly" == mount.removeprefix("type=bind,")
        for mount in mounts
    )
    assert all(str(video) not in mount for mount in mounts)
    assert all("/wiki" not in mount for mount in mounts)
