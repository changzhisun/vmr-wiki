"""Real compile algorithms, fake models, one common artifact-only query engine."""

import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from vmr.artifact.artifact import WikiArtifact
from vmr.artifact.wikiset import write_wikiset
from vmr.compiler.pipeline import compile_video
from vmr.config.migrate import from_legacy
from vmr.core.errors import HarnessError
from vmr.core.jsonio import read_json, write_json
from vmr.query.experiment import Experiment
from agents.runner import AgentResult

ROOT = Path(__file__).resolve().parents[2]


class Runtime:
    provenance = {"runtime": "fixture"}

    def run(self, workspace, prompt, stdout, stderr, **kwargs):
        assert set(p.name for p in workspace.iterdir()) == {
            "wiki",
            "task.json",
            "AGENTS.md",
            "output",
        }
        assert not (workspace / "wiki/artifact.json").exists()
        assert not (workspace / "wiki/internal").exists()
        task = read_json(workspace / "task.json")
        assert task["video_id"] != "video"
        write_json(
            workspace / "output/prediction.json",
            dict(
                query_id=task["query_id"],
                video_id=task["video_id"],
                split=task["split"],
                moments=[
                    dict(start_sec=0, end_sec=1, score=0.9, evidence="visible scene")
                ],
            ),
        )
        return AgentResult(0)


def compile_fixture(method, cfg, captioner, tmp_path):
    runtime = None
    if method == "dense":
        from test_ingest_recovery import Captioner

        cfg["ingest"].update(caption_mode="dense", caption_window_frames=1)
        captioner = Captioner()
    elif method == "hierarchical":
        from test_hierarchy import hierarchical, TreeCaptioner

        hierarchical(cfg)
        captioner = TreeCaptioner()
    elif method == "bidirectional":
        from test_bidirectional import config, BidirectionalVLM

        config(cfg)
        captioner = BidirectionalVLM()
    elif method == "agentic":
        from test_agentic import agentic_config, FixtureIngestRunner
        from harness.config import load_config

        old_paths = cfg["paths"]
        cfg = load_config(agentic_config(tmp_path))
        cfg["paths"] = old_paths
        runtime = FixtureIngestRunner()
    app = from_legacy(cfg)
    video = tmp_path / "videos/video.mp4"
    store = tmp_path / "artifacts"
    artifact = compile_video(
        video,
        app.compile,
        store,
        templates=app.storage.templates,
        captioner=captioner,
        runtime=runtime,
    )
    return app, artifact, store


@pytest.mark.parametrize(
    "method", ["simple", "dense", "hierarchical", "bidirectional", "agentic"]
)
def test_every_artifact_can_use_same_query_engine(method, prepared, tmp_path):
    cfg, captioner = prepared
    app, artifact, store = compile_fixture(method, cfg, captioner, tmp_path)
    artifact.verify()
    assert not (artifact.root / "ingest.json").exists()
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    # Raw video, compile settings and build method no longer participate in Query.
    (tmp_path / "videos/video.mp4").unlink()
    experiment = dict(
        dataset=tmp_path / "datasets/qvhighlights",
        split="train",
        wiki_set=wiki_set,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=Runtime(),
    )
    with Experiment(app.query, **experiment) as exp:
        assert exp.run(exp.queries[0])["status"] == "success"
        assert "artifacts" in exp.metadata and "config_hash" not in exp.metadata
    with Experiment(app.query, **experiment) as exp:
        assert exp.run(exp.queries[0])["attempts"] == 1
    from harness.aggregate import aggregate
    from harness.evaluate import evaluate

    output = tmp_path / "experiment/predictions.jsonl"
    aggregate(tmp_path / "experiment/predictions", output)
    assert (
        evaluate(output, dataset_dir=tmp_path / "datasets/qvhighlights", split="train")[
            "num_queries"
        ]
        == 3
    )


@pytest.mark.parametrize("entrypoint", ["api", "cli"])
@pytest.mark.parametrize("schema", ["shared-v3", "named-v2"])
def test_artifact_portability_without_compiler(prepared, tmp_path, entrypoint, schema):
    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    portable = tmp_path / "portable"
    portable.mkdir()
    from vmr.artifact.store import ArtifactStore

    copied_path = ArtifactStore(portable).path(
        artifact.artifact_id(), manifest_hash=artifact.manifest_hash()
    )
    shutil.copytree(artifact.root, copied_path, copy_function=shutil.copy2)
    # copytree retains source permissions. The original artifact and source are then absent.
    copied = WikiArtifact.open(copied_path)
    write_wikiset(
        portable / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": copied},
        store=portable,
    )
    shutil.copytree(tmp_path / "datasets/qvhighlights", portable / "dataset")
    write_json(portable / "query.json", app.query.snapshot())
    import yaml

    q = app.query
    (portable / "query.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 3,
                "storage": {"templates": str(ROOT / "templates"), "runs": "runs"},
                "profiles": {
                    "agent": {
                        "model": q.model,
                        "container_image": q.container_image,
                        "api_key_env": q.api_key_env,
                        "base_url": q.base_url,
                        "egress_allowed_hosts": list(q.egress_allowed_hosts),
                    }
                },
                "query": {
                    "kind": q.agent,
                    "input_mode": q.input_mode,
                    "max_predictions": q.max_predictions,
                    "timeout_sec": q.timeout_sec,
                },
            }
        )
    )
    if schema == "named-v2":
        path = portable / "query.yaml"
        raw = yaml.safe_load(path.read_text())
        raw["version"] = 2
        profile = raw["profiles"].pop("agent")
        profile["kind"] = raw["query"].pop("kind")
        raw["profiles"]["agents"] = {"portable": profile}
        raw["query"]["agent_profile"] = "portable"
        path.write_text(yaml.safe_dump(raw))
    (tmp_path / "videos/video.mp4").unlink()
    from vmr.artifact.integrity import remove_tree

    remove_tree(store)
    # Copy a minimal install: no compiler, compile config, migration adapters, or source video.
    library = portable / "lib"
    for directory in (
        "vmr/core",
        "vmr/query",
        "vmr/artifact",
        "vmr/runtime",
        "vmr/config",
        "vmr/datasets",
        "vmr/cli",
        "agents",
    ):
        shutil.copytree(
            ROOT / directory,
            library / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    (library / "vmr/__init__.py").write_text("")
    shutil.copy2(ROOT / "vmr/__main__.py", library / "vmr/__main__.py")
    for path in (library / "vmr/config").glob("*.py"):
        if path.name not in (
            "__init__.py",
            "models.py",
            "resolve.py",
            "loader.py",
            "query.py",
        ):
            path.unlink()
    script = """
import sys, importlib.abc, runpy
from pathlib import Path
class RejectCompiler(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'harness' or fullname.startswith(('vmr.compiler', 'vmr.compat')):
            raise AssertionError('Query imported a build/legacy dependency: ' + fullname)
sys.meta_path.insert(0, RejectCompiler())
from vmr.config.models import QueryConfig
from vmr.query.experiment import Experiment
from vmr.core.jsonio import read_json, write_json
from agents.runner import AgentResult
class Runtime:
    provenance = {'runtime':'portable'}
    def run(self, workspace, *args, **kwargs):
        t = read_json(workspace / 'task.json')
        write_json(workspace / 'output/prediction.json', dict(query_id=t['query_id'],video_id=t['video_id'],split=t['split'],moments=[]))
        return AgentResult(0)
r = Path.cwd()
# Exercise the production Docker command in the stripped install too. No host
# repository helpers may be mounted; inspect is the only external call stubbed.
import os
from agents.runner import DockerRunner
q = QueryConfig(**read_json(r/'query.json'))
os.environ[q.api_key_env] = 'fixture-no-api-call'
DockerRunner._inspect = staticmethod(lambda image: 'sha256:fixture')
runner = DockerRunner({'query': q.runtime_config()})
workspace = r/'docker-workspace'
(workspace/'output').mkdir(parents=True)
command = runner.docker_command(workspace, 'portable', 'fixture-network')
sources = []
for i, token in enumerate(command):
    if token == '--mount':
        fields = dict(item.split('=', 1) for item in command[i+1].split(',') if '=' in item)
        if fields.get('type') == 'bind':
            source = Path(fields['src'])
            assert source.exists(), source
            sources.append(source)
assert set(sources) == {workspace, workspace/'output'}
assert not any(token.startswith('PATH=') for token in command)
if sys.argv[2] == 'api':
    with Experiment(QueryConfig(**read_json(r/'query.json')),dataset=r/'dataset',split='train',wiki_set=r/'set.json',root=r/'experiment',templates=Path(sys.argv[1]),runs=r/'runs',runtime=Runtime()) as e:
        assert e.run(e.queries[0])['status'] == 'success'
else:
    # Replace only the external container runtime, exercising the real config
    # loader, module entrypoint, parser, Experiment and parallel batch runner.
    import vmr.query.experiment as experiment_module
    experiment_module.query_runtime = lambda config: Runtime()
    sys.argv = ['vmr', 'query', '--config', str(r/'query.yaml'),
                '--dataset', str(r/'dataset'), '--split', 'train',
                '--wiki-set', str(r/'set.json'), '--experiment', str(r/'experiment'),
                '--jobs', '2']
    runpy.run_module('vmr', run_name='__main__')
    records = list((r/'experiment/run_metadata').glob('*.json'))
    assert len(records) == 3
    assert all(read_json(p)['status'] == 'success' for p in records)
    assert len(list((r/'experiment/predictions').glob('*.json'))) == 3
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(ROOT / "templates"), entrypoint],
        cwd=portable,
        env={**os.environ, "PYTHONPATH": str(library)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_external_compiler_requires_only_registration(tmp_path):
    from vmr.compiler.registry import register, COMPILERS
    from vmr.compiler.protocol import CompilerConfig, CompileResult
    from vmr.config.models import CompileConfig, MediaConfig

    class External:
        name, version = "test-external", 1

        def parse_config(self, raw):
            return CompilerConfig(dict(raw))

        def content_identity(self, config, source):
            return {"rule": 1}

        def compile(self, ctx):
            ctx.output.mkdir()
            (ctx.output / "wiki.md").write_text("# Video\nExternal wiki")
            (ctx.output / "custom.txt").write_text("arbitrary text surface")
            (ctx.output / "secret.txt").write_text("private")
            return CompileResult(
                ctx.output, 3, ("wiki.md", "custom.txt"), (), {"origin": "external"}
            )

    register(External())
    try:
        video = tmp_path / "video"
        video.write_bytes(b"external source")
        config = CompileConfig("test-external", MediaConfig(), CompilerConfig({}))
        artifact = compile_video(video, config, tmp_path / "store")
        assert set(artifact.public_hashes()) == {"wiki.md", "custom.txt"}
        assert (artifact.root / "internal/secret.txt").exists()
        assert (
            compile_video(video, config, tmp_path / "store").artifact_id()
            == artifact.artifact_id()
        )
    finally:
        COMPILERS.pop("test-external")


@pytest.mark.parametrize(
    "behavior,kind",
    [
        ("invalid", "invalid_output"),
        ("extra", "invalid_output"),
        ("symlink", "invalid_output"),
        ("mutate", "tampered"),
        ("timeout", "timeout"),
        ("nonzero", "agent_error"),
    ],
)
def test_artifact_query_preserves_failure_classification(
    behavior, kind, prepared, tmp_path
):
    from test_query_pipeline import ProcessFixtureRunner

    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    with Experiment(
        app.query,
        dataset=tmp_path / "datasets/qvhighlights",
        split="train",
        wiki_set=wiki_set,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=ProcessFixtureRunner(behavior),
    ) as exp:
        result = exp.run(exp.queries[0])
        assert result["failure_kind"] == kind
        assert not (
            exp.root / "predictions" / (exp.queries[0]["query_id"] + ".json")
        ).exists()
    artifact.verify()


def test_query_resume_pins_artifacts_and_query_settings(prepared, tmp_path):
    from dataclasses import replace

    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    options = dict(
        dataset=tmp_path / "datasets/qvhighlights",
        split="train",
        wiki_set=wiki_set,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=Runtime(),
    )
    with Experiment(app.query, **options) as exp:
        assert exp.run(exp.queries[0])["status"] == "success"
    with pytest.raises(HarnessError, match="changed"):
        with Experiment(replace(app.query, max_predictions=3), **options):
            pass
    changed_cfg = copy.deepcopy(cfg)
    changed_cfg["ingest"]["vlm"]["prompt"] += " Changed compiler prompt."
    changed_app = from_legacy(changed_cfg)
    changed = compile_video(
        tmp_path / "videos/video.mp4",
        changed_app.compile,
        store,
        templates=ROOT / "templates",
        captioner=captioner,
    )
    assert changed.artifact_id() != artifact.artifact_id()
    other_set = write_wikiset(
        tmp_path / "other.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": changed},
        store=store,
    )
    with pytest.raises(HarnessError, match="changed"):
        with Experiment(app.query, **{**options, "wiki_set": other_set}):
            pass
    # The original set remains usable even after another compiler config/build exists.
    with Experiment(app.query, **options) as exp:
        assert exp.run(exp.queries[0])["attempts"] == 1


def test_legacy_experiment_is_not_implicitly_upgraded(prepared, tmp_path):
    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    write_json(
        tmp_path / "experiment/experiment.json", {"version": 2, "alias_secret": "old"}
    )
    with pytest.raises(HarnessError, match="Legacy experiment"):
        with Experiment(
            app.query,
            dataset=tmp_path / "datasets/qvhighlights",
            split="train",
            wiki_set=wiki_set,
            root=tmp_path / "experiment",
            templates=ROOT / "templates",
            runs=tmp_path / "runs",
            runtime=Runtime(),
        ):
            pass
    assert read_json(tmp_path / "experiment/experiment.json")["version"] == 2


def test_explicit_legacy_migration_preserves_public_bytes(frozen, tmp_path):
    from vmr.compat.artifact_migrate import migrate_legacy

    cfg, _ = frozen
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights/videos/video"
    before = (root / "wiki.md").read_bytes()
    artifact = migrate_legacy(root, tmp_path / "artifacts")
    assert (artifact.root / "public/wiki.md").read_bytes() == before
    assert (artifact.root / "internal/ingest.json").exists()
    assert (root / "wiki.md").read_bytes() == before


def test_compile_failure_keeps_checkpoint_and_never_publishes(prepared, tmp_path):
    cfg, _ = prepared

    class FailSecond:
        calls = 0

        def caption(self, image):
            self.calls += 1
            if self.calls == 2:
                raise HarnessError("interrupted caption")
            return "A scene"

    app = from_legacy(cfg)
    store = tmp_path / "artifacts"
    failed = FailSecond()
    with pytest.raises(HarnessError, match="interrupted caption"):
        compile_video(
            tmp_path / "videos/video.mp4", app.compile, store, captioner=failed
        )
    assert not list(store.glob("sha256-*"))
    assert list((store / ".work/.compile-checkpoints").rglob("identity.json"))

    class Finish:
        calls = 0

        def caption(self, image):
            self.calls += 1
            return "A scene"

    finish = Finish()
    artifact = compile_video(
        tmp_path / "videos/video.mp4", app.compile, store, captioner=finish
    )
    assert finish.calls == 2  # first completed frame/window was reused
    assert (artifact.root / "internal/build.json").exists()
    assert not list(artifact.root.rglob("ingest.json"))
    artifact.verify()


def test_cli_compile_query_evaluate_without_model_api(prepared, tmp_path, monkeypatch):
    import yaml
    from vmr.cli import main

    monkeypatch.chdir(tmp_path)

    cfg, _ = prepared
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["version"] = 3
    section = raw.pop("wiki")
    section["compiler"] = "simple"
    section.pop("method")
    section["method_config"] = {}
    raw["compile"] = section
    raw["profiles"]["vlms"]["qwen_default"]["prompt"] = "Describe frame"
    raw["profiles"]["agent"]["model"] = "fixture"
    raw["storage"].update(
        root=str(tmp_path), artifacts="artifacts", templates=str(ROOT / "templates")
    )
    raw["storage"].pop("wikis")

    class Captioner:
        def __init__(self, *args, **kwargs):
            pass

        def caption(self, image):
            return "A scene"

    monkeypatch.setattr("vmr.compiler.context.VLMClient", Captioner)
    monkeypatch.setattr("vmr.runtime.docker.DockerRunner", lambda cfg: Runtime())
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump(raw))
    dataset = str(tmp_path / "datasets/qvhighlights")
    main(
        [
            "compile",
            "--config",
            str(path),
            "--dataset",
            dataset,
            "--split",
            "train",
            "--output-set",
            str(tmp_path / "set.json"),
        ]
    )
    # Drop compile configuration before exercising the public Query CLI.
    raw.pop("compile")
    raw["profiles"].pop("vlms")
    path.write_text(yaml.safe_dump(raw))
    main(
        [
            "query",
            "--config",
            str(path),
            "--dataset",
            dataset,
            "--split",
            "train",
            "--wiki-set",
            str(tmp_path / "set.json"),
            "--experiment",
            str(tmp_path / "experiment"),
            "--jobs",
            "2",
        ]
    )
    main(
        [
            "evaluate",
            "--dataset",
            dataset,
            "--split",
            "train",
            "--experiment",
            str(tmp_path / "experiment"),
        ]
    )
    assert read_json(tmp_path / "experiment/metrics.json")["failed_runs"] == 0


@pytest.mark.parametrize("failure_point", ["publication", "cache-index", "root-seal"])
def test_publication_retry_reuses_completed_build(
    prepared, tmp_path, monkeypatch, failure_point
):
    from vmr.compiler import pipeline
    from vmr.artifact.store import ArtifactStore

    cfg, captioner = prepared
    app = from_legacy(cfg)
    store = tmp_path / "artifacts"
    original_publish = ArtifactStore.publish
    original_write = pipeline.write_json
    original_chmod = Path.chmod

    def fail_publish(*args, **kwargs):
        raise OSError("publication unavailable")

    def fail_index(path, value):
        if path.parent.name == ".builds":
            raise OSError("cache index unavailable")
        return original_write(path, value)

    def fail_seal(path, mode, *args, **kwargs):
        if path.parent.name.startswith("sha256-") and mode == 0o555:
            raise OSError("interrupted root seal")
        return original_chmod(path, mode, *args, **kwargs)

    if failure_point == "root-seal":
        monkeypatch.setattr(Path, "chmod", fail_seal)
    elif failure_point == "publication":
        monkeypatch.setattr(ArtifactStore, "publish", fail_publish)
    else:
        monkeypatch.setattr(pipeline, "write_json", fail_index)
    with pytest.raises(OSError):
        compile_video(
            tmp_path / "videos/video.mp4", app.compile, store, captioner=captioner
        )
    calls = captioner.calls
    assert list((store / ".work").glob("*/result.json"))
    monkeypatch.setattr(ArtifactStore, "publish", original_publish)
    monkeypatch.setattr(pipeline, "write_json", original_write)
    monkeypatch.setattr(Path, "chmod", original_chmod)
    artifact = compile_video(
        tmp_path / "videos/video.mp4", app.compile, store, captioner=captioner
    )
    artifact.verify()
    assert captioner.calls == calls
    assert len(list(store.glob("sha256-*"))) == 1


def test_fresh_compiles_share_content_identity_and_pin_build_cache(prepared, tmp_path):
    cfg, captioner = prepared
    app, first, store = compile_fixture("simple", cfg, captioner, tmp_path)
    fresh_store = tmp_path / "fresh-store"
    second = compile_video(
        tmp_path / "videos/video.mp4", app.compile, fresh_store, captioner=captioner
    )
    assert second.public_hashes() == first.public_hashes()
    assert second.artifact_id() == first.artifact_id()
    # Remove only the index: a fresh build must preserve the earlier audit record.
    for index in (store / ".builds").glob("*.json"):
        index.unlink()
    rebuilt = compile_video(
        tmp_path / "videos/video.mp4", app.compile, store, captioner=captioner
    )
    first.verify()
    assert rebuilt.artifact_id() == first.artifact_id()
    assert rebuilt.manifest_hash() != first.manifest_hash()
    cached = compile_video(
        tmp_path / "videos/video.mp4", app.compile, store, captioner=captioner
    )
    assert cached.root == rebuilt.root
    from vmr.artifact.integrity import remove_tree

    remove_tree(fresh_store)


def test_query_full_checks_are_limited_to_run_boundaries(
    prepared, tmp_path, monkeypatch
):
    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    with Experiment(
        app.query,
        dataset=tmp_path / "datasets/qvhighlights",
        split="train",
        wiki_set=wiki_set,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=Runtime(),
    ) as exp:
        checks = []
        original = WikiArtifact.verify

        def verify(self):
            checks.append(self.root)
            return original(self)

        monkeypatch.setattr(WikiArtifact, "verify", verify)
        assert exp.run(exp.queries[0])["status"] == "success"
        assert checks == [artifact.root, artifact.root]


@pytest.mark.parametrize("when", ["before", "during"])
@pytest.mark.parametrize("surface", ["public", "internal"])
def test_query_rejects_source_tampering(prepared, tmp_path, when, surface):
    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )

    def tamper():
        target = next(p for p in (artifact.root / surface).rglob("*") if p.is_file())
        target.chmod(0o644)
        target.write_bytes(target.read_bytes() + b"changed")
        target.chmod(0o444)

    class TamperingRuntime(Runtime):
        def run(self, *args, **kwargs):
            if when == "during":
                tamper()
            return super().run(*args, **kwargs)

    with Experiment(
        app.query,
        dataset=tmp_path / "datasets/qvhighlights",
        split="train",
        wiki_set=wiki_set,
        root=tmp_path / "experiment",
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=TamperingRuntime(),
    ) as exp:
        if when == "before":
            tamper()
        with pytest.raises(HarnessError, match="integrity"):
            exp.run(exp.queries[0])
        assert not list((exp.root / "predictions").glob("*.json"))


@pytest.mark.parametrize("parallel", [False, True])
def test_agentic_build_logs_remain_separate(prepared, tmp_path, parallel):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from threading import Barrier
    from test_agentic import agentic_config, FixtureIngestRunner
    from vmr.compat.config import load_config
    from vmr.artifact.integrity import remove_tree

    app = from_legacy(load_config(agentic_config(tmp_path)))
    barrier = Barrier(2) if parallel else None
    logs = {}

    class LoggingRunner(FixtureIngestRunner):
        def __init__(self, label):
            super().__init__()
            self.label = label

        def run(self, job, prompt, stdout, stderr, **kwargs):
            if barrier:
                barrier.wait(timeout=10)
            result = super().run(job, prompt, stdout, stderr, **kwargs)
            stdout.write_text(self.label)
            logs[self.label] = stdout
            return result

    store = tmp_path / "log-artifacts"

    def build(i):
        from vmr.compiler.protocol import CompilerConfig

        settings = copy.deepcopy(app.compile.method.settings)
        settings["agentic"]["model"] = f"fixture-agent-{i}"
        config = replace(app.compile, method=CompilerConfig(settings))
        return compile_video(
            tmp_path / "videos/video.mp4",
            config,
            store,
            templates=app.storage.templates,
            runtime=LoggingRunner(str(i)),
        )

    try:
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as pool:
                artifacts = list(pool.map(build, range(2)))
        else:
            artifacts = [build(i) for i in range(2)]
        assert len(set(logs.values())) == 2
        for label, path in logs.items():
            assert path.read_text() == label
            assert path.parent.parent.parent == store / ".work"
        for artifact in artifacts:
            artifact.verify()
    finally:
        remove_tree(store)


def test_aggregate_uses_exact_snapshot_dataset_path(prepared, tmp_path):
    from vmr.evaluation.aggregate import aggregate
    from vmr.evaluation.evaluate import evaluate
    from vmr.evaluation.results import experiment_context

    cfg, captioner = prepared
    app, artifact, store = compile_fixture("simple", cfg, captioner, tmp_path)
    directory = tmp_path / "downloaded_annotations_v2"
    shutil.copytree(tmp_path / "datasets/qvhighlights", directory)
    wiki_set = write_wikiset(
        tmp_path / "set.json",
        dataset="qvhighlights",
        split="train",
        artifacts={"video": artifact},
        store=store,
    )
    root = tmp_path / "experiment"
    with Experiment(
        app.query,
        dataset=directory,
        split="train",
        wiki_set=wiki_set,
        root=root,
        templates=ROOT / "templates",
        runs=tmp_path / "runs",
        runtime=Runtime(),
    ) as experiment:
        for query in experiment.queries:
            assert experiment.run(query)["status"] == "success"
    output = root / "predictions.jsonl"
    assert len(aggregate(root / "predictions", output)) == 3
    assert evaluate(output, dataset_dir=directory, split="train")["num_queries"] == 3
    override = tmp_path / "override"
    shutil.copytree(directory, override)
    shutil.rmtree(directory)
    # Explicit override wins even when the saved path has become unavailable.
    assert experiment_context(root, override, None)[0]["name"] == "qvhighlights"


def test_compile_lock_covers_build_and_publication(prepared, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from vmr.artifact.store import ArtifactStore

    cfg, captioner = prepared
    app = from_legacy(cfg)
    started, release = Event(), Event()
    video, store = tmp_path / "videos/video.mp4", tmp_path / "artifacts"

    class BlockingCaptioner:
        def caption(self, path):
            started.set()
            assert release.wait(timeout=10)
            return captioner.caption(path)

    def contender():
        return compile_video(video, app.compile, store, captioner=captioner)

    original_publish = ArtifactStore.publish

    def checked_publish(*args, **kwargs):
        with pytest.raises(HarnessError, match="Compile in progress"):
            contender()
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "publish", checked_publish)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            compile_video, video, app.compile, store, captioner=BlockingCaptioner()
        )
        try:
            assert started.wait(timeout=10)
            with pytest.raises(HarnessError, match="Compile in progress"):
                contender()
        finally:
            release.set()
        artifact = future.result(timeout=15)
    assert contender().root == artifact.root
