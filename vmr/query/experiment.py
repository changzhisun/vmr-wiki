"""Experiment identity pins Wiki artifacts or raw videos and query inputs."""

from pathlib import Path
import subprocess
from vmr.core.errors import HarnessError
from vmr.core.hashing import object_hash, file_hash
from vmr.artifact.wikiset import WikiSet
from .video import video_sources
from vmr.runtime.docker import query_runtime
from vmr.runtime.protocol import AgentRuntime
from vmr.datasets.manifest import load_dataset, select_split, load_query_inputs
from .aliases import Aliases
from .templates import load_query_templates
from .repository import RunRepository
from .engine import QueryEngine


class Experiment:
    def __init__(
        self,
        config,
        *,
        dataset,
        split,
        wiki_set=None,
        video_root=None,
        root,
        templates,
        runs,
        runtime: AgentRuntime | None = None,
    ):
        self.config = config
        self.root, self.runs, self.dataset = Path(root), Path(runs), Path(dataset)
        metadata = load_dataset(self.dataset)
        self.split = select_split(metadata, split)
        self.videos, self.queries = load_query_inputs(
            self.dataset, self.split, metadata
        )
        self.query_index = {q["query_id"]: q for q in self.queries}
        self.wiki_set = None
        self.video_sources = None
        self.video_root = None
        if config.type == "video-wiki":
            if wiki_set is None:
                raise HarnessError("Wiki query requires --wiki-set")
            self.wiki_set = (
                wiki_set if isinstance(wiki_set, WikiSet) else WikiSet.open(wiki_set)
            )
            if (self.wiki_set.data["dataset"], self.wiki_set.data["split"]) != (
                metadata["name"],
                self.split,
            ):
                raise HarnessError("WikiSet dataset/split mismatch")
            if self.wiki_set.data["artifacts"].keys() != self.videos.keys():
                raise HarnessError(
                    "WikiSet must contain exactly the selected split videos"
                )
            self.wiki_set.verify()
        else:
            if wiki_set is not None:
                raise HarnessError("Video query does not use --wiki-set")
            if video_root is None:
                raise HarnessError("Video query requires --video-root")
            self.video_root = Path(video_root).resolve()
            self.video_sources = video_sources(
                self.dataset, self.videos, self.video_root
            )
        self.templates = load_query_templates(
            Path(templates),
            text_only=config.input_mode == "text",
            query_type=config.type,
            templates={
                "agents": config.agents_template,
                "prompt": config.prompt_template,
                "text": {
                    "agents": config.text_agents_template,
                    "prompt": config.text_prompt_template,
                },
            },
        )
        self.runner = runtime if runtime is not None else query_runtime(config)
        source_root = Path(__file__).resolve().parents[2]
        files = [
            p
            for d in (
                "vmr/query",
                "vmr/runtime",
                "vmr/media",
                "vmr/artifact",
                "vmr/core",
                "vmr/datasets",
                "agents",
            )
            for p in (source_root / d).rglob("*.py")
        ]
        source_hash = object_hash(
            {str(p.relative_to(source_root)): file_hash(p) for p in sorted(files)}
        )
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            capture_output=True,
            text=True,
        )
        self.metadata = dict(
            version=3,
            query_type=config.type,
            dataset=metadata["name"],
            split=self.split,
            agent=config.agent,
            model=config.model,
            query_config_hash=object_hash(config.snapshot()),
            source_hash=source_hash,
            templates_hash=object_hash(self.templates),
            dataset_hash=object_hash(metadata),
            videos_hash=object_hash(self.videos),
            queries_hash=object_hash(self.queries),
            runtime=self.runner.provenance,
        )
        if self.wiki_set is not None:
            self.metadata.update(
                wiki_set_hash=self.wiki_set.fingerprint(),
                artifacts=dict(self.wiki_set.data["artifacts"]),
                artifact_manifest_hashes=dict(self.wiki_set.data["manifest_hashes"]),
            )
        else:
            self.metadata["video_hashes"] = {
                vid: source.digest for vid, source in self.video_sources.items()
            }
            self.metadata["video_durations"] = {
                vid: source.duration for vid, source in self.video_sources.items()
            }
            self.metadata["video_root"] = str(self.video_root)
        # Git commit is provenance, not an execution dependency on compiler edits.
        self.snapshot = dict(
            schema_version=1,
            query=config.snapshot(),
            dataset_path=str(self.dataset.resolve()),
            dataset=metadata["name"],
            split=self.split,
            source_commit=git.stdout.strip() or None,
        )
        self.repository = RunRepository(self.root)

    def __enter__(self):
        self.repository.enter(self.metadata, self.snapshot, self.templates)
        self.aliases = Aliases(self.metadata["alias_secret"], self.split, self.queries)
        self.engine = QueryEngine(
            config=self.config,
            root=self.root,
            runs=self.runs,
            query_index=self.query_index,
            split=self.split,
            videos=self.videos,
            wiki_set=self.wiki_set,
            video_sources=self.video_sources,
            metadata=self.metadata,
            templates=self.templates,
            runtime=self.runner,
            aliases=self.aliases,
        )
        return self

    def __exit__(self, *_):
        self.repository.close()

    def run(self, query, *, cancel_event=None):
        if not hasattr(self, "engine"):
            raise HarnessError("Enter the experiment context before running queries")
        return self.engine.run(query, cancel_event=cancel_event)
