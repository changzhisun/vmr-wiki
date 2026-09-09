from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import yaml

from agents.runner import DockerRunner
from harness.alias import Aliases, new_secret
from harness.common import (AGENT_FAILURE_KINDS, HarnessError, RunFailure, atomic_text, cli,
                            file_hash, identifier, ingest_content_hash, now, object_hash,
                            read_json, write_json)
from harness.config import dataset_path, load_config
from harness.dataset import dataset_context, load_query_inputs
from harness.freeze import split_not_ready_error, verify_wiki, wiki_readiness
from harness.validate import validate_prediction
from harness.workspace import query_workspace


def git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


class Experiment:
    def __init__(self, cfg: dict, name: str, *, split: str | None = None, runner=None):
        self.cfg = cfg = deepcopy(cfg)
        self.root = Path(cfg["paths"]["results"]) / identifier(name, "experiment")
        self.dataset, dataset_metadata, self.split = dataset_context(cfg, split)
        cfg["dataset"]["split"] = self.split
        self.videos, self.queries = load_query_inputs(self.dataset, self.split, dataset_metadata)
        self.query_index = {q["query_id"]: q for q in self.queries}
        self.wiki_root = dataset_path(cfg, "wiki") / "videos"
        missing, unfrozen = wiki_readiness(self.wiki_root, self.videos)
        if missing or unfrozen:
            raise split_not_ready_error(cfg["dataset"]["name"], self.split, missing, unfrozen)
        self.freeze = {"dataset": dataset_metadata["name"], "split": self.split, "videos": {}}
        for vid, video in self.videos.items():
            ingest_meta = read_json(self.wiki_root / vid / "ingest.json")
            seal = verify_wiki(self.wiki_root / vid)
            if ingest_content_hash({"ingest": ingest_meta["ingest_config"]}) != ingest_content_hash(cfg) or seal["video_id"] != vid:
                raise HarnessError(f"Frozen dataset wiki mismatch: {vid}")
            self.freeze["videos"][vid] = seal["wiki_hash"]
            if abs(seal["duration"] - video["duration"]) > 0.1:
                raise HarnessError(f"Frozen video duration mismatch: {vid}")
        self.templates = {name: (Path(cfg["paths"]["templates"]) / name).read_text(encoding="utf-8")
                          for name in ("AGENTS.md", "query_prompt.md")}
        self.runner = runner if runner is not None else DockerRunner(cfg)
        source_root = Path(__file__).resolve().parents[1]
        source_files = {str(p.relative_to(source_root)): file_hash(p)
                        for folder in ("harness", "adapters", "agents")
                        for p in sorted((source_root / folder).glob("*.py"))}
        self.metadata = {"version": 2, "dataset": cfg["dataset"]["name"], "split": self.split,
                         "agent": cfg["query"]["agent"], "model": cfg["query"]["model"],
                         "config_hash": object_hash(cfg), "git_commit": git_commit(),
                         "source_hash": object_hash(source_files),
                         "templates_hash": object_hash(self.templates),
                         "dataset_hash": object_hash(dataset_metadata),
                         "videos_hash": object_hash(self.videos),
                         "queries_hash": object_hash(self.queries),
                         "frozen_videos": self.freeze["videos"],
                         "freeze_hash": object_hash(self.freeze),
                         "runtime": self.runner.provenance}

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = self.root / ".experiment.lock"
        try:
            with self.lock.open("x") as stream:
                stream.write(now())
        except FileExistsError as exc:
            raise HarnessError("Experiment is already running (or has a stale .experiment.lock)") from exc
        try:
            manifest = self.root / "experiment.json"
            if manifest.exists():
                saved = read_json(manifest)
                if "alias_secret" not in saved:
                    raise HarnessError("Experiment predates workspace identifier aliasing; "
                                       "use a new experiment name")
                # Adopted before the equality check: the secret is minted once and
                # must survive resume, or every alias would change mid-experiment.
                self.metadata["alias_secret"] = saved["alias_secret"]
                if saved != self.metadata:
                    raise HarnessError("Experiment inputs/config/agent/code changed; use a new experiment name")
                if yaml.safe_load((self.root / "config.yaml").read_text()) != self.cfg:
                    raise HarnessError("Saved experiment config has changed")
            else:
                if any(p.name != self.lock.name for p in self.root.iterdir()):
                    raise HarnessError("Experiment directory contains untracked results")
                self.metadata["alias_secret"] = new_secret()
                write_json(manifest, self.metadata)
                atomic_text(self.root / "config.yaml", yaml.safe_dump(self.cfg, sort_keys=True, allow_unicode=True))
                (self.root / "templates").mkdir()
                for name, content in self.templates.items():
                    atomic_text(self.root / "templates" / name, content)
            for name in ("predictions", "run_metadata", "logs"):
                (self.root / name).mkdir(exist_ok=True)
            self.aliases = Aliases(self.metadata["alias_secret"], self.split, self.queries)
            return self
        except BaseException:
            self.lock.unlink(missing_ok=True)
            raise

    def __exit__(self, *_):
        self.lock.unlink(missing_ok=True)

    def run(self, query: dict) -> dict:
        if self.query_index.get(query.get("query_id")) != query:
            raise HarnessError("Query does not belong to this experiment's selected split")
        qid = query["query_id"]
        metadata_path = self.root / "run_metadata" / f"{qid}.json"
        prediction_path = self.root / "predictions" / f"{qid}.json"
        attempts, superseded = 1, []
        if metadata_path.exists():
            previous = read_json(metadata_path)
            if previous.get("dataset") != self.metadata["dataset"] or previous.get("split") != self.split:
                raise HarnessError("Run metadata belongs to a different dataset/split")
            if previous["status"] == "running":
                previous.update(status="failed", failure_kind="interrupted", finished_at=now(),
                                error="Previous harness process ended without finalizing this attempt")
                prediction_path.unlink(missing_ok=True)
                write_json(metadata_path, previous)
            if previous["status"] == "success":
                if file_hash(prediction_path) != previous["prediction_hash"]:
                    raise HarnessError(f"Saved prediction was modified: {qid}")
                return previous
            # The agent's own failure stays terminal: it never gets a second
            # process at the same query. A harness or interrupt failure is not
            # the agent's score, so that attempt is retried once its cause is
            # fixed; the superseded attempts stay in the record so a retried
            # query is never mistaken for a first attempt. Records written
            # before failure kinds existed carry none and remain terminal.
            kind = previous.get("failure_kind")
            if kind is None or kind in AGENT_FAILURE_KINDS:
                return previous
            attempts = previous.get("attempts", 1) + 1
            superseded = [*previous.get("superseded_failures", []),
                          {"kind": kind, "error": previous.get("error"),
                           "finished_at": previous.get("finished_at")}]
            prediction_path.unlink(missing_ok=True)
        elif prediction_path.exists():
            raise HarnessError(f"Prediction exists without run metadata: {qid}")
        wiki = self.wiki_root / query["video_id"]
        seal = verify_wiki(wiki)
        if seal["wiki_hash"] != self.freeze["videos"][query["video_id"]]:
            raise HarnessError("Wiki version changed after experiment initialization")
        if not hasattr(self, "aliases"):
            raise HarnessError("Enter the experiment context before running queries")
        task = self.aliases.task(query, self.cfg["query"]["max_predictions"])
        # The secret lives in experiment.json only; per-run records do not repeat it.
        metadata = {key: value for key, value in self.metadata.items() if key != "alias_secret"}
        metadata.update({"query_id": qid, "video_id": query["video_id"],
                         "task_query_id": task["query_id"], "task_video_id": task["video_id"],
                         "wiki_hash": seal["wiki_hash"], "frames_jsonl_hash": seal["frames_jsonl_hash"],
                         "status": "running", "failure_kind": None, "exit_code": None,
                         "timed_out": False, "attempts": attempts,
                         "superseded_failures": superseded,
                         "started_at": now(), "finished_at": None})
        write_json(metadata_path, metadata)
        try:
            prediction = None
            with query_workspace(query, task, wiki, self.templates,
                                 Path(self.cfg["paths"]["runs"])) as workspace:
                result = self.runner.run(workspace, self.templates["query_prompt.md"],
                                         self.root / "logs" / f"{qid}.stdout.log",
                                         self.root / "logs" / f"{qid}.stderr.log")
                metadata.update(exit_code=result.exit_code, timed_out=result.timed_out)
                if result.timed_out:
                    raise RunFailure("Agent timed out", "timeout")
                if result.exit_code != 0:
                    raise RunFailure(f"Agent exited with code {result.exit_code}", "agent_error")
                prediction = self.read_prediction(workspace / "output", query, task, seal)
            # Publish only after immutable-input checks and workspace cleanup succeeded.
            write_json(prediction_path, prediction)
            metadata.update(status="success", prediction_hash=file_hash(prediction_path))
        except RunFailure as exc:
            metadata.update(status="failed", failure_kind=exc.kind, error=f"{exc}")
        except Exception as exc:
            # Not a score. Record the cause, then let the caller stop the batch
            # instead of charging a harness or infrastructure fault to the agent.
            metadata.update(status="failed", failure_kind="harness_error",
                            error=f"{type(exc).__name__}: {exc}")
            raise
        except BaseException:
            metadata.update(status="failed", failure_kind="interrupted", error="Harness interrupted")
            raise
        finally:
            metadata["finished_at"] = now()
            write_json(metadata_path, metadata)
        return metadata

    def read_prediction(self, output: Path, query: dict, task: dict, seal: dict) -> dict:
        """Read the agent's only output. Every rejection here is the agent's own.

        The agent echoes the aliases it was given, so they are validated as
        given and then replaced by the real identifiers on the way to disk;
        aggregation and evaluation never see an alias.
        """
        try:
            prediction_file = output / "prediction.json"
            if sorted(p.name for p in output.iterdir()) != ["prediction.json"]:
                raise HarnessError("Expected exactly output/prediction.json")
            if prediction_file.is_symlink() or not prediction_file.is_file():
                raise HarnessError("Prediction must be a regular file, not a symlink")
            if prediction_file.stat().st_size > 1024 * 1024:
                raise HarnessError("Prediction exceeds 1 MiB")
            prediction = validate_prediction(
                read_json(prediction_file), query_id=task["query_id"], split=task["split"],
                video_id=task["video_id"], max_predictions=self.cfg["query"]["max_predictions"],
                duration=min(seal["duration"], self.videos[query["video_id"]]["duration"]))
        except (OSError, ValueError) as exc:
            raise RunFailure(f"{exc}", "invalid_output") from exc
        return {**prediction, "query_id": query["query_id"],
                "video_id": query["video_id"], "split": self.split}


def arguments(batch: bool = False):
    parser = argparse.ArgumentParser(description="Run each query in a fresh isolated agent container")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--split", help="Exact dataset.json split name (or dataset.split in config)")
    parser.add_argument("--agent", choices=["codex", "claude_code"])
    parser.add_argument("--model")
    parser.add_argument("--experiment", required=True)
    if not batch:
        parser.add_argument("--query-id", required=True)
    return parser.parse_args()


def configured(args) -> dict:
    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = identifier(args.dataset)
    if args.split is not None:
        cfg["dataset"]["split"] = args.split
    for name in ("agent", "model"):
        if getattr(args, name):
            cfg["query"][name] = getattr(args, name)
    return cfg


def main():
    args = arguments()
    with Experiment(configured(args), args.experiment) as experiment:
        query = next((q for q in experiment.queries if q["query_id"] == args.query_id), None)
        if query is None:
            raise HarnessError(f"Unknown query: {args.query_id}")
        result = experiment.run(query)
        print(f"{args.query_id}: {result['status']}")
        if result["status"] != "success":
            raise SystemExit(1)


if __name__ == "__main__":
    cli(main)
