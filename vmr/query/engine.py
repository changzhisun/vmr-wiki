from __future__ import annotations
from pathlib import Path
from vmr.core.errors import HarnessError, RunFailure
from vmr.core.hashing import file_hash
from vmr.core.jsonio import read_json, write_json
from vmr.core.time import now
from vmr.runtime.docker import AgentCancelled, trace_path
from vmr.datasets.validation import validate_prediction
from .workspace import query_workspace
from .prediction import clamp_prediction_ends


class QueryEngine:
    """Execute a single isolated query against pinned artifacts."""

    def __init__(
        self,
        *,
        config,
        root,
        runs,
        query_index,
        split,
        videos,
        wiki_set,
        metadata,
        templates,
        runtime,
        aliases,
    ):
        self.config, self.root, self.runs = config, root, runs
        self.query_index, self.split, self.videos = query_index, split, videos
        self.wiki_set, self.metadata = wiki_set, metadata
        self.templates, self.runner, self.aliases = templates, runtime, aliases

    def run(self, query: dict, *, cancel_event=None) -> dict:
        if self.query_index.get(query.get("query_id")) != query:
            raise HarnessError(
                "Query does not belong to this experiment's selected split"
            )
        qid = query["query_id"]
        metadata_path = self.root / "run_metadata" / f"{qid}.json"
        prediction_path = self.root / "predictions" / f"{qid}.json"
        stdout_path = self.root / "logs" / f"{qid}.stdout.log"
        stderr_path = self.root / "logs" / f"{qid}.stderr.log"
        trace_file = trace_path(stdout_path)
        artifact = self.wiki_set.artifact(query["video_id"])
        if artifact.artifact_id() != self.metadata["artifacts"][query["video_id"]]:
            raise HarnessError("Artifact changed after experiment initialization")
        attempts, superseded = 1, []
        if metadata_path.exists():
            previous = read_json(metadata_path)
            if (
                previous.get("dataset") != self.metadata["dataset"]
                or previous.get("split") != self.split
            ):
                raise HarnessError("Run metadata belongs to a different dataset/split")
            if previous["status"] == "running":
                previous.update(
                    status="failed",
                    failure_kind="interrupted",
                    finished_at=now(),
                    error="Previous harness process ended without finalizing this attempt",
                )
                prediction_path.unlink(missing_ok=True)
                write_json(metadata_path, previous)
            if previous["status"] == "success":
                if file_hash(prediction_path) != previous["prediction_hash"]:
                    raise HarnessError(f"Saved prediction was modified: {qid}")
                return previous
            kind = previous.get("failure_kind")
            attempts = previous.get("attempts", 1) + 1
            superseded = [
                *previous.get("superseded_failures", []),
                {
                    "kind": kind,
                    "error": previous.get("error"),
                    "finished_at": previous.get("finished_at"),
                },
            ]
            prediction_path.unlink(missing_ok=True)
            for path in (stdout_path, stderr_path, trace_file):
                path.unlink(missing_ok=True)
        elif prediction_path.exists():
            raise HarnessError(f"Prediction exists without run metadata: {qid}")
        visible_duration = artifact.duration_sec()
        duration = min(visible_duration, self.videos[query["video_id"]]["duration"])
        task = self.aliases.task(query, self.config.max_predictions, duration)
        # The secret lives in experiment.json only; per-run records do not repeat it.
        metadata = {
            key: value for key, value in self.metadata.items() if key != "alias_secret"
        }
        metadata.update(
            {
                "query_id": qid,
                "video_id": query["video_id"],
                "task_query_id": task["query_id"],
                "task_video_id": task["video_id"],
                "artifact_id": artifact.artifact_id(),
                "artifact_manifest_hash": artifact.manifest_hash(),
                "trace_path": str(trace_file.relative_to(self.root)),
                "effective_duration": duration,
                "output_adjustments": [],
                "status": "running",
                "failure_kind": None,
                "exit_code": None,
                "timed_out": False,
                "attempts": attempts,
                "superseded_failures": superseded,
                "started_at": now(),
                "finished_at": None,
            }
        )
        write_json(metadata_path, metadata)
        try:
            prediction = None
            with query_workspace(
                query,
                task,
                artifact,
                self.templates,
                self.runs,
                text_only=self.config.input_mode == "text",
            ) as workspace:
                cancel = (
                    {"cancel_event": cancel_event} if cancel_event is not None else {}
                )
                result = self.runner.run(
                    workspace,
                    self.templates["query_prompt.md"],
                    stdout_path,
                    stderr_path,
                    **cancel,
                )
                metadata.update(exit_code=result.exit_code, timed_out=result.timed_out)
                if result.timed_out:
                    raise RunFailure("Agent timed out", "timeout")
                if result.exit_code != 0:
                    raise RunFailure(
                        f"Agent exited with code {result.exit_code}", "agent_error"
                    )
                prediction, adjustments = self.read_prediction(
                    workspace / "output", query, task, duration
                )
            metadata["output_adjustments"] = adjustments
            # Publish only after immutable-input checks and workspace cleanup succeeded.
            write_json(prediction_path, prediction)
            metadata.update(
                status="success", prediction_hash=file_hash(prediction_path)
            )
        except RunFailure as exc:
            metadata.update(status="failed", failure_kind=exc.kind, error=f"{exc}")
        except AgentCancelled as exc:
            metadata.update(status="failed", failure_kind="interrupted", error=f"{exc}")
            raise
        except Exception as exc:
            # Not a score. Record the cause, then let the caller stop the batch
            # instead of charging a harness or infrastructure fault to the agent.
            metadata.update(
                status="failed",
                failure_kind="harness_error",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        except BaseException:
            metadata.update(
                status="failed", failure_kind="interrupted", error="Harness interrupted"
            )
            raise
        finally:
            metadata["finished_at"] = now()
            write_json(metadata_path, metadata)
        return metadata

    def read_prediction(
        self, output: Path, query: dict, task: dict, duration: float
    ) -> tuple[dict, list[dict]]:
        """Read the agent's only output. Every rejection here is the agent's own.

        The agent echoes the aliases it was given, so they are validated as
        given and then replaced by the real identifiers on the way to disk;
        aggregation and evaluation never see an alias.
        """
        prediction_file = output / "prediction.json"
        try:
            if sorted(p.name for p in output.iterdir()) != ["prediction.json"]:
                raise RunFailure(
                    "Expected exactly output/prediction.json", "invalid_output"
                )
            if prediction_file.is_symlink() or not prediction_file.is_file():
                raise RunFailure(
                    "Prediction must be a regular file, not a symlink", "invalid_output"
                )
            if prediction_file.stat().st_size > 1024 * 1024:
                raise RunFailure("Prediction exceeds 1 MiB", "invalid_output")
        except RunFailure:
            raise
        except OSError as exc:
            raise RunFailure(f"{exc}", "invalid_output") from exc
        try:
            prediction = validate_prediction(
                read_json(prediction_file),
                query_id=task["query_id"],
                split=task["split"],
                video_id=task["video_id"],
                max_predictions=self.config.max_predictions,
            )
        except (HarnessError, OSError) as exc:
            raise RunFailure(f"{exc}", "invalid_output") from exc
        adjustments = clamp_prediction_ends(prediction, duration=duration)
        try:
            prediction = validate_prediction(
                prediction,
                query_id=task["query_id"],
                split=task["split"],
                video_id=task["video_id"],
                max_predictions=self.config.max_predictions,
                duration=duration,
            )
        except HarnessError as exc:
            raise RunFailure(f"{exc}", "invalid_output") from exc
        return (
            {
                **prediction,
                "query_id": query["query_id"],
                "video_id": query["video_id"],
                "split": self.split,
            },
            adjustments,
        )
