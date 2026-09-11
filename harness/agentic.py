"""Agent-compiled wiki: a coding agent decides how to sample and describe.

The container receives the video and nothing else. It has no query, no ground
truth and no dataset identifier, so the published wiki stays query-independent
and identity-neutral exactly like a VLM-generated one.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

from harness.agentic_validate import validate_agent_wiki
from harness.common import HarnessError, file_hash, now, parse_json, write_json
from harness.freeze import remove_tree, tree_hashes
from harness.ingest import media_command

LOG = logging.getLogger(__name__)

PUBLISHED = ("wiki.md", "frames.jsonl")


def probe_media(video: Path) -> dict:
    """Media facts the agent would otherwise have to rediscover."""
    metadata = parse_json(media_command([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,codec_name,width,height,avg_frame_rate", "-of", "json", str(video),
    ]))
    streams = metadata.get("streams") or []
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        raise HarnessError("Input contains no video stream")
    first = video_streams[0]
    fps = None
    rate = first.get("avg_frame_rate") or ""
    if "/" in rate:
        numerator, _, denominator = rate.partition("/")
        try:
            fps = round(float(numerator) / float(denominator), 6) if float(denominator) else None
        except (TypeError, ValueError, ZeroDivisionError):
            fps = None
    return {"fps": fps, "width": first.get("width"), "height": first.get("height"),
            "codec": first.get("codec_name"),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def build_task(cfg: dict, duration: float, video_stream_duration: float, media: dict) -> dict:
    """The only description of the task the agent ever sees.

    It deliberately carries no video id, split or query: the wiki must serve
    every future retrieval query, and a public benchmark identifier would be an
    exact lookup key into data the model may have memorized.
    """
    ingest = cfg["ingest"]
    options = ingest["agentic"]
    return {
        "version": "1.0",
        "task": "video_to_wiki",
        "input": {"video": "/input/video.mp4"},
        "output": {"directory": "/workspace/output"},
        "scratch": {"directory": "/scratch", "budget_gb": options["scratch_size_gb"]},
        "duration": duration,
        "video_stream_duration": video_stream_duration,
        "fps": media["fps"],
        "width": media["width"],
        "height": media["height"],
        "has_audio": media["has_audio"],
        "image_max_size": ingest["image_max_size"],
        "jpeg_qscale": ingest["jpeg_quality"],
        "max_frames": options["max_frames"],
        "frame_extraction": options["frame_extraction"],
        "wiki": options["wiki"],
    }


def _job_workspace(job: Path, instructions: str, task: dict) -> None:
    (job / "AGENTS.md").write_text(instructions, encoding="utf-8")
    write_json(job / "task.json", task)
    for name in ("AGENTS.md", "task.json"):
        (job / name).chmod(0o444)
    (job / "output").mkdir(mode=0o777)
    (job / "output").chmod(0o777)  # container runs as an unprivileged UID
    job.chmod(0o555)


def _publish_files(output: Path, staging: Path, rows: list[dict]) -> None:
    for name in PUBLISHED:
        shutil.copyfile(output / name, staging / name)
    for row in rows:
        shutil.copyfile(output / row["frame"], staging / row["frame"])


def publish_agentic(video, video_id, output, staging, checkpoint, client, cfg,
                    duration, video_stream_duration, source_hash, content_hash,
                    ffmpeg_version, check, *, runner=None):
    if runner is None:
        from agents.runner import AgentIngestRunner
        runner = AgentIngestRunner(cfg)
    options = cfg["ingest"]["agentic"]
    templates = Path(cfg["paths"]["templates"])
    instructions = (templates / "wiki_agents.md").read_text(encoding="utf-8")
    prompt = (templates / "wiki_prompt.md").read_text(encoding="utf-8")

    check()
    media = probe_media(video)
    task = build_task(cfg, duration, video_stream_duration, media)

    runs = Path(cfg["paths"]["runs"])
    runs.mkdir(parents=True, exist_ok=True)
    logs = output.parent.parent / ".ingest-logs" / video_id
    logs.mkdir(parents=True, exist_ok=True)

    # The container can read its own bind mount source through /proc, so even
    # the host directory names avoid the video identity.
    job = Path(tempfile.mkdtemp(prefix="ingest-", dir=runs)).resolve()
    scratch = Path(tempfile.mkdtemp(prefix="scratch-", dir=runs)).resolve()
    try:
        _job_workspace(job, instructions, task)
        scratch.chmod(0o777)
        immutable = {name: file_hash(job / name) for name in ("AGENTS.md", "task.json")}

        check()
        started = time.monotonic()
        result = runner.run(job, prompt, logs / "agent.stdout.log", logs / "agent.stderr.log",
                            video=video, scratch=scratch)
        elapsed = time.monotonic() - started
        check()
        if result.timed_out:
            raise HarnessError(
                f"Agent timed out after {options['timeout_sec']}s compiling the wiki; "
                f"logs: {logs}")
        if result.exit_code != 0:
            raise HarnessError(
                f"Agent exited with code {result.exit_code} without a published wiki; logs: {logs}")
        if any(file_hash(job / name) != digest for name, digest in immutable.items()):
            raise HarnessError("Agent changed its own immutable instructions or task")

        # Identity leakage is checked against the exact strings that would make
        # the wiki a lookup key into the public benchmark.
        forbidden = tuple({video_id, video.name, video.stem})
        rows = validate_agent_wiki(
            job / "output", duration=video_stream_duration, max_frames=options["max_frames"],
            max_wiki_bytes=options["max_wiki_bytes"], max_frame_bytes=options["max_frame_bytes"],
            forbidden_tokens=forbidden)

        validated = {name: file_hash(job / "output" / name) for name in PUBLISHED}
        validated.update({row["frame"]: file_hash(job / "output" / row["frame"]) for row in rows})
        _publish_files(job / "output", staging, rows)
        if tree_hashes(staging) != validated:
            raise HarnessError("Agent output changed while being published")

        if file_hash(video) != source_hash:
            raise HarnessError("Source video changed during ingest")
        check()
        metadata = {
            "version": 1, "video_id": video_id, "duration": duration,
            "video_stream_duration": video_stream_duration,
            "source_sha256": source_hash, "ingest_config": cfg["ingest"],
            "ingest_config_hash": content_hash, "created_at": now(),
            "ffmpeg_version": ffmpeg_version,
            "agentic_version": cfg["ingest"]["agentic_version"],
            # Provenance, not content: the image, endpoint and egress policy can
            # change without changing what the agent was asked to produce.
            "agent_provenance": {**runner.provenance, "agent": options["agent"],
                                 "model": options["model"]},
            "media": media,
            "telemetry": {"agent_sec": elapsed, "exit_code": result.exit_code,
                          "frame_count": len(rows),
                          "wiki_md_bytes": (staging / "wiki.md").stat().st_size,
                          "reasons": {reason: sum(row["reason"] == reason for row in rows)
                                      for reason in sorted({row["reason"] for row in rows})}},
            "content_hashes": tree_hashes(staging),
        }
        write_json(staging / "ingest.json", metadata)
        staging.rename(output)
        try:
            remove_tree(checkpoint.root)
        except OSError:
            LOG.warning("Published Wiki; could not remove checkpoint: %s", checkpoint.root)
        return metadata
    finally:
        remove_tree(job)
        remove_tree(scratch)
