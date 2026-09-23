from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.jsonio import atomic_text, parse_json, read_json, write_json, write_jsonl
from vmr.core.validation import cli, identifier, nonempty, number, positive_int
from vmr.core.hashing import file_hash, object_hash
from vmr.compat.identity import compile_content_hash as ingest_content_hash
from vmr.core.time import now

from harness.checkpoint import IngestCheckpoint
from vmr.artifact.integrity import remove_tree, tree_hashes
from harness.freeze import verify_wiki
from vmr.vlm.client import VLMClient

logger = logging.getLogger(__name__)


from vmr.media.ffmpeg import media_command


from vmr.media.probe import _optional_duration


from vmr.media.probe import probe_durations


from vmr.media.probe import probe_duration


from vmr.media.sampling import sample_times


from vmr.compiler.methods.window import (
    caption_windows,
    dense_target_timestamps,
    unwrap_markdown_json_fence,
    resolve_window_timestamp,
    parse_dense_events,
    dense_events,
    extract_frame,
    compact_dense_events,
    render_wiki,
)


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise HarnessError("Ingest cancelled")


def ingest_video(
    video: Path,
    video_id: str,
    output: Path,
    cfg: dict,
    *,
    captioner=None,
    runner=None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """No query or GT argument is accepted. Existing completed ingest is never recaptioned."""
    identifier(video_id, "video_id")
    video = video.resolve()
    source_hash = file_hash(video)
    content_hash = ingest_content_hash(cfg)
    if output.exists():
        metadata = read_json(output / "ingest.json")
        stored_hash = ingest_content_hash({"ingest": metadata["ingest_config"]})
        if (metadata["source_sha256"], stored_hash, metadata["video_id"]) != (
            source_hash,
            content_hash,
            video_id,
        ):
            raise HarnessError(
                "Existing ingest uses different video/settings; choose a new wiki root"
            )
        if (output / "frozen.json").exists():
            verify_wiki(output)
        elif tree_hashes(output, ("ingest.json",)) != metadata["content_hashes"]:
            raise HarnessError("Existing ingest content has changed")
        return metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the video before API calls: concurrent ingest cannot recaption it.
    lock = output.parent / f".{output.name}.ingest.lock"
    try:
        fd = lock.open("x")
    except FileExistsError as exc:
        raise HarnessError(
            f"Ingest already in progress (or stale lock): {lock}"
        ) from exc
    staging = None
    try:
        fd.close()
        _check_cancelled(cancel_event)
        if output.exists():
            raise HarnessError(
                "Ingest completed concurrently; rerun to verify existing output"
            )
        duration, video_stream_duration = probe_durations(video)
        ffmpeg_version = media_command(["ffmpeg", "-version"]).splitlines()[0]
        checkpoint = IngestCheckpoint(
            output,
            {
                "version": 4,
                "output": str(output.resolve()),
                "video_id": video_id,
                "source_sha256": source_hash,
                "content_hash": content_hash,
                "ffmpeg_version": ffmpeg_version,
                "duration": duration,
                "video_stream_duration": video_stream_duration,
            },
        )
        _check_cancelled(cancel_event)
        staging = Path(tempfile.mkdtemp(prefix=f".{video_id}.", dir=output.parent))
        (staging / "frames").mkdir()
        from vmr.compiler.registry import get_compiler
        from vmr.compiler.protocol import CompileContext, VideoSource

        compiler = get_compiler(cfg["ingest"]["caption_mode"])
        client = captioner
        if client is None and compiler.requires_vlm:
            client = VLMClient(
                cfg["ingest"]["vlm"],
                timestamp_mode=cfg["ingest"].get(
                    "dense_timestamp_mode", "absolute_seconds"
                ),
                cancel_event=cancel_event,
            )
        context = CompileContext(
            VideoSource(video, source_hash),
            compiler.parse_config(cfg["ingest"]),
            output,
            Path(cfg.get("paths", {}).get("templates", "templates")),
            client,
            runner,
            cancel_event,
            build=dict(
                video_id=video_id,
                staging=staging,
                checkpoint=checkpoint,
                cfg=cfg,
                duration=duration,
                video_stream_duration=video_stream_duration,
                content_hash=content_hash,
                ffmpeg_version=ffmpeg_version,
                check=lambda: _check_cancelled(cancel_event),
            ),
        )
        return dict(compiler.compile(context).provenance)

    finally:
        if staging is not None and staging.exists():
            remove_tree(staging)
        lock.unlink(missing_ok=True)
        # If we never produced output, remove the (now empty) parent directory
        # while keeping the private checkpoint for resume. rmdir only succeeds when
        # empty (non-empty = concurrent ingest or other videos: leave it).
        if not output.exists():
            try:
                output.parent.rmdir()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Ingest one video into a query-independent visual wiki"
    )
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    from harness.config import load_config

    metadata = ingest_video(
        args.video, args.video_id, args.output.resolve(), load_config(args.config)
    )
    print(f"Ingested {metadata['video_id']}: {metadata['duration']}s")


if __name__ == "__main__":
    cli(main)
