"""Private build resources shared by the established compiler backends."""

from dataclasses import replace
from pathlib import Path
import tempfile
from vmr.core.errors import HarnessError
from vmr.core.hashing import object_hash
from vmr.media.probe import probe_durations
from vmr.media.ffmpeg import media_command
from .checkpoint import CompileCheckpoint
from vmr.vlm.client import VLMClient


def prepare_backend_context(context, compiler):
    def check():
        if context.cancel_event is not None and context.cancel_event.is_set():
            raise HarnessError("Compile cancelled")

    check()
    duration, stream_duration = probe_durations(context.source.path)
    ffmpeg = media_command(["ffmpeg", "-version"]).splitlines()[0]
    content_hash = object_hash(
        compiler.content_identity(context.config, context.source)
    )
    cfg = dict(
        build_record="build.json",
        ingest=dict(context.config.settings),
        paths=dict(
            templates=str(context.templates),
            runs=str(context.output.parent / "runs"),
            logs=str(context.output.parent / "logs"),
        ),
    )
    checkpoint = CompileCheckpoint(
        context.output,
        dict(
            version=4,
            output=str(context.output.resolve()),
            video_id="video",
            source_sha256=context.source.sha256,
            content_hash=content_hash,
            ffmpeg_version=ffmpeg,
            duration=duration,
            video_stream_duration=stream_duration,
        ),
    )
    context.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".build-", dir=context.output.parent))
    (staging / "frames").mkdir()
    client = context.captioner
    if client is None and compiler.requires_vlm:
        client = VLMClient(
            cfg["ingest"]["vlm"],
            timestamp_mode=cfg["ingest"].get(
                "dense_timestamp_mode", "absolute_seconds"
            ),
            cancel_event=context.cancel_event,
        )
    return replace(
        context,
        captioner=client,
        build=dict(
            video_id="video",
            staging=staging,
            checkpoint=checkpoint,
            cfg=cfg,
            duration=duration,
            video_stream_duration=stream_duration,
            content_hash=content_hash,
            ffmpeg_version=ffmpeg,
            check=check,
        ),
    )
