from __future__ import annotations
from pathlib import Path
from vmr.core.errors import HarnessError
from vmr.core.validation import number
from vmr.core.jsonio import parse_json
from .ffmpeg import media_command


def _optional_duration(raw, field: str) -> float | None:
    try:
        duration = number(float(raw), field, 0)
    except (TypeError, ValueError, HarnessError):
        return None
    return duration if duration > 0 else None


def probe_durations(video: Path) -> tuple[float, float]:
    """Return ``(container_duration, video_stream_duration)``.

    The container duration remains the public duration in the generated Wiki,
    while the selected video stream duration bounds frame sampling. Either
    value falls back to the other when FFprobe cannot provide it.
    """
    metadata = parse_json(
        media_command(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type,duration:format=duration",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    if not metadata.get("streams"):
        raise HarnessError("Input contains no video stream")
    container = _optional_duration(
        metadata.get("format", {}).get("duration"), "container duration"
    )
    stream = _optional_duration(
        metadata["streams"][0].get("duration"), "video stream duration"
    )
    if container is None and stream is None:
        raise HarnessError("Unable to determine video duration")
    if container is None:
        container = stream
    if stream is None:
        stream = container
    assert container is not None and stream is not None
    return container, stream


def probe_duration(video: Path) -> float:
    """Return the container duration for display and metadata compatibility."""
    return probe_durations(video)[0]
