from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import subprocess
import tempfile
import threading
from pathlib import Path

from harness.common import (HarnessError, atomic_text, cli, file_hash, identifier,
                            ingest_content_hash, now, number, object_hash, parse_json,
                            read_json, write_json, write_jsonl)
from harness.config import load_config
from harness.freeze import remove_tree, tree_hashes, verify_wiki
from harness.vlm import VLMClient


def media_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=True)
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"{command[0]} failed: {exc.stderr[-2000:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"{command[0]} timed out") from exc
    return result.stdout


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
    metadata = parse_json(media_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_type,duration:format=duration", "-of", "json", str(video),
    ]))
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


def sample_times(duration: float, interval: float) -> list[float]:
    if number(duration, "duration") <= 0 or number(interval, "interval") <= 0:
        raise HarnessError("duration and interval must be positive")
    return [round(i * interval, 9) for i in range(math.ceil(duration / interval))
            if i * interval < duration]


def extract_frame(video: Path, timestamp: float, output: Path, cfg: dict) -> None:
    size = cfg["image_max_size"]
    scale = f"scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease"
    media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-ss", f"{timestamp:.9f}", "-i", str(video), "-map", "0:v:0",
                   "-frames:v", "1", "-vf", scale, "-q:v", str(cfg["jpeg_quality"]),
                   "-threads", "1", "-y", str(output)])
    if not output.is_file() or output.stat().st_size == 0:
        raise HarnessError(f"No frame decoded at {timestamp}s")


def render_wiki(duration: float, interval: float, frames: list[dict]) -> str:
    """Query-independent and identity-neutral: the video id is a lookup key
    into public benchmark data, and the agent never needs it to locate moments."""
    lines = ["# Video", "", "## Metadata", "",
             f"- Duration: {duration} seconds", f"- Sampling interval: {interval} seconds",
             f"- Number of frames: {len(frames)}", "", "## Timeline", ""]
    for frame in frames:
        lines += [f"### {frame['timestamp']}s", "", frame["caption"], "",
                  f"Frame: `{frame['frame']}`", ""]
    return "\n".join(lines)


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise HarnessError("Ingest cancelled")


def ingest_video(video: Path, video_id: str, output: Path, cfg: dict, *, captioner=None,
                 cancel_event: threading.Event | None = None) -> dict:
    """No query or GT argument is accepted. Existing completed ingest is never recaptioned."""
    identifier(video_id, "video_id")
    video = video.resolve()
    source_hash = file_hash(video)
    content_hash = ingest_content_hash(cfg)
    if output.exists():
        metadata = read_json(output / "ingest.json")
        stored_hash = ingest_content_hash({"ingest": metadata["ingest_config"]})
        if (metadata["source_sha256"], stored_hash, metadata["video_id"]) != (
                source_hash, content_hash, video_id):
            raise HarnessError("Existing ingest uses different video/settings; choose a new wiki root")
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
        raise HarnessError(f"Ingest already in progress (or stale lock): {lock}") from exc
    staging = None
    try:
        fd.close()
        _check_cancelled(cancel_event)
        if output.exists():
            raise HarnessError("Ingest completed concurrently; rerun to verify existing output")
        duration, video_stream_duration = probe_durations(video)
        _check_cancelled(cancel_event)
        client = captioner if captioner is not None else VLMClient(cfg["ingest"]["vlm"])
        staging = Path(tempfile.mkdtemp(prefix=f".{video_id}.", dir=output.parent))
        (staging / "frames").mkdir()
        frames = []
        sampling_times = sample_times(
            video_stream_duration, cfg["ingest"]["sample_interval_sec"]
        )
        for index, timestamp in enumerate(sampling_times, 1):
            _check_cancelled(cancel_event)
            relative = f"frames/{index:06d}.jpg"
            extract_frame(video, timestamp, staging / relative, cfg["ingest"])
            _check_cancelled(cancel_event)
            caption = client.caption(staging / relative)
            _check_cancelled(cancel_event)
            from harness.common import nonempty
            frames.append({"frame_id": f"f{index:06d}", "timestamp": timestamp,
                           "frame": relative, "caption": nonempty(caption, "caption")})
        write_jsonl(staging / "frames.jsonl", frames)
        atomic_text(staging / "wiki.md", render_wiki(
            duration, cfg["ingest"]["sample_interval_sec"], frames))
        if file_hash(video) != source_hash:
            raise HarnessError("Source video changed during ingest")
        _check_cancelled(cancel_event)
        metadata = {"version": 1, "video_id": video_id, "duration": duration,
                    "video_stream_duration": video_stream_duration,
                    "source_sha256": source_hash, "ingest_config": cfg["ingest"],
                    # Hash of content-determining settings only; transport,
                    # auth, timeouts, and retries remain provenance metadata.
                    "ingest_config_hash": content_hash, "created_at": now(),
                    "ffmpeg_version": media_command(["ffmpeg", "-version"]).splitlines()[0],
                    "content_hashes": tree_hashes(staging)}
        write_json(staging / "ingest.json", metadata)
        staging.rename(output)
        return metadata
    finally:
        if staging is not None and staging.exists():
            remove_tree(staging)
        lock.unlink(missing_ok=True)
        # If we never produced output, remove the (now empty) parent directory
        # so a failed ingest leaves no trace at all. rmdir only succeeds when
        # empty (non-empty = concurrent ingest or other videos: leave it).
        if not output.exists():
            try:
                output.parent.rmdir()
            except OSError:
                pass


def main():
    parser = argparse.ArgumentParser(description="Ingest one video into a query-independent visual wiki")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    metadata = ingest_video(args.video, args.video_id, args.output.resolve(), load_config(args.config))
    print(f"Ingested {metadata['video_id']}: {metadata['duration']}s")


if __name__ == "__main__":
    cli(main)
