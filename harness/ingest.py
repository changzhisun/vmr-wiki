from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import math
import re
import subprocess
import tempfile
import threading
from pathlib import Path

from harness.common import (HarnessError, atomic_text, cli, file_hash, identifier,
                            ingest_content_hash, nonempty, now, number, object_hash,
                            parse_json, positive_int, read_json, write_json, write_jsonl)
from harness.config import load_config
from harness.freeze import remove_tree, tree_hashes, verify_wiki
from harness.vlm import VLMClient

logger = logging.getLogger(__name__)


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


def caption_windows(frames: list[dict], window_frames: int, stride_frames: int) -> list[list[dict]]:
    """Group sampled frames into chronological, possibly overlapping windows.

    The final window may contain fewer frames than ``window_frames``. A stride
    larger than the window intentionally leaves unsampled gaps.
    """
    window = positive_int(window_frames, "caption_window_frames")
    stride = positive_int(stride_frames, "caption_stride_frames")
    return [frames[start:start + window] for start in range(0, len(frames), stride)]


# Whole-response Markdown fence only. Prose around a fence, or a fence in the
# middle of other text, is still invalid: we unwrap the wrapper, not the JSON.
_MARKDOWN_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)[ \t]*\r?\n?```\Z",
    re.IGNORECASE | re.DOTALL,
)


def unwrap_markdown_json_fence(text: str) -> str:
    """Strip a surrounding ```json fence. Do not extract JSON from other prose."""
    stripped = text.strip()
    match = _MARKDOWN_JSON_FENCE.fullmatch(stripped)
    return match.group("body").strip() if match else stripped


_TIMESTAMP_EPS = 1e-9


def _on_timeline(value: float, timestamps: list[float]) -> float | None:
    for stamp in timestamps:
        if abs(value - stamp) <= _TIMESTAMP_EPS:
            return stamp
    return None


def _as_window_index(value: float, count: int) -> int | None:
    nearest = round(value)
    if abs(value - nearest) > _TIMESTAMP_EPS:
        return None
    index = int(nearest)
    if count <= 0:
        return None
    if 0 <= index < count:
        return index
    # Exclusive 0-based end, or 1-based last frame (n frames numbered 1..n).
    if index == count:
        return count - 1
    return None


def _exclusive_end(value: float, timestamps: list[float]) -> float | None:
    if len(timestamps) < 2:
        return None
    step = timestamps[-1] - timestamps[-2]
    if step <= 0:
        return None
    exclusive = timestamps[-1] + step
    if abs(value - exclusive) <= _TIMESTAMP_EPS:
        return timestamps[-1]
    return None


def resolve_window_timestamp(value: float, timestamps: list[float]) -> float | None:
    """Map a model timestamp onto this window's sampled times.

    Absolute times already on the timeline win. Otherwise a whole number in
    ``[0, len(timestamps)]`` is a frame index: ``0..n-1`` as usual, and ``n`` as
    the exclusive end / 1-based last frame. A time one step past the last
    sample is the same exclusive end. Invented precision such as ``1.5`` is
    still rejected.
    """
    matched = _on_timeline(value, timestamps)
    if matched is not None:
        return matched
    index = _as_window_index(value, len(timestamps))
    if index is not None:
        return timestamps[index]
    return _exclusive_end(value, timestamps)


def _dense_events_payload(payload: object) -> list:
    """Take the events array; ignore extra object keys and a bare list."""
    if isinstance(payload, str):
        payload = parse_json(payload)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("events", "Events", "event"):
            if key in payload:
                events = payload[key]
                if not isinstance(events, list):
                    raise HarnessError("Dense caption events must be a list")
                return events
        keys = ", ".join(sorted(map(str, payload)))
        raise HarnessError(f"Dense caption must contain an events array (keys=[{keys}])")
    raise HarnessError(
        f"Dense caption must be a JSON object with events (got {type(payload).__name__})"
    )


def parse_dense_events(text: str, timestamps: list[float]) -> list[dict]:
    """Validate and normalize one dense-caption response.

    Event boundaries must select timestamps from the window verbatim. This
    prevents the captioner from inventing temporal precision unavailable in
    the sampled frames. A surrounding Markdown code fence is discarded.
    Extra object keys and a top-level events list are accepted. Whole-number
    frame indices, including an exclusive end of ``n``, map onto the window
    timeline.
    """
    payload = parse_json(unwrap_markdown_json_fence(text))
    events = _dense_events_payload(payload)
    window = ", ".join(str(stamp) for stamp in timestamps)
    normalized = []
    previous_start: float | None = None
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict) or set(event) != {"start", "end", "caption"}:
            raise HarnessError(
                f"Dense caption event {index} must contain only start, end, and caption"
            )
        start = resolve_window_timestamp(
            number(event["start"], f"dense event {index} start"), timestamps
        )
        end = resolve_window_timestamp(
            number(event["end"], f"dense event {index} end"), timestamps
        )
        if start is None or end is None:
            raise HarnessError(
                f"Dense caption event {index} uses a timestamp outside its window "
                f"(start={event['start']!r}, end={event['end']!r}; window=[{window}])"
            )
        if start > end:
            raise HarnessError(f"Dense caption event {index} start must be <= end")
        if previous_start is not None and start < previous_start:
            raise HarnessError("Dense caption events must be in chronological order")
        previous_start = start
        normalized.append({
            "start": start,
            "end": end,
            "caption": nonempty(event["caption"], f"dense event {index} caption").strip(),
        })
    return normalized


def dense_events(client, images: list[Path], timestamps: list[float], max_repairs: int,
                 *, cancel_event: threading.Event | None = None) -> list[dict]:
    """Caption one window, re-asking a bounded number of times on a violation.

    A rejected dense answer is usually the captioner ignoring the window
    rather than a broken endpoint, so the rejection is fed back instead of
    failing the whole video on the first bad window.
    """
    correction = None
    for remaining in range(max_repairs, -1, -1):
        _check_cancelled(cancel_event)
        # Only a repair passes correction, so the first call of a run stays
        # identical to a captioner that does not accept one.
        repair = {} if correction is None else {"correction": correction}
        response = client.caption(images, timestamps=timestamps, **repair)
        try:
            return parse_dense_events(response, timestamps)
        except HarnessError as exc:
            if not remaining:
                raise
            correction = str(exc)
            logger.warning("Re-asking the captioner for window %s: %s",
                           f"[{timestamps[0]}, {timestamps[-1]}]", exc)
    raise AssertionError("unreachable")  # pragma: no cover


def extract_frame(video: Path, timestamp: float, output: Path, cfg: dict) -> None:
    size = cfg["image_max_size"]
    scale = f"scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease"
    media_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                   "-ss", f"{timestamp:.9f}", "-i", str(video), "-map", "0:v:0",
                   "-frames:v", "1", "-vf", scale, "-q:v", str(cfg["jpeg_quality"]),
                   "-threads", "1", "-y", str(output)])
    if not output.is_file() or output.stat().st_size == 0:
        raise HarnessError(f"No frame decoded at {timestamp}s")


def render_wiki(duration: float, interval: float, entries: list[dict], *,
                sampled_frame_count: int | None = None, window_frames: int = 1,
                stride_frames: int = 1, caption_mode: str = "simple") -> str:
    """Query-independent and identity-neutral: the video id is a lookup key
    into public benchmark data, and the agent never needs it to locate moments."""
    lines = ["# Video", "", "## Metadata", "",
             f"- Duration: {duration} seconds", f"- Sampling interval: {interval} seconds"]
    if caption_mode == "dense":
        lines += [
            "- Caption mode: dense",
            f"- Number of sampled frames: {sampled_frame_count}",
            f"- Number of caption windows: {len(entries)}",
            f"- Number of dense events: {sum(len(entry['events']) for entry in entries)}",
            f"- Caption window: {window_frames} sampled frames",
            f"- Caption stride: {stride_frames} sampled frames",
        ]
    elif window_frames == 1:
        lines += [f"- Number of frames: {len(entries)}"]
    else:
        lines += [
            f"- Number of sampled frames: {sampled_frame_count}",
            f"- Number of caption windows: {len(entries)}",
            f"- Caption window: {window_frames} sampled frames",
            f"- Caption stride: {stride_frames} sampled frames",
        ]
    lines += ["", "## Timeline", ""]
    for entry in entries:
        if caption_mode == "dense":
            start, end = entry["start_timestamp"], entry["end_timestamp"]
            window_range = f"{start}s" if start == end else f"{start}s–{end}s"
            lines += [f"### Window {window_range}", ""]
            if entry["events"]:
                for event in entry["events"]:
                    event_start, event_end = event["start"], event["end"]
                    event_range = (
                        f"{event_start}s" if event_start == event_end
                        else f"{event_start}s–{event_end}s"
                    )
                    lines += [f"#### Event {event_range}", "", event["caption"], ""]
            else:
                lines += ["No meaningful visual event.", ""]
            lines.append("Frames:")
            lines += [
                f"- {frame['timestamp']}s: `{frame['frame']}`" for frame in entry["frames"]
            ]
            lines.append("")
            continue
        if "frames" not in entry:
            lines += [f"### {entry['timestamp']}s", "", entry["caption"], "",
                      f"Frame: `{entry['frame']}`", ""]
            continue
        start, end = entry["start_timestamp"], entry["end_timestamp"]
        heading = f"### {start}s" if start == end else f"### {start}s–{end}s"
        lines += [heading, "", entry["caption"], "", "Frames:"]
        lines += [f"- {frame['timestamp']}s: `{frame['frame']}`" for frame in entry["frames"]]
        lines.append("")
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
        sampling_times = sample_times(
            video_stream_duration, cfg["ingest"]["sample_interval_sec"]
        )
        sampled_frames = [
            {"frame_id": f"f{index:06d}", "timestamp": timestamp,
             "frame": f"frames/{index:06d}.jpg"}
            for index, timestamp in enumerate(sampling_times, 1)
        ]
        window_frames = cfg["ingest"]["caption_window_frames"]
        stride_frames = cfg["ingest"]["caption_stride_frames"]
        caption_mode = cfg["ingest"]["caption_mode"]
        windows = caption_windows(sampled_frames, window_frames, stride_frames)
        used_ids = {frame["frame_id"] for window in windows for frame in window}
        for frame in sampled_frames:
            if frame["frame_id"] not in used_ids:
                continue
            _check_cancelled(cancel_event)
            extract_frame(video, frame["timestamp"], staging / frame["frame"], cfg["ingest"])

        entries = []
        for window_index, window in enumerate(windows, 1):
            _check_cancelled(cancel_event)
            images = [staging / frame["frame"] for frame in window]
            if caption_mode == "dense":
                timestamps = [frame["timestamp"] for frame in window]
                events = dense_events(
                    client, images, timestamps, cfg["ingest"]["caption_max_repairs"],
                    cancel_event=cancel_event,
                )
                _check_cancelled(cancel_event)
                entries.append({
                    "window_id": f"w{window_index:06d}",
                    "start_timestamp": window[0]["timestamp"],
                    "end_timestamp": window[-1]["timestamp"],
                    "frames": window,
                    "events": events,
                })
                continue
            caption_input = images[0] if window_frames == 1 else images
            caption = nonempty(client.caption(caption_input), "caption")
            _check_cancelled(cancel_event)
            if window_frames == 1:
                entries.append({**window[0], "caption": caption})
            else:
                entries.append({
                    "window_id": f"w{window_index:06d}",
                    "start_timestamp": window[0]["timestamp"],
                    "end_timestamp": window[-1]["timestamp"],
                    "frames": window,
                    "caption": caption,
                })
        write_jsonl(staging / "frames.jsonl", entries)
        atomic_text(staging / "wiki.md", render_wiki(
            duration, cfg["ingest"]["sample_interval_sec"], entries,
            sampled_frame_count=len(used_ids), window_frames=window_frames,
            stride_frames=stride_frames, caption_mode=caption_mode))
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
