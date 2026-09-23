import logging
import shutil
import threading
import time
from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.jsonio import atomic_text, parse_json, write_json, write_jsonl
from vmr.core.hashing import file_hash
from vmr.core.validation import nonempty, number, positive_int
from vmr.core.time import now

from vmr.artifact.integrity import remove_tree, tree_hashes

logger = logging.getLogger(__name__)


from vmr.media.sampling import sample_times


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise HarnessError("Compile cancelled")


def caption_windows(
    frames: list[dict], window_frames: int, stride_frames: int
) -> list[list[dict]]:
    """Group sampled frames into chronological, possibly overlapping windows.

    Emit one short window only when the whole video is shorter than the
    configured size. Otherwise use full windows and, when necessary, add one
    full window anchored at the end so no redundant shrinking tail is sent to
    the captioner. A stride larger than the window intentionally leaves gaps.
    """
    window = positive_int(window_frames, "caption_window_frames")
    stride = positive_int(stride_frames, "caption_stride_frames")
    if not frames:
        return []
    if len(frames) <= window:
        return [frames]
    last_start = len(frames) - window
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return [frames[start : start + window] for start in starts]


def dense_target_timestamps(windows: list[list[dict]]) -> list[list[float]]:
    """Assign every dense window a central, non-repeated target region.

    Adjacent regions share only their boundary sample so an action can span
    the transition. The first and last windows also own the video boundaries,
    where symmetric context cannot exist.
    """
    if not windows:
        return []
    if all(len(window) == 1 for window in windows):
        return [[window[0]["timestamp"]] for window in windows]
    centers = [(len(window) - 1) // 2 for window in windows]
    targets = []
    for index, (window, center) in enumerate(zip(windows, centers)):
        start = 0 if index == 0 else center
        if index + 1 == len(windows):
            end = len(window) - 1
        else:
            next_timestamp = windows[index + 1][centers[index + 1]]["timestamp"]
            matches = [
                i
                for i, frame in enumerate(window)
                if frame["timestamp"] == next_timestamp
            ]
            if not matches:
                raise HarnessError(
                    "Dense stride is too large for centered target regions; "
                    "use a smaller stride or larger caption window"
                )
            end = matches[0]
        targets.append([frame["timestamp"] for frame in window[start : end + 1]])
    return targets


from vmr.core.structured import unwrap_markdown_json_fence

_TIMESTAMP_EPS = 1e-9


def _on_timeline(value: float, timestamps: list[float]) -> float | None:
    for stamp in timestamps:
        if abs(value - stamp) <= _TIMESTAMP_EPS:
            return stamp
    return None


def resolve_window_timestamp(
    value: float, timestamps: list[float], mode: str = "absolute_seconds"
) -> float | None:
    """Use one explicit coordinate system for the entire response, never guess."""
    if mode == "absolute_seconds":
        return _on_timeline(value, timestamps)
    if mode != "frame_index":
        raise HarnessError("Unknown dense timestamp mode")
    if value.is_integer() and 0 <= value < len(timestamps):
        return timestamps[int(value)]
    return None


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
        raise HarnessError(
            f"Dense caption must contain an events array (keys=[{keys}])"
        )
    raise HarnessError(
        f"Dense caption must be a JSON object with events (got {type(payload).__name__})"
    )


def parse_dense_events(
    text: str, timestamps: list[float], timestamp_mode: str = "absolute_seconds"
) -> list[dict]:
    """Validate and normalize one dense-caption response.

    Event boundaries must select timestamps from the window verbatim. This
    prevents the captioner from inventing temporal precision unavailable in
    the sampled frames. A surrounding Markdown code fence is discarded.
    Extra top-level keys and a bare events list are accepted. Frame-index mode
    uses zero-based inclusive indices for both boundaries. No exclusive ends
    or mixed coordinate systems are inferred. Centered targets are a prompt
    focus, not a parser restriction: an event may span context timestamps.
    """
    payload = parse_json(unwrap_markdown_json_fence(text))
    events = _dense_events_payload(payload)
    allowed_values = (
        list(range(len(timestamps))) if timestamp_mode == "frame_index" else timestamps
    )
    window = ", ".join(str(stamp) for stamp in allowed_values)
    normalized = []
    previous_start: float | None = None
    for index, event in enumerate(events, 1):
        if not isinstance(event, dict) or set(event) != {
            "start",
            "end",
            "kind",
            "caption",
        }:
            raise HarnessError(
                f"Dense caption event {index} must contain only start, end, kind, and caption"
            )
        start = resolve_window_timestamp(
            number(event["start"], f"dense event {index} start"),
            timestamps,
            timestamp_mode,
        )
        end = resolve_window_timestamp(
            number(event["end"], f"dense event {index} end"), timestamps, timestamp_mode
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
        raw_kind = event["kind"]
        kind = raw_kind.strip().lower() if isinstance(raw_kind, str) else None
        if kind not in ("state", "action", "transition"):
            raise HarnessError(
                f"Dense caption event {index} kind must be state, action, or transition"
            )
        normalized.append(
            {
                "start": start,
                "end": end,
                "kind": kind,
                "caption": nonempty(
                    event["caption"], f"dense event {index} caption"
                ).strip(),
            }
        )
    return normalized


def dense_events(
    client,
    images: list[Path],
    timestamps: list[float],
    max_repairs: int,
    *,
    cancel_event: threading.Event | None = None,
    timestamp_mode: str = "absolute_seconds",
    target_timestamps: list[float] | None = None,
    record=None,
) -> list[dict]:
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
        # The model sees only values from the selected coordinate system.
        timeline = (
            list(range(len(timestamps)))
            if timestamp_mode == "frame_index"
            else timestamps
        )
        targets = target_timestamps if target_timestamps is not None else timestamps
        target_timeline = (
            [timestamps.index(stamp) for stamp in targets]
            if timestamp_mode == "frame_index"
            else targets
        )
        started = time.monotonic()
        attempt = {
            "correction": correction,
            "timestamp_mode": timestamp_mode,
            "target_timestamps": targets,
        }
        try:
            response = client.caption(
                images, timestamps=timeline, target_timestamps=target_timeline, **repair
            )
            attempt["raw_response"] = response
            events = parse_dense_events(response, timestamps, timestamp_mode)
            attempt.update(status="success", normalized_events=events)
            return events
        except BaseException as exc:
            attempt.update(
                status="failed",
                error=str(exc) if isinstance(exc, HarnessError) else type(exc).__name__,
            )
            # Interrupts and unexpected client errors must leave a complete
            # audit record so a later resumption can aggregate all attempts.
            if not isinstance(exc, HarnessError):
                raise
            # Transport failures and truncated responses follow the VLM retry
            # policy, not the schema-repair budget.
            if "raw_response" not in attempt:
                raise
            if not remaining:
                raise
            correction = str(exc)
            logger.warning(
                "Re-asking the captioner for window %s: %s",
                f"[{timestamps[0]}, {timestamps[-1]}]",
                exc,
            )
        finally:
            attempt["elapsed_sec"] = time.monotonic() - started
            attempt["requests"] = getattr(client, "last_requests", [])
            if record is not None:
                record(attempt)
    raise AssertionError("unreachable")  # pragma: no cover


from vmr.media.frames import extract_frame


def compact_dense_events(entries: list[dict]) -> list[dict]:
    """Remove exact duplicate dense segments for the query-facing Wiki.

    Raw window answers remain untouched in frames.jsonl and caption_audit.jsonl.
    Never union event ranges: even repeated captions in overlapping windows may
    describe separate observations, and widening them would reduce retrieval
    precision. Caption comparison ignores only case and whitespace.
    """
    unique = {}
    for entry in entries:
        for event in entry["events"]:
            current = dict(event)
            key = (
                current["start"],
                current["end"],
                current["kind"],
                " ".join(current["caption"].lower().split()),
            )
            unique.setdefault(key, current)
    return sorted(
        unique.values(),
        key=lambda event: (
            event["start"],
            event["end"],
            event["kind"],
            event["caption"],
        ),
    )


def _time_range(start: float, end: float) -> str:
    return f"{start}s" if start == end else f"{start}s–{end}s"


def render_wiki(
    duration: float,
    interval: float,
    entries: list[dict],
    *,
    sampled_frame_count: int | None = None,
    window_frames: int = 1,
    stride_frames: int = 1,
    caption_mode: str = "simple",
) -> str:
    """Query-independent and identity-neutral: the video id is a lookup key
    into public benchmark data, and the agent never needs it to locate moments."""
    lines = [
        "# Video",
        "",
        "## Metadata",
        "",
        f"- Duration: {duration} seconds",
        f"- Sampling interval: {interval} seconds",
    ]
    if caption_mode == "dense":
        compact_events = compact_dense_events(entries)
        lines += [
            "- Caption mode: dense",
            f"- Number of sampled frames: {sampled_frame_count}",
            f"- Number of caption windows: {len(entries)}",
            f"- Number of dense events: {sum(len(entry['events']) for entry in entries)}",
            f"- Number of displayed segments: {len(compact_events)}",
            f"- Caption window: {window_frames} sampled frames",
            f"- Caption stride: {stride_frames} sampled frames",
            "- Event boundaries are sampled observations, not exact action boundaries.",
            "- Detailed context windows and frame paths: `frames.jsonl`.",
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
    if caption_mode == "dense":
        buckets = {}
        for event in compact_events:
            first_bucket = int(event["start"] // 30) * 30
            last_bucket = int(event["end"] // 30) * 30
            for bucket in range(first_bucket, last_bucket + 1, 30):
                buckets.setdefault(bucket, []).append(event)
        for bucket, events in sorted(buckets.items()):
            lines += [
                f"### Segments overlapping {bucket}s–{min(bucket + 30, duration)}s",
                "",
            ]
            for event in events:
                lines.append(
                    f"- `{_time_range(event['start'], event['end'])}` "
                    f"**{event['kind']}** — {event['caption']}"
                )
        if not buckets:
            lines.append("No meaningful visual segment was identified.")
        lines.append("")
        return "\n".join(lines)
    for entry in entries:
        if "frames" not in entry:
            lines += [
                f"### {entry['timestamp']}s",
                "",
                entry["caption"],
                "",
                f"Frame: `{entry['frame']}`",
                "",
            ]
            continue
        start, end = entry["start_timestamp"], entry["end_timestamp"]
        heading = f"### {start}s" if start == end else f"### {start}s–{end}s"
        lines += [heading, "", entry["caption"], "", "Frames:"]
        lines += [
            f"- {frame['timestamp']}s: `{frame['frame']}`" for frame in entry["frames"]
        ]
        lines.append("")
    return "\n".join(lines)


def publish_window(
    video,
    video_id,
    output,
    staging,
    checkpoint,
    client,
    cfg,
    duration,
    video_stream_duration,
    source_hash,
    content_hash,
    ffmpeg_version,
    check,
    *,
    cancel_event=None,
    runner=None,
):
    sampling_times = sample_times(
        video_stream_duration, cfg["ingest"]["sample_interval_sec"]
    )
    sampled_frames = [
        {
            "frame_id": f"f{index:06d}",
            "timestamp": timestamp,
            "frame": f"frames/{index:06d}.jpg",
        }
        for index, timestamp in enumerate(sampling_times, 1)
    ]
    window_frames = cfg["ingest"]["caption_window_frames"]
    stride_frames = cfg["ingest"]["caption_stride_frames"]
    caption_mode = cfg["ingest"]["caption_mode"]
    windows = caption_windows(sampled_frames, window_frames, stride_frames)
    targets = (
        dense_target_timestamps(windows)
        if caption_mode == "dense"
        else [None] * len(windows)
    )
    used_ids = {frame["frame_id"] for window in windows for frame in window}
    extraction_sec = 0.0
    extraction_total_sec = 0.0
    reused_frames = reused_windows = 0
    for frame in sampled_frames:
        if frame["frame_id"] not in used_ids:
            continue
        _check_cancelled(cancel_event)
        image = checkpoint.image(frame)
        if image is None:
            image = checkpoint.root / frame["frame"]
            image.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            extract_frame(video, frame["timestamp"], image, cfg["ingest"])
            elapsed = time.monotonic() - started
            extraction_sec += elapsed
            checkpoint.seal_image(frame, elapsed)
        else:
            reused_frames += 1
        extraction_total_sec += checkpoint.read(frame["frame"] + ".json")["elapsed_sec"]
        shutil.copyfile(image, staging / frame["frame"])

    entries = []
    audit = []
    for window_index, (window, target_timestamps) in enumerate(
        zip(windows, targets), 1
    ):
        _check_cancelled(cancel_event)
        window_id = f"w{window_index:06d}"
        checkpoint_name = f"windows/{window_id}.json"
        saved = checkpoint.read(checkpoint_name)
        if saved is not None:
            if saved["frames"] != window:
                raise HarnessError(f"Checkpoint window changed: {window_id}")
            entries.append(saved["entry"])
            audit.append(
                {"window_id": window_id, "attempts": checkpoint.attempts(window_id)}
            )
            reused_windows += 1
            continue
        # Stable paths allow encoded image reuse across windows and retries.
        images = [checkpoint.root / frame["frame"] for frame in window]

        def record(attempt):
            attempt["model"] = cfg["ingest"]["vlm"]["model"]
            # Transport can change between resumptions without changing
            # content identity; retain what was used for each attempt.
            attempt["transport"] = {
                key: cfg["ingest"]["vlm"].get(key)
                for key in (
                    "provider",
                    "base_url",
                    "api_key_env",
                    "timeout_sec",
                    "max_retries",
                )
            }
            checkpoint.record_attempt(window_id, attempt)

        if caption_mode == "dense":
            timestamps = [frame["timestamp"] for frame in window]
            events = dense_events(
                client,
                images,
                timestamps,
                cfg["ingest"]["caption_max_repairs"],
                cancel_event=cancel_event,
                timestamp_mode=cfg["ingest"].get(
                    "dense_timestamp_mode", "absolute_seconds"
                ),
                target_timestamps=target_timestamps,
                record=record,
            )
            _check_cancelled(cancel_event)
            entries.append(
                {
                    "window_id": f"w{window_index:06d}",
                    "start_timestamp": window[0]["timestamp"],
                    "end_timestamp": window[-1]["timestamp"],
                    "target_start_timestamp": target_timestamps[0],
                    "target_end_timestamp": target_timestamps[-1],
                    "frames": window,
                    "events": events,
                }
            )
        else:
            caption_input = images[0] if window_frames == 1 else images
            started = time.monotonic()
            attempt = {}
            try:
                caption = nonempty(client.caption(caption_input), "caption")
                attempt.update(status="success", raw_response=caption)
            except BaseException as exc:
                attempt.update(
                    status="failed",
                    error=str(exc)
                    if isinstance(exc, HarnessError)
                    else type(exc).__name__,
                )
                raise
            finally:
                attempt.update(
                    elapsed_sec=time.monotonic() - started,
                    requests=getattr(client, "last_requests", []),
                )
                record(attempt)
            _check_cancelled(cancel_event)
            if window_frames == 1:
                entries.append({**window[0], "caption": caption})
            else:
                entries.append(
                    {
                        "window_id": window_id,
                        "start_timestamp": window[0]["timestamp"],
                        "end_timestamp": window[-1]["timestamp"],
                        "frames": window,
                        "caption": caption,
                    }
                )
        checkpoint.write(checkpoint_name, {"frames": window, "entry": entries[-1]})
        audit.append(
            {"window_id": window_id, "attempts": checkpoint.attempts(window_id)}
        )
    write_jsonl(staging / "caption_audit.jsonl", audit)
    attempts = [attempt for row in audit for attempt in row["attempts"]]
    requests = [request for attempt in attempts for request in attempt["requests"]]
    write_jsonl(staging / "frames.jsonl", entries)
    atomic_text(
        staging / "wiki.md",
        render_wiki(
            duration,
            cfg["ingest"]["sample_interval_sec"],
            entries,
            sampled_frame_count=len(used_ids),
            window_frames=window_frames,
            stride_frames=stride_frames,
            caption_mode=caption_mode,
        ),
    )
    if file_hash(video) != source_hash:
        raise HarnessError("Source video changed during ingest")
    _check_cancelled(cancel_event)
    metadata = {
        "version": 1,
        "video_id": video_id,
        "duration": duration,
        "video_stream_duration": video_stream_duration,
        "source_sha256": source_hash,
        "ingest_config": cfg["ingest"],
        # Hash of content-determining settings only; transport,
        # auth, timeouts, and retries remain provenance metadata.
        "ingest_config_hash": content_hash,
        "created_at": now(),
        "ffmpeg_version": ffmpeg_version,
        "caption_processing_version": cfg["ingest"].get(
            "caption_processing_version", 4
        ),
        "telemetry": {
            "extraction_sec_this_run": extraction_sec,
            "extraction_sec": extraction_total_sec,
            "caption_sec": sum(a["elapsed_sec"] for a in attempts),
            "caption_attempts": len(attempts),
            "rejected_attempts": sum(a["status"] != "success" for a in attempts),
            "api_requests": len(requests),
            "requests_with_usage": sum(bool(r.get("usage")) for r in requests),
            "requests_with_total_tokens": sum(
                "total_tokens" in r.get("usage", {}) for r in requests
            ),
            "usage": {
                key: sum(r.get("usage", {}).get(key, 0) for r in requests)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "reused_frames": reused_frames,
            "reused_windows": reused_windows,
            "window_count": len(windows),
        },
        "content_hashes": tree_hashes(staging),
    }
    write_json(staging / cfg.get("build_record", "ingest.json"), metadata)
    staging.rename(output)
    try:
        remove_tree(checkpoint.root)
    except OSError:
        logger.warning(
            "Wiki published successfully; could not remove checkpoint: %s",
            checkpoint.root,
        )
    return metadata
