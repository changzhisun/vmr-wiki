"""Query-independent, bounded-memory visual change analysis and frame selection."""
from __future__ import annotations

import math
import os
import select
import subprocess
import time
from pathlib import Path

from harness.common import HarnessError


def frame_timestamps(video: Path, duration: float) -> list[float]:
    """Read presentation timestamps by demuxing, without decoding full images.
    Snap sampling requests to real frames, especially near EOF and on VFR clips.
    """
    from harness.ingest import media_command
    from harness.common import parse_json
    payload = parse_json(media_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=start_time:packet=pts_time", "-of", "json", str(video)]))
    try:
        origin = float(payload.get("streams", [{}])[0].get("start_time", 0))
        stamps = sorted({round(float(packet["pts_time"]) - origin, 9)
                         for packet in payload.get("packets", []) if "pts_time" in packet})
    except (ValueError, TypeError, IndexError) as exc:
        raise HarnessError("Invalid source video presentation timestamps") from exc
    stamps = [stamp for stamp in stamps if math.isfinite(stamp) and 0 <= stamp < duration]
    if not stamps:
        raise HarnessError("Cannot determine source video frame presentation timestamps")
    return stamps


def uniform_times(start: float, end: float, count: int, duration: float) -> list[float]:
    # Seeking at EOF produces no image. These are requested sampling times,
    # not claims about source-frame PTS or exact semantic boundaries.
    end = min(end, max(0.0, duration - min(0.001, duration / 2)))
    start = min(start, end)
    if count <= 1 or start == end:
        return [round(start, 9)]
    return sorted({round(start + (end - start) * i / (count - 1), 9)
                   for i in range(count)})


def analyze_changes(video: Path, duration: float, config: dict, check_cancelled) -> list[dict]:
    """Stream tiny grayscale frames; histogram change estimates cuts, pixel MAD
    estimates motion/visual change (including camera motion, not optical flow).
    Only scores survive, so RAM does not grow with decoded video size.
    """
    fps = config["analysis_fps"]
    width, height = 64, 36
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video),
               "-map", "0:v:0", "-t", str(duration), "-vf",
               f"fps={fps},scale={width}:{height},format=gray", "-threads", "1",
               "-f", "rawvideo", "pipe:1"]
    # stderr goes to a temporary file so neither output pipe can deadlock.
    import tempfile
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        started = time.monotonic()
        previous = previous_histogram = None
        pending = bytearray()
        scores = []
        count = 0
        try:
            while True:
                check_cancelled()
                if time.monotonic() - started > config["analysis_timeout_sec"]:
                    raise HarnessError("Visual change analysis timed out")
                if not select.select([process.stdout], [], [], 0.2)[0]:
                    continue
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                pending.extend(chunk)
                while len(pending) >= width * height:
                    frame = bytes(pending[:width * height])
                    del pending[:width * height]
                    histogram = [0] * 16
                    for value in frame:
                        histogram[value // 16] += 1
                    timestamp = count / fps
                    if previous is not None and timestamp < duration:
                        scores.append({
                            "timestamp": round(timestamp, 9),
                            "scene": sum(abs(a - b) for a, b in zip(histogram, previous_histogram))
                                     / (2 * len(frame)),
                            "motion": sum(abs(a - b) for a, b in zip(frame, previous))
                                      / (255 * len(frame)),
                        })
                    previous, previous_histogram = frame, histogram
                    count += 1
            process.wait(timeout=10)
            if process.returncode or pending or count == 0:
                errors.seek(0)
                raise HarnessError("Visual change analysis failed: " +
                                   errors.read()[-2000:].decode(errors="replace"))
            return scores
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def mixed_times(start: float, end: float, duration: float, scores: list[dict],
                config: dict, *, global_scan: bool = False) -> list[float]:
    count = min(config["max_frames"], max(2, math.ceil((end - start) /
                                                    config["min_sample_interval_sec"])))
    if global_scan:
        return uniform_times(start, end, count, duration)
    uniform_count = max(2, math.ceil(count * config["uniform_fraction"]))
    selected = set(uniform_times(start, end, uniform_count, duration))
    candidates = [row for row in scores if start <= row["timestamp"] < end]
    separation = (end - start) / (count * 3)
    scene_budget = (count - len(selected)) // 2
    offset = 1 / config["analysis_fps"]
    for kind, budget in (("scene", scene_budget), ("motion", count - len(selected) - scene_budget)):
        added = 0
        for row in sorted(candidates, key=lambda r: (-r[kind], r["timestamp"])):
            if row[kind] <= 0 or (kind == "scene" and row[kind] < config["scene_threshold"]):
                continue
            near = [row["timestamp"], row["timestamp"] - offset, row["timestamp"] + offset]
            for stamp in near if kind == "scene" else near[:1]:
                if added >= budget or len(selected) >= count:
                    break
                if not start <= stamp < min(end, duration):
                    continue
                stamp = round(stamp, 9)
                distance = min(separation, offset / 2) if kind == "scene" else separation
                if all(abs(stamp - old) >= distance for old in selected):
                    selected.add(stamp)
                    added += 1
            if added >= budget:
                break
    # Quiet scenes still receive the full available uniform coverage.
    for stamp in uniform_times(start, end, count, duration):
        if len(selected) >= count:
            break
        selected.add(stamp)
    return sorted(selected)


def context_bounds(segments: list[dict], index: int, start: float, end: float,
                   overlap: float) -> tuple[float, float]:
    """Each shared boundary gets overlap * shorter target duration of context,
    half on each side. Semantic targets themselves remain a partition.
    """
    node = segments[index]
    length = node["end"] - node["start"]
    left = (overlap * min(length, segments[index - 1]["end"] - segments[index - 1]["start"]) / 2
            if index else 0)
    right = (overlap * min(length, segments[index + 1]["end"] - segments[index + 1]["start"]) / 2
             if index + 1 < len(segments) else 0)
    return max(start, node["start"] - left), min(end, node["end"] + right)
