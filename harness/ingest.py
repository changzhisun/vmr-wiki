from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import subprocess
import tempfile
from pathlib import Path

from harness.common import (HarnessError, atomic_text, cli, file_hash, identifier,
                            now, number, object_hash, parse_json, read_json,
                            write_json, write_jsonl)
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


def probe_duration(video: Path) -> float:
    metadata = parse_json(media_command([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_type,duration:format=duration", "-of", "json", str(video),
    ]))
    if not metadata.get("streams"):
        raise HarnessError("Input contains no video stream")
    raw = metadata["streams"][0].get("duration", metadata.get("format", {}).get("duration"))
    try:
        duration = number(float(raw), "video duration", 0)
    except (TypeError, ValueError) as exc:
        raise HarnessError("Unable to determine video duration") from exc
    if duration <= 0:
        raise HarnessError("Video duration must be positive")
    return duration


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


def render_wiki(video_id: str, duration: float, interval: float, frames: list[dict]) -> str:
    lines = [f"# Video: {video_id}", "", "## Metadata", "",
             f"- Duration: {duration} seconds", f"- Sampling interval: {interval} seconds",
             f"- Number of frames: {len(frames)}", "", "## Timeline", ""]
    for frame in frames:
        lines += [f"### {frame['timestamp']}s", "", frame["caption"], "",
                  f"Frame: `{frame['frame']}`", ""]
    return "\n".join(lines)


def ingest_video(video: Path, video_id: str, output: Path, cfg: dict, *, captioner=None) -> dict:
    """No query or GT argument is accepted. Existing completed ingest is never recaptioned."""
    identifier(video_id, "video_id")
    video = video.resolve()
    source_hash = file_hash(video)
    ingest_hash = object_hash(cfg["ingest"])
    if output.exists():
        metadata = read_json(output / "ingest.json")
        if (metadata["source_sha256"], metadata["ingest_config_hash"], metadata["video_id"]) != (
                source_hash, ingest_hash, video_id):
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
        if output.exists():
            raise HarnessError("Ingest completed concurrently; rerun to verify existing output")
        duration = probe_duration(video)
        client = captioner if captioner is not None else VLMClient(cfg["ingest"]["vlm"])
        staging = Path(tempfile.mkdtemp(prefix=f".{video_id}.", dir=output.parent))
        (staging / "frames").mkdir()
        frames = []
        for index, timestamp in enumerate(sample_times(duration, cfg["ingest"]["sample_interval_sec"]), 1):
            relative = f"frames/{index:06d}.jpg"
            extract_frame(video, timestamp, staging / relative, cfg["ingest"])
            caption = client.caption(staging / relative)
            from harness.common import nonempty
            frames.append({"frame_id": f"f{index:06d}", "timestamp": timestamp,
                           "frame": relative, "caption": nonempty(caption, "caption")})
        write_jsonl(staging / "frames.jsonl", frames)
        atomic_text(staging / "wiki.md", render_wiki(
            video_id, duration, cfg["ingest"]["sample_interval_sec"], frames))
        if file_hash(video) != source_hash:
            raise HarnessError("Source video changed during ingest")
        metadata = {"version": 1, "video_id": video_id, "duration": duration,
                    "source_sha256": source_hash, "ingest_config": cfg["ingest"],
                    "ingest_config_hash": ingest_hash, "created_at": now(),
                    "ffmpeg_version": media_command(["ffmpeg", "-version"]).splitlines()[0],
                    "content_hashes": tree_hashes(staging)}
        write_json(staging / "ingest.json", metadata)
        staging.rename(output)
        return metadata
    finally:
        if staging is not None and staging.exists():
            remove_tree(staging)
        lock.unlink(missing_ok=True)


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
