from __future__ import annotations
from pathlib import Path
from vmr.core.errors import HarnessError
from .ffmpeg import media_command


def extract_frame(video: Path, timestamp: float, output: Path, cfg: dict) -> None:
    size = cfg["image_max_size"]
    scale = f"scale=w='min({size},iw)':h='min({size},ih)':force_original_aspect_ratio=decrease"
    media_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-ss",
            f"{timestamp:.9f}",
            "-i",
            str(video),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-vf",
            scale,
            "-q:v",
            str(cfg["jpeg_quality"]),
            "-threads",
            "1",
            "-y",
            str(output),
        ]
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise HarnessError(f"No frame decoded at {timestamp}s")
