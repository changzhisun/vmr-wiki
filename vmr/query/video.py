"""Pinned raw video sources for queries without a WikiSet."""

from dataclasses import dataclass
from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.hashing import file_hash


@dataclass(frozen=True)
class VideoSource:
    path: Path
    digest: str
    duration: float

    def verify(self) -> None:
        if self.path.is_symlink() or not self.path.is_file():
            raise HarnessError(f"Video source is not a regular file: {self.path}")
        if file_hash(self.path) != self.digest:
            raise HarnessError(
                f"Video changed after experiment initialization: {self.path}"
            )


def video_sources(
    dataset: Path, videos: dict, video_root: Path
) -> dict[str, VideoSource]:
    from vmr.media.probe import probe_durations

    root = Path(video_root).resolve()
    if not root.is_dir():
        raise HarnessError(f"Video root is not a directory: {video_root}")
    sources = {}
    for vid, row in videos.items():
        path = Path(row["video_path"])
        if not path.is_absolute():
            path = dataset / path
        if path.is_symlink() or not path.is_file():
            raise HarnessError(f"Video source is not a regular file: {path}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise HarnessError(f"Video source is outside --video-root: {path}")
        container, _stream = probe_durations(resolved)
        source = VideoSource(resolved, file_hash(resolved), container)
        sources[vid] = source
    return sources
