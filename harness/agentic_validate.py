"""Deterministic host-side validation of an agent-compiled wiki.

The agent validates its own output before exiting, but a self-check is not
evidence. Every rule below is re-applied here, and a violation fails the video
rather than publishing a wiki the query stage would reject later.
"""
from __future__ import annotations

import re
from pathlib import Path

from harness.common import HarnessError, nonempty, number, read_jsonl

# Exactly these three artifacts. Anything else is a leaked intermediate.
ARTIFACTS = ("frames.jsonl", "frames/", "wiki.md")
WIKI_TITLE = "# Video"

FRAME_ID = re.compile(r"^f[0-9]{6}$")
FRAME_PATH = re.compile(r"^frames/[0-9]{6}\.jpg$")

REQUIRED_FIELDS = ("frame_id", "timestamp", "frame", "reason", "description")
OPTIONAL_FIELDS = ("entities", "objects", "location", "shot_id")
REASONS = ("periodic_sample", "scene_boundary", "event_boundary", "action_boundary",
           "boundary_refinement", "semantic_evidence")

# Any frames/... file the agent linked from the wiki, in Markdown links,
# inline code and bare prose alike. An extension is required so prose naming
# the directory itself is not mistaken for a reference.
WIKI_FRAME_REFERENCE = re.compile(r"frames/[A-Za-z0-9_.\-/]*\.[A-Za-z0-9]+")

# Only real traversal: prose ellipses are not path segments.
PATH_TRAVERSAL = re.compile(r"(?:^|[\s(\[`'\"/])\.\./")

# A container or host path in a published wiki makes it unusable elsewhere and
# reveals the layout the agent ran under.
FORBIDDEN_PREFIXES = ("/work", "/workspace", "/input", "/scratch", "/home", "/tmp",
                      "/Users", "file://")

# A short identifier cannot be told apart from ordinary prose, and a false
# positive discards a whole ingest. Only distinctive tokens are searched for,
# on word boundaries; public benchmark video ids are well above this length.
MIN_LEAK_TOKEN = 8


def _leaked(text: str, tokens) -> str | None:
    for token in tokens:
        if not token or len(token) < MIN_LEAK_TOKEN:
            continue
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(token) + r"(?![A-Za-z0-9_])", text):
            return token
    return None


def _entries(root: Path) -> list[Path]:
    listing = sorted(root.iterdir())
    for path in listing:
        if path.is_symlink():
            raise HarnessError(f"Symlink is forbidden in agent output: {path.name}")
    return listing


def _validate_files(output: Path, max_wiki_bytes: int, max_frame_bytes: int,
                    max_frames: int) -> dict[str, Path]:
    if not output.is_dir() or output.is_symlink():
        raise HarnessError("Agent produced no output directory")
    names = [path.name + ("/" if path.is_dir() else "") for path in _entries(output)]
    if sorted(names) != sorted(ARTIFACTS):
        raise HarnessError(
            f"Expected exactly {', '.join(ARTIFACTS)} in output/; got {', '.join(names) or 'nothing'}")
    frames_dir = output / "frames"
    if not frames_dir.is_dir():
        raise HarnessError("output/frames must be a directory")
    if (output / "wiki.md").stat().st_size > max_wiki_bytes:
        raise HarnessError(f"wiki.md exceeds {max_wiki_bytes} bytes")
    if (output / "frames.jsonl").stat().st_size > max_wiki_bytes:
        raise HarnessError(f"frames.jsonl exceeds {max_wiki_bytes} bytes")
    images = {}
    for path in _entries(frames_dir):
        if not path.is_file():
            raise HarnessError(f"output/frames must contain only image files: {path.name}")
        if path.stat().st_size == 0:
            raise HarnessError(f"Empty frame image: {path.name}")
        if path.stat().st_size > max_frame_bytes:
            raise HarnessError(f"Frame image exceeds {max_frame_bytes} bytes: {path.name}")
        images["frames/" + path.name] = path
    if not images:
        raise HarnessError("Agent registered no evidence frames")
    if len(images) > max_frames:
        raise HarnessError(f"Agent produced {len(images)} frames; the limit is {max_frames}")
    return images


def _validate_registry(output: Path, images: dict[str, Path], duration: float) -> list[dict]:
    rows = read_jsonl(output / "frames.jsonl")
    if not rows:
        raise HarnessError("frames.jsonl registers no frames")
    seen_ids, seen_paths, previous = set(), set(), None
    for index, row in enumerate(rows, 1):
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise HarnessError(f"frames.jsonl line {index} is missing {', '.join(missing)}")
        unknown = set(row) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS)
        if unknown:
            raise HarnessError(
                f"frames.jsonl line {index} has unknown field(s): {', '.join(sorted(unknown))}")
        frame_id = row["frame_id"]
        if not isinstance(frame_id, str) or not FRAME_ID.fullmatch(frame_id):
            raise HarnessError(f"frames.jsonl line {index}: frame_id must match ^f[0-9]{{6}}$")
        if frame_id in seen_ids:
            raise HarnessError(f"Duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)
        frame = row["frame"]
        if not isinstance(frame, str) or not FRAME_PATH.fullmatch(frame):
            raise HarnessError(
                f"frames.jsonl line {index}: frame must be a relative frames/NNNNNN.jpg path")
        if frame not in images:
            raise HarnessError(f"frames.jsonl line {index} registers a missing image: {frame}")
        if frame in seen_paths:
            raise HarnessError(f"Duplicate frame path: {frame}")
        seen_paths.add(frame)
        # The registry is the index; an unordered one cannot be searched by time.
        timestamp = number(row["timestamp"], f"frames.jsonl line {index} timestamp", 0)
        if timestamp > duration:
            raise HarnessError(
                f"frames.jsonl line {index} timestamp {timestamp} exceeds the video duration {duration}")
        if previous is not None and timestamp <= previous:
            raise HarnessError("frames.jsonl timestamps must be strictly increasing")
        previous = timestamp
        if row["reason"] not in REASONS:
            raise HarnessError(
                f"frames.jsonl line {index}: reason must be one of {', '.join(REASONS)}")
        nonempty(row["description"], f"frames.jsonl line {index} description")
        for field in ("entities", "objects"):
            if field in row and (not isinstance(row[field], list)
                                 or not all(isinstance(item, str) and item.strip()
                                            for item in row[field])):
                raise HarnessError(f"frames.jsonl line {index}: {field} must be a list of strings")
        for field in ("location", "shot_id"):
            if field in row:
                nonempty(row[field], f"frames.jsonl line {index} {field}")
    orphans = sorted(set(images) - seen_paths)
    if orphans:
        raise HarnessError(
            f"{len(orphans)} frame image(s) are not registered in frames.jsonl: {orphans[0]}")
    return rows


def _validate_wiki(output: Path, registered: set[str], forbidden: tuple[str, ...]) -> None:
    text = (output / "wiki.md").read_text(encoding="utf-8")
    if text.split("\n", 1)[0].strip() != WIKI_TITLE:
        raise HarnessError(
            f"wiki.md must start with {WIKI_TITLE!r}: a titled wiki names its own video, "
            "and the query agent would receive a public dataset identifier")
    if not text.strip():
        raise HarnessError("wiki.md is empty")
    for prefix in FORBIDDEN_PREFIXES:
        if prefix in text:
            raise HarnessError(f"wiki.md contains a non-portable absolute path: {prefix}")
    if PATH_TRAVERSAL.search(text):
        raise HarnessError("wiki.md must not contain path traversal (../)")
    for reference in WIKI_FRAME_REFERENCE.findall(text):
        if reference not in registered:
            raise HarnessError(f"wiki.md references an unregistered frame: {reference}")
    leak = _leaked(text, forbidden)
    if leak is not None:
        raise HarnessError(
            f"wiki.md leaks the video identity {leak!r}; the wiki must stay identity-neutral")


def validate_agent_wiki(output: Path, *, duration: float, max_frames: int,
                        max_wiki_bytes: int, max_frame_bytes: int,
                        forbidden_tokens: tuple[str, ...] = ()) -> list[dict]:
    """Return the validated frame registry, or raise on the first violation."""
    images = _validate_files(output, max_wiki_bytes, max_frame_bytes, max_frames)
    rows = _validate_registry(output, images, duration)
    _validate_wiki(output, set(images), forbidden_tokens)
    leak = _leaked((output / "frames.jsonl").read_text(encoding="utf-8"), forbidden_tokens)
    if leak is not None:
        raise HarnessError(
            f"frames.jsonl leaks the video identity {leak!r}; the wiki must stay identity-neutral")
    return rows
