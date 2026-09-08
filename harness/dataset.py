"""Dataset identity and opaque split selection. Never interpret split names."""
from __future__ import annotations

from pathlib import Path

from harness.common import (HarnessError, identifier, nonempty, number, read_json,
                            read_jsonl, unique_index)
from harness.validate import validate_moment, validate_query


MIGRATION = "Re-run the Dataset Adapter to create dataset.json and split-bearing manifests."


def dataset_context(cfg: dict, split: str | None = None, *, evaluation: bool = False):
    from harness.config import dataset_path
    directory = dataset_path(cfg, "datasets")
    metadata = load_dataset(directory, cfg["dataset"]["name"])
    selected = select_split(metadata, split if split is not None else cfg["dataset"].get("split"),
                            evaluation=evaluation)
    return directory, metadata, selected


def validate_dataset(metadata: dict) -> dict:
    if not isinstance(metadata, dict):
        raise HarnessError("dataset.json must be an object")
    identifier(metadata.get("name"), "dataset name")
    splits = metadata.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise HarnessError("dataset.json must define a nonempty splits mapping")
    for name, info in splits.items():
        nonempty(name, "split")
        if not isinstance(info, dict) or type(info.get("has_ground_truth")) is not bool:
            raise HarnessError(f"Split {name!r}: has_ground_truth must be a boolean")
    if "default_eval_split" in metadata:
        default = select_split(metadata, metadata["default_eval_split"])
        if not splits[default]["has_ground_truth"]:
            raise HarnessError("default_eval_split must have ground truth")
    if "evaluator" in metadata:
        identifier(metadata["evaluator"], "evaluator")
    return metadata


def load_dataset(directory: Path, expected_name: str | None = None) -> dict:
    path = directory / "dataset.json"
    if not path.is_file():
        raise HarnessError(f"Missing {path}. {MIGRATION}")
    metadata = validate_dataset(read_json(path))
    if expected_name is not None and metadata["name"] != expected_name:
        raise HarnessError(f"Dataset name mismatch: expected {expected_name!r}, got {metadata['name']!r}")
    return metadata


def select_split(metadata: dict, split: str | None, *, evaluation: bool = False) -> str:
    if split is None and evaluation:
        split = metadata.get("default_eval_split")
    if split is None:
        raise HarnessError("Specify --split using an exact name from dataset.json")
    nonempty(split, "split")
    if split not in metadata["splits"]:
        available = "\n".join(f"- {name}" for name in metadata["splits"])
        raise HarnessError(f"Unknown split {split!r}.\n\nAvailable splits:\n{available}")
    return split


def row_split(row: dict, metadata: dict) -> str:
    if "split" not in row:
        raise HarnessError(f"Manifest row has no split. {MIGRATION}")
    return select_split(metadata, row["split"])


def validate_videos(rows: list[dict], metadata: dict, split: str) -> dict[str, dict]:
    """One membership row per (split, video_id); one physical identity per video_id."""
    selected, identities, memberships = {}, {}, set()
    for video in rows:
        member_split = row_split(video, metadata)
        if set(video) != {"video_id", "video_path", "duration", "split"}:
            raise HarnessError("Video must contain video_id, video_path, duration, split")
        vid = identifier(video["video_id"], "video_id")
        nonempty(video["video_path"], "video_path")
        if number(video["duration"], "duration") <= 0:
            raise HarnessError("duration must be positive")
        key = member_split, vid
        if key in memberships:
            raise HarnessError(f"Duplicate video_id {vid!r} in split {member_split!r}")
        memberships.add(key)
        identity = video["video_path"], video["duration"]
        if vid in identities and identities[vid] != identity:
            raise HarnessError(f"Conflicting identity for shared video {vid!r}")
        identities[vid] = identity
        if member_split == split:
            selected[vid] = video
    return selected


def load_videos(directory: Path, metadata: dict, split: str) -> dict[str, dict]:
    select_split(metadata, split)
    return validate_videos(read_jsonl(directory / "videos.jsonl"), metadata, split)


def load_query_inputs(directory: Path, split: str, metadata: dict | None = None) -> tuple[dict, list[dict]]:
    """Never opens GT. Returns only the selected split's annotation data."""
    metadata = metadata if metadata is not None else load_dataset(directory)
    split = select_split(metadata, split)
    videos = load_videos(directory, metadata, split)
    queries = [row for row in read_jsonl(directory / "queries.jsonl") if row_split(row, metadata) == split]
    unique_index(queries, "query_id")
    if not queries:
        raise HarnessError(f"Split {split!r} contains no queries")
    for query in queries:
        validate_query(query)
        if query["video_id"] not in videos:
            raise HarnessError(f"Query video {query['video_id']!r} is not in split {split!r}")
    return videos, queries


def require_ground_truth(metadata: dict, split: str) -> None:
    if not metadata["splits"][split]["has_ground_truth"]:
        raise HarnessError(f"Split {split!r} has_ground_truth=false: local evaluation is unavailable; "
                           "this split can only generate predictions.")


def validate_ground_truth(rows: list[dict], metadata: dict, split: str,
                          videos: dict, queries: list[dict]) -> dict:
    truth = unique_index([row for row in rows if row_split(row, metadata) == split], "query_id")
    query_index = unique_index(queries, "query_id")
    if truth.keys() != query_index.keys():
        raise HarnessError(f"GT and query IDs must match exactly within split {split!r}")
    for qid, gt in truth.items():
        if set(gt) != {"query_id", "video_id", "split", "moments"}:
            raise HarnessError("GT must contain query_id, video_id, split, moments")
        if gt["video_id"] != query_index[qid]["video_id"]:
            raise HarnessError(f"GT video mismatch for {qid}")
        if not isinstance(gt["moments"], list) or not gt["moments"]:
            raise HarnessError("Each labeled query needs at least one GT moment")
        for moment in gt["moments"]:
            validate_moment(moment, videos[gt["video_id"]]["duration"])
    return truth
