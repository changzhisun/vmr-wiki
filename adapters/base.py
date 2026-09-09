from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from harness.common import HarnessError, file_hash, unique_index, write_json, write_jsonl
from harness.dataset import validate_dataset, validate_ground_truth, validate_videos
from harness.validate import validate_query


class DatasetAdapter(ABC):
    name: str
    gt_semantics: str
    evaluator: str = "generic"

    @abstractmethod
    def convert(self, annotations: Path, video_root: Path, split: str) -> tuple[list, list, list]:
        """Convert one named annotation split, preserving its original name."""

    def prepare(self, annotations: Path | dict[str, Path], video_root: Path, output: Path,
                *, split: str | None = None, default_eval_split: str | None = None) -> None:
        if isinstance(annotations, Path):
            if split is None:
                raise HarnessError("Specify the annotation's original --split; it is never guessed")
            sources = {split: annotations}
        else:
            if split is not None:
                raise HarnessError("Do not combine a split mapping with a single --split")
            sources = annotations
        videos, queries, truth, splits = [], [], [], {}
        for name, source in sources.items():
            v, q, gt = self.convert(source, video_root, name)
            if not q:
                raise HarnessError(f"Split {name!r} contains no queries")
            if any(row.get("split") != name for row in [*v, *q, *gt]):
                raise HarnessError(f"Adapter changed the original split name {name!r}")
            videos.extend(v)
            queries.extend(q)
            truth.extend(gt)
            splits[name] = {"has_ground_truth": bool(gt)}
        metadata = {"name": self.name, "splits": splits, "evaluator": self.evaluator,
                    "gt_semantics": self.gt_semantics,
                    "annotation_sha256": {name: file_hash(path) for name, path in sources.items()}}
        if default_eval_split is not None:
            metadata["default_eval_split"] = default_eval_split
        validate_dataset(metadata)
        for name in splits:
            video_index = validate_videos(videos, metadata, name)
            split_queries = [q for q in queries if q["split"] == name]
            unique_index(split_queries, "query_id")
            for query in split_queries:
                validate_query(query)
                if query["video_id"] not in video_index:
                    raise HarnessError("Query and video split memberships do not match")
            if splits[name]["has_ground_truth"]:
                validate_ground_truth(truth, metadata, name, video_index, split_queries)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "dataset.json", metadata)
        write_jsonl(output / "videos.jsonl", videos)
        write_jsonl(output / "queries.jsonl", queries)
        if truth:
            write_jsonl(output / "ground_truth.jsonl", truth)
