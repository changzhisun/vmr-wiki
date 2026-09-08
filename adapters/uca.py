from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from adapters.base import DatasetAdapter
from harness.common import (HarnessError, cli, identifier, nonempty,
                            number, read_jsonl)


class UCAAdapter(DatasetAdapter):
    """Convert UCA-VMR annotations into the unified VMR Wiki manifests.

    UCA-VMR stores one row per query with a single gold moment per query.
    The ``path`` field is relative to the video root, e.g.
    ``Videos/Abuse/Abuse043_x264.mp4``.
    """

    name = "uca"
    gt_semantics = "any_acceptable_moment"
    evaluator = "generic"

    def convert(self, annotations: Path, video_root: Path, split: str) -> tuple[list, list, list]:
        nonempty(split, "split")
        video_root = video_root.resolve()
        videos, queries, truth = {}, [], []
        for row in read_jsonl(annotations):
            raw_id = row.get("query_id")
            if type(raw_id) not in (int, str):
                raise HarnessError("UCA-VMR query_id must be an integer or string")
            qid = identifier(str(raw_id), "query_id")
            vid = identifier(row.get("video_id"), "video_id")
            duration = number(row.get("duration"), "duration", 0)
            rel_path = nonempty(row.get("path"), "path")
            video_path = (video_root / rel_path).resolve()
            if not video_path.is_relative_to(video_root):
                raise HarnessError(f"{qid}: path {rel_path!r} escapes the video root")
            video = {"video_id": vid, "video_path": str(video_path),
                     "duration": duration, "split": split}
            if vid in videos and videos[vid] != video:
                raise HarnessError(f"Inconsistent metadata for video {vid}")
            videos[vid] = video
            query = row.get("query")
            if not isinstance(query, dict):
                raise HarnessError(f"{qid}: query must be an object with a text field")
            query_text = query.get("text")
            if not isinstance(query_text, str) or not query_text.strip():
                raise HarnessError(f"{qid}: query.text must be a nonempty string")
            queries.append({"query_id": qid, "video_id": vid, "split": split,
                            "query": query_text})
            moments = row.get("gold_moments")
            if moments is None:
                continue
            if not isinstance(moments, list) or not moments:
                raise HarnessError(f"{qid}: gold_moments, when present, must be a nonempty list")
            gt = []
            for window in moments:
                if not isinstance(window, list) or len(window) != 2:
                    raise HarnessError(f"{qid}: each gold moment must be [start, end]")
                gt.append({"start_sec": window[0], "end_sec": window[1]})
            truth.append({"query_id": qid, "video_id": vid, "split": split, "moments": gt})
        return list(videos.values()), queries, truth

    @staticmethod
    def discover_splits(directory: Path) -> dict[str, Path]:
        # Only this adapter knows UCA-VMR's release filename convention: <split>.jsonl
        sources = {p.stem: p for p in sorted(directory.glob("*.jsonl"))}
        if not sources:
            raise HarnessError("No <split>.jsonl annotation files found")
        return sources


def main():
    parser = argparse.ArgumentParser(description="Convert UCA-VMR splits without renaming them")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", action="append", help="PATH with --split, or repeated SPLIT=PATH")
    source.add_argument("--annotation-dir", type=Path, help="Discover <split>.jsonl files")
    parser.add_argument("--split", help="Exact original split name for a single annotation file")
    parser.add_argument("--default-eval-split", help="Optional labeled split used when evaluation omits --split")
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/uca"))
    args = parser.parse_args()
    adapter = UCAAdapter()
    if args.annotation_dir:
        if args.split:
            raise HarnessError("--split is only for a single --annotations PATH")
        sources = adapter.discover_splits(args.annotation_dir)
    elif args.split:
        if len(args.annotations) != 1:
            raise HarnessError("Use repeated SPLIT=PATH for multiple annotation files")
        sources = {args.split: Path(args.annotations[0])}
    else:
        sources = {}
        for entry in args.annotations:
            name, separator, path = entry.partition("=")
            if not separator or not name or not path:
                raise HarnessError("Use --annotations PATH --split NAME, or --annotations SPLIT=PATH")
            if name in sources:
                raise HarnessError(f"Duplicate split source: {name!r}")
            sources[name] = Path(path)
    adapter.prepare(sources, args.video_root, args.output, default_eval_split=args.default_eval_split)
    print(args.output)


if __name__ == "__main__":
    cli(main)