from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from adapters.base import DatasetAdapter
from harness.common import (HarnessError, cli, file_hash, identifier, nonempty,
                            number, read_json, read_jsonl, write_json)


class QVHighlightsAdapter(DatasetAdapter):
    name = "qvhighlights"
    gt_semantics = "multiple_relevant_instances"
    evaluator = "qvhighlights"

    def convert(self, annotations: Path, video_root: Path, split: str) -> tuple[list, list, list]:
        nonempty(split, "split")
        videos, queries, truth = {}, [], []
        for row in read_jsonl(annotations):
            if "split" in row and row["split"] != split:
                raise HarnessError("Raw annotation split disagrees with supplied split")
            raw_id = row.get("qid")
            if type(raw_id) not in (int, str):
                raise HarnessError("QVHighlights qid must be an integer or string")
            qid = identifier(str(raw_id), "qid")
            vid = identifier(row.get("vid"), "vid")
            duration = number(row.get("duration"), "duration", 0)
            video = {"video_id": vid, "video_path": str((video_root / f"{vid}.mp4").resolve()),
                     "duration": duration, "split": split}
            if vid in videos and videos[vid] != video:
                raise HarnessError(f"Inconsistent metadata for video {vid}")
            videos[vid] = video
            query = nonempty(row.get("query"), "query")
            queries.append({"query_id": qid, "video_id": vid, "split": split, "query": query})
            if "relevant_windows" not in row:
                continue
            windows = row.get("relevant_windows")
            if not isinstance(windows, list) or not windows:
                raise HarnessError(f"{qid}: relevant_windows, when present, must be a nonempty list")
            moments = []
            for window in windows:
                if not isinstance(window, list) or len(window) != 2:
                    raise HarnessError(f"{qid}: each relevant window must be [start, end]")
                moments.append({"start_sec": window[0], "end_sec": window[1]})
            truth.append({"query_id": qid, "video_id": vid, "split": split, "moments": moments})
        return list(videos.values()), queries, truth

    @staticmethod
    def discover_splits(directory: Path) -> dict[str, Path]:
        # Only this adapter knows QVHighlights' release filename convention.
        prefix, suffix = "highlight_", "_release.jsonl"
        sources = {p.name[len(prefix):-len(suffix)]: p
                   for p in sorted(directory.glob(f"{prefix}*{suffix}"))}
        if not sources:
            raise HarnessError("No highlight_<split>_release.jsonl annotation files found")
        return sources


def evaluate_predictions(predictions: dict, ground_truth: dict, *, official_root: Path | None = None) -> dict:
    from adapters.qvhighlights_metrics import evaluate_qvhighlights
    result = {"gt_semantics": "multiple_relevant_instances", "primary_metric": "MR-full-mAP"}
    if official_root is not None:
        result["official_source_hashes"] = {
            name: file_hash(official_root / "standalone_eval" / name) for name in ("eval.py", "utils.py")}
        with tempfile.TemporaryDirectory(prefix="vmr-evaluate-") as directory:
            payload, output = Path(directory) / "input.json", Path(directory) / "output.json"
            write_json(payload, {"predictions": predictions, "ground_truth": ground_truth})
            try:
                subprocess.run([sys.executable, str(Path(__file__).with_name("qvhighlights_official.py")),
                                "--root", str(official_root.resolve()), "--input", str(payload),
                                "--output", str(output)], check=True, timeout=600)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise HarnessError("Official evaluator failed; no silent fallback was applied") from exc
            result["benchmark"] = read_json(output)
        result["implementation"] = "official-with-empty-prediction-adapter"
    else:
        result["benchmark"] = evaluate_qvhighlights(predictions, ground_truth)
        result["implementation"] = "dataset-adapter"
    result["primary_score"] = result["benchmark"]["brief"]["MR-full-mAP"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Convert QVHighlights splits without renaming them")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", action="append", help="PATH with --split, or repeated SPLIT=PATH")
    source.add_argument("--annotation-dir", type=Path, help="Discover highlight_<split>_release.jsonl files")
    parser.add_argument("--split", help="Exact original split name for a single annotation file")
    parser.add_argument("--default-eval-split", help="Optional labeled split used when evaluation omits --split")
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/qvhighlights"))
    args = parser.parse_args()
    adapter = QVHighlightsAdapter()
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
