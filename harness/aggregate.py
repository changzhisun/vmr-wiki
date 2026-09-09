from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from harness.common import (HarnessError, cli, file_hash, identifier, object_hash,
                            read_json, unique_index, write_json, write_jsonl)
from harness.config import dataset_path, load_config
from harness.results import bundle_metadata_path, check_identity, experiment_context, validate_result


def aggregate(input_dir: Path, output: Path, max_predictions: int | None = None, *,
              dataset_dir: Path | None = None, split: str | None = None) -> list[dict]:
    if not input_dir.is_dir():
        raise HarnessError(f"Prediction directory not found: {input_dir}")
    dataset, split, videos, queries, saved, experiment = experiment_context(input_dir.parent, dataset_dir, split)
    if max_predictions is None:
        max_predictions = saved["query"]["max_predictions"] if saved else 5
    query_index = unique_index(queries, "query_id")
    rows = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            raise HarnessError(f"Unexpected prediction entry: {path}")
        prediction = validate_result(read_json(path), query_index, videos, split, max_predictions)
        if prediction["query_id"] != path.stem:
            raise HarnessError(f"Prediction filename/query_id mismatch: {path.name}")
        meta_path = input_dir.parent / "run_metadata" / path.name
        if experiment or meta_path.exists():
            metadata = read_json(meta_path)
            check_identity(metadata, dataset["name"], split)
            if (metadata.get("query_id") != prediction["query_id"]
                    or metadata.get("video_id") != prediction["video_id"]
                    or metadata.get("status") != "success"
                    or metadata.get("prediction_hash") != file_hash(path)):
                raise HarnessError(f"Result is not an intact successful run: {path.name}")
        rows.append(prediction)
    unique_index(rows, "query_id")
    if output.resolve().parent == input_dir.resolve():
        raise HarnessError("Write aggregate output outside the individual prediction directory")
    write_jsonl(output, rows)
    write_json(bundle_metadata_path(output), {
        "dataset": dataset["name"], "split": split, "dataset_hash": object_hash(dataset),
        "queries_hash": object_hash(queries), "prediction_sha256": file_hash(output),
    })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate one dataset/split; reject mixed predictions")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", help="Dataset name; defaults to saved experiment identity")
    parser.add_argument("--split", help="Exact dataset.json split name; defaults to saved experiment split")
    parser.add_argument("--config", type=Path, help="Config for dataset paths; otherwise use experiment config")
    parser.add_argument("--max-predictions", type=int)
    args = parser.parse_args()
    directory = None
    if args.dataset or args.config:
        saved = args.input.parent / "config.yaml"
        cfg = load_config(args.config or (saved if saved.exists() else "config.yaml"))
        if args.dataset:
            cfg["dataset"]["name"] = identifier(args.dataset)
        directory = dataset_path(cfg, "datasets")
        if args.split is None:
            args.split = cfg["dataset"].get("split")
    rows = aggregate(args.input, args.output, args.max_predictions, dataset_dir=directory, split=args.split)
    print(f"Aggregated {len(rows)} queries")


if __name__ == "__main__":
    cli(main)
