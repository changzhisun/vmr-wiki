from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import yaml

from harness.common import HarnessError, cli, file_hash, read_json, unique_index, write_jsonl
from harness.validate import validate_prediction


def aggregate(input_dir: Path, output: Path, max_predictions: int | None = None) -> list[dict]:
    if not input_dir.is_dir():
        raise HarnessError(f"Prediction directory not found: {input_dir}")
    saved_config = input_dir.parent / "config.yaml"
    if max_predictions is None:
        max_predictions = (yaml.safe_load(saved_config.read_text())["query"]["max_predictions"]
                           if saved_config.exists() else 5)
    rows = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix != ".json" or path.is_symlink() or not path.is_file():
            raise HarnessError(f"Unexpected prediction entry: {path}")
        prediction = validate_prediction(read_json(path), query_id=path.stem,
                                         max_predictions=max_predictions)
        meta_path = input_dir.parent / "run_metadata" / path.name
        if (input_dir.parent / "experiment.json").exists():
            metadata = read_json(meta_path)
            if metadata["status"] != "success" or metadata["prediction_hash"] != file_hash(path):
                raise HarnessError(f"Result is not an intact successful run: {path.name}")
        rows.append(prediction)
    unique_index(rows, "query_id")
    write_jsonl(output, rows)  # all moments and evidence are retained, including empty lists
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate validated predictions, preserving every moment")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-predictions", type=int)
    args = parser.parse_args()
    print(f"Aggregated {len(aggregate(args.input, args.output, args.max_predictions))} queries")


if __name__ == "__main__":
    cli(main)

