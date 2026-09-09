from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from adapters import get_evaluator
from harness.common import (HarnessError, cli, file_hash, identifier, number, object_hash, positive_int,
                            read_json, read_jsonl, unique_index, write_json)
from harness.config import dataset_path, load_config
from harness.dataset import (load_dataset, load_query_inputs, require_ground_truth,
                             select_split, validate_ground_truth)
from harness.metrics import retrieval_metrics
from harness.results import check_identity, experiment_context, validate_result, verify_bundle


def evaluate(pred_path: Path, gt_path: Path | None = None, *, dataset_dir: Path | None = None,
             split: str | None = None, evaluator: str | None = None,
             top_k=(1, 5), iou_thresholds=(0.3, 0.5, 0.7), max_predictions: int = 5,
             metadata_dir: Path | None = None, official_root: Path | None = None,
             allow_unverified_predictions: bool = False) -> dict:
    if dataset_dir is None:
        if gt_path is None:
            raise HarnessError("Evaluation requires a dataset context")
        dataset_dir = gt_path.parent
    dataset = load_dataset(dataset_dir)
    split = select_split(dataset, split, evaluation=True)
    # MUST precede opening a GT path, predictions, or invoking any evaluator.
    require_ground_truth(dataset, split)
    if not top_k or not iou_thresholds:
        raise HarnessError("Evaluation settings cannot be empty")
    for k in top_k:
        positive_int(k, "top_k")
    for threshold in iou_thresholds:
        if not 0 < number(threshold, "IoU threshold") <= 1:
            raise HarnessError("IoU thresholds must be in (0, 1]")
    videos, queries = load_query_inputs(dataset_dir, split, dataset)
    experiment_path = pred_path.parent / "experiment.json"
    if experiment_path.exists():
        experiment_context(pred_path.parent, dataset_dir, split)
        if metadata_dir is None:
            metadata_dir = pred_path.parent / "run_metadata"
    gt_path = gt_path if gt_path is not None else dataset_dir / "ground_truth.jsonl"
    if gt_path.resolve().parent != dataset_dir.resolve():
        raise HarnessError("GT must belong to the selected dataset directory")
    truth = validate_ground_truth(read_jsonl(gt_path), dataset, split, videos, queries)
    predictions = unique_index(read_jsonl(pred_path), "query_id")
    query_index = unique_index(queries, "query_id")
    for prediction in predictions.values():
        validate_result(prediction, query_index, videos, split, max_predictions)
    # Missing provenance is accepted only through an explicit raw-submission
    # opt-in; deleting a sidecar can never silently downgrade verification.
    verify_bundle(pred_path, dataset, split, queries, required=not allow_unverified_predictions)
    failed = set(truth) - set(predictions)
    statuses = {}
    if metadata_dir is not None:
        if not metadata_dir.is_dir():
            raise HarnessError(f"Run metadata directory not found: {metadata_dir}")
        for path in sorted(metadata_dir.glob("*.json")):
            meta = read_json(path)
            check_identity(meta, dataset["name"], split)
            qid = identifier(meta.get("query_id"), "query_id")
            if path.stem != qid or qid not in truth or qid in statuses:
                raise HarnessError(f"Invalid run metadata: {path}")
            status = meta.get("status")
            if status not in ("success", "failed", "running"):
                raise HarnessError(f"Unknown run status: {status}")
            if meta.get("video_id") != truth[qid]["video_id"]:
                raise HarnessError(f"Metadata video mismatch for {qid}")
            if status != "success":
                if qid in predictions:
                    raise HarnessError(f"Failed run must not have a prediction: {qid}")
                failed.add(qid)
            elif qid not in predictions:
                raise HarnessError(f"Successful run missing prediction: {qid}")
            statuses[qid] = status
        if set(predictions) - set(statuses):
            raise HarnessError("Predictions lack run metadata")
    evaluator = evaluator or dataset.get("evaluator", "generic")
    result = {
        "dataset": dataset["name"], "split": split,
        "evaluator": evaluator, "metric_units": "percent", "num_queries": len(truth),
        "failed_runs": len(failed), "failed_query_ids": sorted(failed),
        "missing_predictions": len(truth.keys() - predictions.keys()),
        "abstained_queries": sum(not row["moments"] for row in predictions.values()),
        "average_predictions_per_query": sum(len(p["moments"]) for p in predictions.values()) / len(truth),
        "top_k": list(top_k), "iou_thresholds": list(iou_thresholds),
        "retrieval": retrieval_metrics(predictions, truth, list(top_k), list(iou_thresholds)),
        "inputs": {"prediction_sha256": file_hash(pred_path), "ground_truth_sha256": file_hash(gt_path),
                   "dataset_hash": object_hash(dataset), "queries_hash": object_hash(queries)},
    }
    result.update(get_evaluator(evaluator)(predictions, truth, official_root=official_root))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate one declared split; unlabeled splits can only generate predictions")
    parser.add_argument("--dataset", help="Dataset name; defaults to saved experiment or config")
    parser.add_argument("--split", help="Exact split name; saved config or default_eval_split when omitted")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gt", type=Path, help="Optional GT path inside the selected dataset directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluator", help="Evaluator adapter override; defaults to dataset.json")
    parser.add_argument("--official-root", type=Path, help="Official evaluator checkout understood by the dataset adapter")
    parser.add_argument("--metadata-dir", type=Path)
    parser.add_argument("--allow-unverified-predictions", action="store_true",
                        help="Accept raw prediction JSONL without an aggregate provenance sidecar")
    args = parser.parse_args()
    saved = args.pred.parent / "config.yaml"
    cfg = load_config(args.config or (saved if saved.exists() else "config.yaml"))
    if args.dataset:
        cfg["dataset"]["name"] = identifier(args.dataset)
    directory = args.gt.parent if args.gt is not None and args.dataset is None else dataset_path(cfg, "datasets")
    if args.dataset:
        load_dataset(directory, args.dataset)
    evaluation = cfg["evaluation"]
    result = evaluate(args.pred, args.gt, dataset_dir=directory,
                      split=args.split if args.split is not None else cfg["dataset"].get("split"),
                      evaluator=args.evaluator, top_k=evaluation["top_k"],
                      iou_thresholds=evaluation["iou_thresholds"],
                      max_predictions=cfg["query"]["max_predictions"],
                      metadata_dir=args.metadata_dir, official_root=args.official_root,
                      allow_unverified_predictions=args.allow_unverified_predictions)
    output = args.output or args.pred.parent / "metrics.json"
    write_json(output, result)
    print(f"{output}: {result['dataset']}/{result['split']}, {result['num_queries']} queries, {result['failed_runs']} failed runs")


if __name__ == "__main__":
    cli(main)
