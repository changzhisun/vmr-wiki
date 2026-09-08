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

import yaml

from adapters.qvhighlights_metrics import evaluate_qvhighlights
from harness.common import (HarnessError, cli, file_hash, identifier, number, positive_int,
                            read_json, read_jsonl, unique_index, write_json)
from harness.metrics import retrieval_metrics
from harness.validate import validate_moment, validate_prediction


def evaluate(pred_path: Path, gt_path: Path, *, evaluator: str = "generic",
             top_k=(1, 5), iou_thresholds=(0.3, 0.5, 0.7), max_predictions: int = 5,
             metadata_dir: Path | None = None, official_root: Path | None = None) -> dict:
    if evaluator not in ("generic", "qvhighlights"):
        raise HarnessError(f"Unknown evaluator: {evaluator}")
    if not top_k or not iou_thresholds:
        raise HarnessError("Evaluation settings cannot be empty")
    for k in top_k:
        positive_int(k, "top_k")
    for threshold in iou_thresholds:
        if not 0 < number(threshold, "IoU threshold") <= 1:
            raise HarnessError("IoU thresholds must be in (0, 1]")
    truth = unique_index(read_jsonl(gt_path), "query_id")
    if not truth:
        raise HarnessError("Ground truth must not be empty")
    for gt in truth.values():
        identifier(gt.get("video_id"), "video_id")
        if not isinstance(gt.get("moments"), list) or not gt["moments"]:
            raise HarnessError("Each labeled query needs at least one GT moment")
        for moment in gt["moments"]:
            validate_moment(moment)
    predictions = unique_index(read_jsonl(pred_path), "query_id")
    unknown = predictions.keys() - truth.keys()
    if unknown:
        raise HarnessError(f"Predictions contain unknown queries: {sorted(unknown)}")
    for qid, prediction in predictions.items():
        validate_prediction(prediction, query_id=qid, video_id=truth[qid]["video_id"],
                            max_predictions=max_predictions)
    failed = set(truth) - set(predictions)
    statuses = {}
    if metadata_dir is not None:
        for path in sorted(metadata_dir.glob("*.json")):
            meta = read_json(path)
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
    result = {
        "evaluator": evaluator, "metric_units": "percent", "num_queries": len(truth),
        "failed_runs": len(failed), "failed_query_ids": sorted(failed),
        "missing_predictions": len(truth.keys() - predictions.keys()),
        "abstained_queries": sum(not row["moments"] for row in predictions.values()),
        "average_predictions_per_query": sum(len(p["moments"]) for p in predictions.values()) / len(truth),
        "top_k": list(top_k), "iou_thresholds": list(iou_thresholds),
        "retrieval": retrieval_metrics(predictions, truth, list(top_k), list(iou_thresholds)),
        "inputs": {"prediction_sha256": file_hash(pred_path), "ground_truth_sha256": file_hash(gt_path)},
    }
    if evaluator == "qvhighlights":
        if official_root is not None:
            eval_source = official_root / "standalone_eval" / "eval.py"
            utils_source = official_root / "standalone_eval" / "utils.py"
            result["official_source_hashes"] = {"eval.py": file_hash(eval_source), "utils.py": file_hash(utils_source)}
            with tempfile.TemporaryDirectory(prefix="vmr-evaluate-") as directory:
                payload, output = Path(directory) / "input.json", Path(directory) / "output.json"
                write_json(payload, {"predictions": predictions, "ground_truth": truth})
                try:
                    subprocess.run([sys.executable, str(Path(__file__).with_name("official_eval.py")),
                                    "--root", str(official_root.resolve()), "--input", str(payload),
                                    "--output", str(output)], check=True, timeout=600)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    raise HarnessError("Official evaluator failed; no silent fallback was applied") from exc
                result["benchmark"] = read_json(output)
            result["implementation"] = "official-with-empty-prediction-adapter"
        else:
            result["benchmark"] = evaluate_qvhighlights(predictions, truth)
            result["implementation"] = "dataset-adapter"
        result["gt_semantics"] = "multiple_relevant_instances"
        result["primary_metric"] = "MR-full-mAP"
        result["primary_score"] = result["benchmark"]["brief"]["MR-full-mAP"]
    else:
        if official_root is not None:
            raise HarnessError("--official-root is only supported for qvhighlights")
        result["implementation"] = "generic"
        result["gt_semantics"] = "any_acceptable_moment"
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate ranked moments; missing queries count as failures")
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--evaluator", choices=["generic", "qvhighlights"])
    parser.add_argument("--official-root", type=Path, help="Checkout of jayleicn/moment_detr")
    parser.add_argument("--metadata-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config or args.pred.parent / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    evaluation = cfg.get("evaluation", {})
    dataset_meta = args.gt.parent / "dataset_metadata.json"
    dataset_evaluator = read_json(dataset_meta).get("evaluator") if dataset_meta.exists() else None
    evaluator = args.evaluator or dataset_evaluator or evaluation.get("evaluator", "generic")
    meta = args.metadata_dir
    if meta is None and (args.pred.parent / "run_metadata").is_dir():
        meta = args.pred.parent / "run_metadata"
    result = evaluate(args.pred, args.gt, evaluator=evaluator,
                      top_k=evaluation.get("top_k", [1, 5]),
                      iou_thresholds=evaluation.get("iou_thresholds", [0.3, 0.5, 0.7]),
                      max_predictions=cfg.get("query", {}).get("max_predictions", 5),
                      metadata_dir=meta, official_root=args.official_root)
    output = args.output or args.pred.parent / "metrics.json"
    write_json(output, result)
    print(f"{output}: {result['num_queries']} queries, {result['failed_runs']} failed runs")


if __name__ == "__main__":
    cli(main)
