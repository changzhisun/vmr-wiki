"""Moment retrieval semantics of moment_detr/standalone_eval (no highlight task).

Reference: https://github.com/jayleicn/moment_detr/tree/main/standalone_eval
AP uses at most the first 10 predictions at IoU 0.50:0.05:0.95. This implementation
explicitly counts missing/empty predictions as zero and represents empty length
groups as null instead of the upstream evaluator's NaNs/errors.
"""
from __future__ import annotations

from harness.metrics import average_precision, retrieval_metrics


THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
RANGES = {"short": (0, 10), "middle": (10, 30), "long": (30, 150), "full": (0, 150)}


def grouped_truth(ground_truth: dict, name: str) -> dict:
    if name == "full":
        return ground_truth
    low, high = RANGES[name]
    result = {}
    for qid, gt in ground_truth.items():
        moments = [m for m in gt["moments"] if low < m["end_sec"] - m["start_sec"] <= high]
        if moments:
            result[qid] = {**gt, "moments": moments}
    return result


def brief_metrics(groups: dict) -> dict:
    brief = {f"MR-{name}-mAP": groups[name]["MR-mAP"]["average"] for name in RANGES}
    for threshold in (0.5, 0.75):
        brief[f"MR-full-mAP@{threshold}"] = groups["full"]["MR-mAP"][str(threshold)]
    for threshold in (0.5, 0.7):
        brief[f"MR-full-R1@{threshold}"] = groups["full"]["MR-R1"][str(threshold)]
    return brief


def evaluate_qvhighlights(predictions: dict, ground_truth: dict) -> dict:
    groups = {}
    for name in RANGES:
        truth = grouped_truth(ground_truth, name)
        ap, r1 = {}, {}
        for threshold in THRESHOLDS:
            ap[str(threshold)] = (sum(average_precision(
                predictions.get(qid, {}).get("moments", [])[:10], gt["moments"], threshold)
                for qid, gt in truth.items()) / len(truth) * 100 if truth else None)
        # Upstream averages unrounded AP values, then rounds each metric to 2 decimals.
        ap["average"] = sum(ap.values()) / len(THRESHOLDS) if truth else None
        ap = {key: round(value, 2) if value is not None else None for key, value in ap.items()}
        if truth:
            generic = retrieval_metrics(predictions, truth, [1], THRESHOLDS)
            r1 = {str(t): round(generic[f"R@1,IoU={t:g}"], 2) for t in THRESHOLDS}
        else:
            r1 = {str(t): None for t in THRESHOLDS}
        groups[name] = {"num_queries": len(truth), "MR-mAP": ap, "MR-R1": r1}
    return {"brief": brief_metrics(groups), **groups}

