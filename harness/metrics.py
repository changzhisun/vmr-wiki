from __future__ import annotations


def temporal_iou(a: dict, b: dict) -> float:
    intersection = max(0.0, min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"]))
    union = a["end_sec"] - a["start_sec"] + b["end_sec"] - b["start_sec"] - intersection
    return intersection / union if union > 0 else 0.0


def retrieval_metrics(predictions: dict, ground_truth: dict, top_k: list[int], thresholds: list[float]) -> dict:
    """Query-level recall: any top-K prediction hits any acceptable GT. Percent units."""
    n = len(ground_truth)
    result = {}
    for k in top_k:
        for threshold in thresholds:
            hits = sum(any(temporal_iou(p, g) >= threshold
                           for p in predictions.get(qid, {}).get("moments", [])[:k]
                           for g in gt["moments"]) for qid, gt in ground_truth.items())
            result[f"R@{k},IoU={threshold:g}"] = 100.0 * hits / n
    return result


def average_precision(predictions: list[dict], ground_truth: list[dict], threshold: float) -> float:
    """Interpolated AP with one-to-one greedy matching, as in QVHighlights/ActivityNet.

    Input order is the ranking; ties retain that order. Repeated predictions cannot
    match a GT instance twice. No prediction and uncovered GT contribute zero.
    """
    if not ground_truth or not predictions:
        return 0.0
    matched = set()
    precision, recall = [0.0], [0.0]
    true_positives = 0
    for rank, prediction in enumerate(predictions, 1):
        overlaps = [(temporal_iou(prediction, gt), i) for i, gt in enumerate(ground_truth)]
        for overlap, index in sorted(overlaps, reverse=True):
            if overlap < threshold:
                break
            if index not in matched:
                matched.add(index)
                true_positives += 1
                break
        precision.append(true_positives / rank)
        recall.append(true_positives / len(ground_truth))
    precision.append(0.0)
    recall.append(1.0)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    return sum((recall[i] - recall[i - 1]) * precision[i]
               for i in range(1, len(recall)) if recall[i] != recall[i - 1])

