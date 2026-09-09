"""Run explicitly supplied QVHighlights official code in an evaluator-only process."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import sys
from pathlib import Path

from adapters.qvhighlights_metrics import RANGES, THRESHOLDS, brief_metrics, grouped_truth
from harness.common import cli, read_json, write_json


def evaluate_official(root: Path, predictions: dict, ground_truth: dict) -> dict:
    sys.path.insert(0, str(root.resolve()))
    from standalone_eval.eval import compute_mr_ap, compute_mr_r1

    groups = {}
    for name in RANGES:
        truth = grouped_truth(ground_truth, name)
        if not truth:
            ap = {str(t): None for t in THRESHOLDS}
            ap["average"] = None
            r1 = {str(t): None for t in THRESHOLDS}
        else:
            submission, labels = [], []
            for qid, gt in truth.items():
                windows = [[m["start_sec"], m["end_sec"], m["score"]]
                           for m in predictions.get(qid, {}).get("moments", [])]
                # Official R1 indexes [0] and AP skips empty arrays. A zero-length
                # nonmatching sentinel keeps these failed/abstained queries in the denominator.
                submission.append({"qid": qid, "pred_relevant_windows": windows or [[0.0, 0.0, 0.0]]})
                labels.append({"qid": qid, "relevant_windows": [
                    [m["start_sec"], m["end_sec"]] for m in gt["moments"]]})
            ap = compute_mr_ap(submission, labels, num_workers=1)
            r1 = compute_mr_r1(submission, labels)
        groups[name] = {"num_queries": len(truth), "MR-mAP": ap, "MR-R1": r1}
    return {"brief": brief_metrics(groups), **groups}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = read_json(args.input)
    write_json(args.output, evaluate_official(args.root, payload["predictions"], payload["ground_truth"]))


if __name__ == "__main__":
    cli(main)

