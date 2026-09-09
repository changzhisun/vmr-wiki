from pathlib import Path

from conftest import write_evaluation_dataset

import pytest

from adapters.qvhighlights_metrics import evaluate_qvhighlights
from harness.aggregate import aggregate
from harness.common import HarnessError, write_json, write_jsonl
from harness.evaluate import evaluate
from harness.metrics import average_precision, temporal_iou


def moment(start, end, score=1.0):
    return {"start_sec": start, "end_sec": end, "score": score, "evidence": "test"}


def test_iou_boundary_and_ap_duplicate_penalty():
    assert temporal_iou(moment(0, 1), moment(1, 2)) == 0
    assert temporal_iou(moment(0, 2), moment(1, 3)) == pytest.approx(1 / 3)
    gt = [moment(0, 1), moment(2, 3)]
    assert average_precision([moment(0, 1)], gt, 0.5) == 0.5
    assert average_precision([moment(0, 1), moment(0, 1), moment(2, 3)], gt, 0.5) == pytest.approx(5 / 6)
    assert average_precision([], gt, 0.5) == 0


def test_any_gt_hit_missing_denominator_and_abstention(tmp_path):
    gt = [{"query_id": qid, "video_id": "v", "split": "train", "moments": [moment(0, 1), moment(2, 3)]}
          for qid in ["a", "b", "c"]]
    pred = [{"query_id": "a", "video_id": "v", "split": "train", "moments": [moment(2, 3)]},
            {"query_id": "b", "video_id": "v", "split": "train", "moments": []}]
    write_evaluation_dataset(tmp_path, gt)
    write_jsonl(tmp_path / "gt.jsonl", gt)
    write_jsonl(tmp_path / "pred.jsonl", pred)
    metrics = evaluate(tmp_path / "pred.jsonl", tmp_path / "gt.jsonl", evaluator="qvhighlights",
                       allow_unverified_predictions=True)
    assert metrics["failed_runs"] == 1
    assert metrics["abstained_queries"] == 1
    assert metrics["retrieval"]["R@1,IoU=0.5"] == pytest.approx(100 / 3)
    assert metrics["primary_score"] == 16.67
    assert metrics["benchmark"]["long"]["MR-mAP"]["average"] is None


def test_qvh_length_groups_and_top_ten_cap():
    gt = {"q": {"query_id": "q", "video_id": "v", "split": "train", "moments": [moment(0, 10), moment(10, 40), moment(40, 100)]}}
    pred = {"q": {"moments": [moment(110, 120)] * 10 + [moment(0, 10)]}}
    result = evaluate_qvhighlights(pred, gt)
    assert result["brief"]["MR-full-mAP"] == 0
    assert all(result[name]["num_queries"] == 1 for name in ("short", "middle", "long", "full"))


def test_aggregate_validates_and_preserves_multiple_moments(tmp_path):
    inputs = tmp_path / "predictions"
    inputs.mkdir()
    row = {"query_id": "q", "video_id": "v", "split": "train", "moments": [moment(0, 1), moment(2, 3)]}
    write_evaluation_dataset(tmp_path, [row])
    write_json(inputs / "q.json", row)
    assert aggregate(inputs, tmp_path / "all.jsonl", dataset_dir=tmp_path, split="train") == [row]
    with pytest.raises(HarnessError):
        aggregate(inputs, tmp_path / "all.jsonl", max_predictions=1, dataset_dir=tmp_path, split="train")


def test_unknown_and_duplicate_predictions_rejected(tmp_path):
    truth = {"query_id": "q", "video_id": "v", "split": "train", "moments": [moment(0, 1)]}
    write_evaluation_dataset(tmp_path, [truth])
    write_jsonl(tmp_path / "gt.jsonl", [truth])
    for rows in ([truth, truth], [{**truth, "query_id": "unknown"}], [{**truth, "video_id": "wrong"}]):
        write_jsonl(tmp_path / "pred.jsonl", rows)
        with pytest.raises(HarnessError):
            evaluate(tmp_path / "pred.jsonl", tmp_path / "gt.jsonl",
                     allow_unverified_predictions=True)
