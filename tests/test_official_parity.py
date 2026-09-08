"""Opt in with VMR_OFFICIAL_ROOT=/path/to/moment_detr and the official extra installed."""
import copy
import os
import random
from pathlib import Path

import pytest

from adapters.qvhighlights_metrics import evaluate_qvhighlights
from harness.common import write_jsonl
from harness.evaluate import evaluate
from harness.official_eval import evaluate_official


@pytest.mark.skipif(not os.environ.get("VMR_OFFICIAL_ROOT"), reason="official checkout not configured")
def test_official_parity_randomized_multimoment_and_failed_queries(tmp_path):
    pytest.importorskip("sklearn")
    root = Path(os.environ["VMR_OFFICIAL_ROOT"])
    rng = random.Random(20260908)
    truth, predictions = {}, {}
    for i in range(80):
        qid = str(i)
        gt = []
        for j in range(rng.randint(1, 4)):
            start = rng.uniform(0, 100)
            gt.append({"start_sec": start, "end_sec": min(150, start + rng.uniform(1, 50))})
        truth[qid] = {"query_id": qid, "video_id": "v", "moments": gt}
        if i % 7 == 0:  # missing predictions must stay in the denominator
            continue
        pred = []
        for j in range(0 if i % 9 == 0 else rng.randint(1, 10)):
            target = rng.choice(gt)
            start = max(0, target["start_sec"] + rng.uniform(-5, 5))
            end = max(start + 0.1, min(150, target["end_sec"] + rng.uniform(-5, 5)))
            pred.append({"start_sec": start, "end_sec": end, "score": round(1 - j * 0.08, 3), "evidence": "fixture"})
        predictions[qid] = {"query_id": qid, "video_id": "v", "moments": pred}
    actual = evaluate_qvhighlights(predictions, truth)
    expected = evaluate_official(root, copy.deepcopy(predictions), copy.deepcopy(truth))
    assert actual == expected
    write_jsonl(tmp_path / "gt.jsonl", truth.values())
    write_jsonl(tmp_path / "pred.jsonl", predictions.values())
    metrics = evaluate(tmp_path / "pred.jsonl", tmp_path / "gt.jsonl", evaluator="qvhighlights",
                       official_root=root, max_predictions=10)
    assert metrics["benchmark"] == expected
    assert metrics["implementation"] == "official-with-empty-prediction-adapter"

