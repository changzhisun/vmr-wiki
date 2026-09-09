from pathlib import Path

import pytest

from harness.ablation import load_suite, prepare, run, summarize
from harness.common import HarnessError, read_jsonl, write_json
from harness.freeze import remove_tree
from harness.run_query import run_status
from test_query_pipeline import ProcessFixtureRunner


def test_prepare_is_matched_offline_and_reports_missing_trials(prepared, tmp_path):
    cfg, _ = prepared
    suite = tmp_path / "suite"
    manifest = prepare(cfg, suite, split="train", limit_videos=1, seed=7)
    assert manifest["video_ids"] == ["video"]
    assert manifest["num_queries"] == 3
    _, configs = load_suite(suite)
    assert len(configs) == 4
    assert all(c["query"] == cfg["query"] for c in configs.values())
    assert {c["ingest"]["sample_interval_sec"] for c in configs.values()} == {1.0}
    rows = read_jsonl(suite / "datasets/qvhighlights/videos.jsonl")
    assert Path(rows[0]["video_path"]).is_absolute()
    report = summarize(suite)
    assert all(not r["complete"] and r["metrics"] is None for r in report["variants"])
    assert all(r["caption_total_tokens"] is None and r["caption_attempts"] is None
               for r in report["variants"])
    other = prepare(cfg, tmp_path / "other", split="train", limit_videos=1, seed=7)
    assert other["video_ids"] == manifest["video_ids"]
    with pytest.raises(HarnessError, match="already exists"):
        prepare(cfg, suite, split="train")


@pytest.mark.parametrize("target", ["config", "dataset", "source", "video"])
def test_suite_rejects_changed_inputs(prepared, tmp_path, monkeypatch, target):
    cfg, _ = prepared
    suite = tmp_path / "suite"
    prepare(cfg, suite, split="train")
    if target == "config":
        path = suite / "configs/simple_1_1.yaml"
        path.write_text(path.read_text() + "\n# changed\n")
    elif target == "dataset":
        write_json(suite / "datasets/qvhighlights/dataset.json", {})
    elif target == "source":
        monkeypatch.setattr("harness.ablation.source_hash", lambda: "changed")
    else:
        row = read_jsonl(suite / "datasets/qvhighlights/videos.jsonl")[0]
        Path(row["video_path"]).write_bytes(b"changed")
    with pytest.raises(HarnessError, match="changed"):
        load_suite(suite)


def test_offline_ablation_end_to_end(prepared, tmp_path, monkeypatch):
    cfg, _ = prepared
    suite = tmp_path / "suite"
    prepare(cfg, suite, split="train", limit_videos=1)

    class FixtureVLM:
        def __init__(self, config, **kwargs):
            pass

        def caption(self, images, *, timestamps=None, **kwargs):
            return '{"events":[]}' if timestamps is not None else "A red scene."

    monkeypatch.setattr("harness.ingest.VLMClient", FixtureVLM)
    monkeypatch.setattr("harness.run_query.DockerRunner", lambda cfg: ProcessFixtureRunner())
    try:
        run(suite, "all", 1)
        report = summarize(suite)
        assert len(report["variants"]) == 4
        for row in report["variants"]:
            assert row["complete"] and row["finished_queries"] == 3
            assert row["metrics"]["primary_score"] == 100.0
            assert row["metrics"]["failed_runs"] == 0
            assert row["caption_attempts"] > 0
            assert row["caption_total_tokens"] is None  # Fake VLM has no usage.
            assert set(row["stage_wall_sec"]) == {"ingest", "query", "evaluate"}
        # A rerun verifies/reuses completed work, with no further captions.
        monkeypatch.setattr(FixtureVLM, "caption", lambda *args, **kwargs: pytest.fail("recaptioned"))
        run(suite, "all", 1)
        prediction = suite / "results/simple_1_1/predictions.jsonl"
        prediction.write_text(prediction.read_text() + "\n")
        with pytest.raises(HarnessError, match="predictions changed"):
            summarize(suite)
    finally:
        if (suite / "wiki").exists():
            remove_tree(suite / "wiki")


def test_partial_token_usage_is_not_reported_as_zero(prepared, tmp_path):
    cfg, _ = prepared
    suite = tmp_path / "suite"
    prepare(cfg, suite, split="train")
    write_json(suite / "wiki/simple_1_1/qvhighlights/videos/video/ingest.json",
               {"telemetry": {"api_requests": 1, "requests_with_usage": 1,
                              "requests_with_total_tokens": 0, "usage": {"prompt_tokens": 10}}})
    row = next(r for r in summarize(suite)["variants"] if r["variant"] == "simple_1_1")
    assert not row["tokens_complete"] and row["caption_total_tokens"] is None


def test_failure_status_contains_reason_and_log_locations(tmp_path):
    status = run_status({"query_id": "q1", "status": "failed", "failure_kind": "timeout",
                         "error": "Agent\n timed out"}, tmp_path)
    assert "[timeout] Agent timed out" in status
    assert str(tmp_path / "run_metadata/q1.json") in status
    assert str(tmp_path / "logs/q1.stderr.log") in status
    assert run_status({"query_id": "q1", "status": "success"}, tmp_path) == "q1: success"
