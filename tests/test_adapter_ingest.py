from pathlib import Path

import pytest

from adapters.qvhighlights import QVHighlightsAdapter
from harness.common import HarnessError, read_jsonl, write_jsonl
from harness.freeze import freeze_dataset, freeze_wiki, verify_wiki
from harness.ingest import sample_times
from harness.ingest_all import ingest_all


def test_adapter_preserves_all_gt_and_hides_labels(prepared):
    cfg, _ = prepared
    dataset = Path(cfg["paths"]["datasets"]) / "qvhighlights"
    queries = read_jsonl(dataset / "queries.jsonl")
    truth = read_jsonl(dataset / "ground_truth.jsonl")
    assert len(queries) == 3
    assert all(set(q) == {"query_id", "video_id", "query", "split"} for q in queries)
    assert all(len(g["moments"]) == 2 for g in truth)
    assert len(read_jsonl(dataset / "videos.jsonl")) == 1


@pytest.mark.parametrize("rows", [
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x"}],
    [{"qid": 1, "vid": "../v", "duration": 3, "query": "x", "relevant_windows": [[0, 1]]}],
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x", "relevant_windows": [[2, 4]]}],
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x", "relevant_windows": [[0, 1]]}] * 2,
])
def test_adapter_rejects_invalid_data(tmp_path, rows):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, rows)
    with pytest.raises(HarnessError):
        QVHighlightsAdapter().prepare(raw, tmp_path, tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_sampling_edges():
    assert sample_times(10.0, 5.0) == [0.0, 5.0]
    assert sample_times(10.1, 5.0) == [0.0, 5.0, 10.0]
    assert sample_times(0.1, 5.0) == [0.0]
    for interval in (0, -1, float("nan")):
        with pytest.raises(HarnessError):
            sample_times(10.0, interval)


def test_ingest_once_query_independent_and_freeze(prepared):
    cfg, captioner = prepared
    dataset = Path(cfg["paths"]["datasets"]) / "qvhighlights"
    # Ingest must work without being able to parse either query or ground truth.
    (dataset / "queries.jsonl").write_text("not JSON")
    (dataset / "ground_truth.jsonl").write_text("not JSON")
    ingest_all(cfg, captioner=captioner)
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    frames = read_jsonl(root / "frames.jsonl")
    assert [f["timestamp"] for f in frames] == [0.0, 1.0, 2.0]
    assert all(set(f) == {"frame_id", "timestamp", "frame", "caption"} for f in frames)
    manifest = freeze_dataset(cfg)
    assert freeze_dataset(cfg) == manifest
    assert verify_wiki(root)["wiki_hash"] == manifest["videos"]["video"]
    assert (root / "frames" / "000001.jpg").stat().st_mode & 0o222 == 0
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3
    cfg["ingest"]["sample_interval_sec"] = 2.0
    with pytest.raises(HarnessError):
        ingest_all(cfg, captioner=captioner)


def test_frame_tampering_detected(frozen):
    cfg, _ = frozen
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    image = root / "frames" / "000001.jpg"
    image.chmod(0o644)
    image.write_bytes(b"corrupted")
    with pytest.raises(HarnessError, match="integrity"):
        verify_wiki(root)


def test_failed_ingest_has_no_partial_output(prepared):
    cfg, _ = prepared
    class BrokenCaptioner:
        def caption(self, _):
            raise HarnessError("API unavailable")
    with pytest.raises(HarnessError):
        ingest_all(cfg, captioner=BrokenCaptioner())
    wiki = Path(cfg["paths"]["wiki"]) / "qvhighlights"
    assert list(wiki.iterdir()) == []


def test_cache_invalidates_on_content_settings_not_on_transport(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3

    # Transport-only changes (endpoint, timeout, retries) must NOT invalidate
    # an existing ingest: verification reuses the cached captions.
    cfg["ingest"]["vlm"]["base_url"] += "/other"
    cfg["ingest"]["vlm"]["timeout_sec"] = 30
    cfg["ingest"]["vlm"]["max_retries"] = 5
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3

    # A content-determining change (model) MUST invalidate the cache.
    cfg["ingest"]["vlm"]["base_url"] = cfg["ingest"]["vlm"]["base_url"].removesuffix("/other")
    cfg["ingest"]["vlm"]["timeout_sec"] = 120
    cfg["ingest"]["vlm"]["max_retries"] = 3
    cfg["ingest"]["vlm"]["model"] = "other-model"
    with pytest.raises(HarnessError, match="different video/settings"):
        ingest_all(cfg, captioner=captioner, jobs=1)

