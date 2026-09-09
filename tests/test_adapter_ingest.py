from pathlib import Path
import threading

import pytest

from adapters.qvhighlights import QVHighlightsAdapter
from harness.common import HarnessError, read_json, read_jsonl, write_json, write_jsonl
from harness.freeze import freeze_dataset, freeze_wiki, verify_wiki
from harness.ingest import ingest_video, probe_duration, probe_durations, sample_times
from harness.ingest_all import _run_parallel, ingest_all
from harness.run_query import Experiment


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
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x", "relevant_windows": None}],
    [{"qid": 1, "vid": "../v", "duration": 3, "query": "x", "relevant_windows": [[0, 1]]}],
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x", "relevant_windows": [[2, 4]]}],
    [{"qid": 1, "vid": "v", "duration": 3, "query": "x", "relevant_windows": [[0, 1]]}] * 2,
])
def test_adapter_rejects_invalid_data(tmp_path, rows):
    raw = tmp_path / "raw.jsonl"
    write_jsonl(raw, rows)
    with pytest.raises(HarnessError):
        QVHighlightsAdapter().prepare(raw, tmp_path, tmp_path / "dataset", split="train")
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


def test_verify_wiki_requires_freeze(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    with pytest.raises(HarnessError, match="ingested but not frozen"):
        verify_wiki(root)
    with pytest.raises(HarnessError, match="Frozen wiki not found"):
        verify_wiki(root.parent / "missing")


def test_freeze_dataset_requires_ingest(prepared):
    cfg, _ = prepared
    with pytest.raises(HarnessError, match="have no ingest and cannot be frozen"):
        freeze_dataset(cfg)


def test_freeze_reports_which_ingest_setting_changed(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    cfg["ingest"]["vlm"]["max_tokens"] = cfg["ingest"]["vlm"]["max_tokens"] + 1
    with pytest.raises(HarnessError, match="vlm.max_tokens"):
        freeze_dataset(cfg)


def test_probe_duration_prefers_container_over_stream(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    monkeypatch.setattr("harness.ingest.media_command", lambda _cmd: (
        '{"streams":[{"codec_type":"video","duration":"10.366667"}],'
        '"format":{"duration":"12.020000"}}'
    ))
    assert probe_durations(video) == pytest.approx((12.02, 10.366667))
    assert probe_duration(video) == pytest.approx(12.02)


def test_ingest_samples_only_within_video_stream(cfg, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    output = tmp_path / "wiki" / "clip"
    cfg["ingest"]["sample_interval_sec"] = 5.0
    monkeypatch.setattr("harness.ingest.probe_durations", lambda _path: (12.02, 10.0))
    monkeypatch.setattr(
        "harness.ingest.extract_frame",
        lambda _video, _timestamp, path, _cfg: path.write_bytes(b"jpeg"),
    )
    monkeypatch.setattr("harness.ingest.media_command", lambda _cmd: "ffmpeg version test")

    class Captioner:
        def caption(self, _path):
            return "A visible scene."

    metadata = ingest_video(video, "clip", output, cfg, captioner=Captioner())
    assert metadata["duration"] == pytest.approx(12.02)
    assert metadata["video_stream_duration"] == pytest.approx(10.0)
    assert [row["timestamp"] for row in read_jsonl(output / "frames.jsonl")] == [0.0, 5.0]


def test_pipeline_does_not_compare_ingest_and_annotation_durations(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos" / "video"
    metadata = read_json(root / "ingest.json")
    metadata["duration"] += 5.0
    write_json(root / "ingest.json", metadata)
    freeze_dataset(cfg)

    class Runner:
        provenance = {"runtime": "test"}

    experiment = Experiment(cfg, "duration-not-checked", runner=Runner())
    assert experiment.freeze["videos"]["video"] == verify_wiki(root)["wiki_hash"]


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
    wiki = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos"
    assert not wiki.exists() or list(wiki.iterdir()) == []


def test_cache_invalidates_on_content_settings_not_transport(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3

    # Transport/auth changes remain provenance and do not invalidate captions.
    original_provider = cfg["ingest"]["vlm"]["provider"]
    original_base_url = cfg["ingest"]["vlm"]["base_url"]
    cfg["ingest"]["vlm"]["provider"] = "another-compatible-provider"
    cfg["ingest"]["vlm"]["base_url"] += "/other"
    cfg["ingest"]["vlm"]["timeout_sec"] = 30
    cfg["ingest"]["vlm"]["max_retries"] = 5
    cfg["ingest"]["vlm"]["api_key_env"] = "OTHER_API_KEY"
    ingest_all(cfg, captioner=captioner)
    assert captioner.calls == 3

    # A content-determining model change still invalidates the cache.
    cfg["ingest"]["vlm"]["provider"] = original_provider
    cfg["ingest"]["vlm"]["base_url"] = original_base_url
    cfg["ingest"]["vlm"]["timeout_sec"] = 120
    cfg["ingest"]["vlm"]["max_retries"] = 3
    cfg["ingest"]["vlm"]["model"] = "other-model"
    with pytest.raises(HarnessError, match="different video/settings"):
        ingest_all(cfg, captioner=captioner, jobs=1)


def test_parallel_interrupt_propagates_cancellation_and_joins_workers(monkeypatch):
    worker_started = threading.Event()
    worker_stopped = threading.Event()

    def fake_ingest(vid, video, path, output, cfg, *, captioner, cancel_event):
        if vid == "slow":
            worker_started.set()
            cancel_event.wait(2)
            worker_stopped.set()
            raise HarnessError("Ingest cancelled")
        assert worker_started.wait(1)
        return {"video_id": vid}, False

    class InterruptingBar:
        def update(self):
            raise KeyboardInterrupt

    monkeypatch.setattr("harness.ingest_all._ingest_one", fake_ingest)
    tasks = [("slow", {}, Path("slow"), Path("out-slow")),
             ("fast", {}, Path("fast"), Path("out-fast"))]
    cancelled = threading.Event()
    with pytest.raises(KeyboardInterrupt):
        _run_parallel(tasks, {}, None, 2, InterruptingBar(), cancelled)
    assert cancelled.is_set()
    assert worker_stopped.is_set()
