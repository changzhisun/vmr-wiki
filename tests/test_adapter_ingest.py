import json
from pathlib import Path
import threading

import pytest
import yaml

from adapters.qvhighlights import QVHighlightsAdapter
from harness.common import (HarnessError, ingest_content_hash, parse_json, read_json, read_jsonl,
                            write_json, write_jsonl)
from harness.config import load_config
from harness.freeze import freeze_dataset, freeze_wiki, verify_wiki
from harness.ingest import (caption_windows, ingest_video, probe_duration, probe_durations,
                            parse_dense_events, sample_times)
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


def test_default_config_uses_overlapping_multi_frame_captions():
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    ingest = config["ingest"]
    assert ingest["sample_interval_sec"] == 1.0
    assert ingest["caption_mode"] == "dense"
    assert ingest["caption_window_frames"] == 4
    assert ingest["caption_stride_frames"] == 1
    assert ingest["vlm"]["max_tokens"] == 2048
    assert ingest["vlm"]["prompt"].count("{{FRAME_TIMESTAMPS}}") == 1


def test_caption_mode_and_dense_prompt_template_are_validated(tmp_path):
    source = Path(__file__).resolve().parents[1] / "config.yaml"
    raw = yaml.safe_load(source.read_text())
    raw["ingest"]["caption_mode"] = "dense"
    raw["ingest"]["vlm"]["prompt"] = "No frame timeline placeholder."
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="exactly one.*FRAME_TIMESTAMPS"):
        load_config(path)

    raw["ingest"].pop("caption_mode")
    path.write_text(yaml.safe_dump(raw))
    assert load_config(path)["ingest"]["caption_mode"] == "simple"


def test_caption_windows_support_overlap_and_partial_tail():
    frames = [{"frame_id": f"f{i}"} for i in range(5)]
    windows = caption_windows(frames, window_frames=3, stride_frames=2)
    assert [[frame["frame_id"] for frame in window] for window in windows] == [
        ["f0", "f1", "f2"], ["f2", "f3", "f4"], ["f4"],
    ]
    with pytest.raises(HarnessError, match="caption_window_frames"):
        caption_windows(frames, window_frames=0, stride_frames=1)
    with pytest.raises(HarnessError, match="caption_stride_frames"):
        caption_windows(frames, window_frames=1, stride_frames=0)


def test_dense_events_are_strict_and_use_only_window_timestamps():
    text = json.dumps({"events": [
        {"start": 0, "end": 1, "caption": " A person opens a door. "},
        {"start": 2, "end": 2, "caption": "The person looks inside."},
    ]})
    assert parse_dense_events(text, [0.0, 1.0, 2.0]) == [
        {"start": 0.0, "end": 1.0, "caption": "A person opens a door."},
        {"start": 2.0, "end": 2.0, "caption": "The person looks inside."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0, "end": 2, "caption": "A person walks."},
        {"start": 0, "end": 1, "caption": "The person waves."},
    ]}), [0.0, 1.0, 2.0])
    with pytest.raises(HarnessError, match="outside its window"):
        parse_dense_events(
            '{"events":[{"start":0,"end":1.5,"caption":"Invented precision."}]}',
            [0.0, 1.0, 2.0],
        )
    with pytest.raises(HarnessError, match="chronological order"):
        parse_dense_events(json.dumps({"events": [
            {"start": 2, "end": 2, "caption": "Later."},
            {"start": 1, "end": 1, "caption": "Earlier."},
        ]}), [0.0, 1.0, 2.0])
    assert parse_dense_events('{"events":[],"summary":"extra"}', [0.0]) == []
    assert parse_dense_events(
        json.dumps([{"start": 0, "end": 0, "caption": "Bare list."}]), [0.0]
    ) == [{"start": 0.0, "end": 0.0, "caption": "Bare list."}]
    with pytest.raises(HarnessError, match=r"keys=\[summary\]"):
        parse_dense_events('{"summary":"no events"}', [0.0])


def test_dense_events_map_frame_indices_onto_shifted_windows():
    timeline = [115.0, 116.0, 117.0, 118.0]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0, "end": 1, "caption": "A person enters."},
        {"start": 2, "end": 3, "caption": "The person stops."},
    ]}), timeline) == [
        {"start": 115.0, "end": 116.0, "caption": "A person enters."},
        {"start": 117.0, "end": 118.0, "caption": "The person stops."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 115, "end": 118, "caption": "Absolute times."},
    ]}), timeline) == [
        {"start": 115.0, "end": 118.0, "caption": "Absolute times."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0, "end": 4, "caption": "Exclusive end as frame count."},
    ]}), timeline) == [
        {"start": 115.0, "end": 118.0, "caption": "Exclusive end as frame count."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0.0, "end": 4.0, "caption": "Covers the four-frame window."},
    ]}), [0.0, 1.0, 2.0, 3.0]) == [
        {"start": 0.0, "end": 3.0, "caption": "Covers the four-frame window."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 4, "end": 4, "caption": "One-based last frame."},
    ]}), [0.0, 1.0, 2.0, 3.0]) == [
        {"start": 3.0, "end": 3.0, "caption": "One-based last frame."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 115, "end": 119, "caption": "Exclusive time past last sample."},
    ]}), timeline) == [
        {"start": 115.0, "end": 118.0, "caption": "Exclusive time past last sample."},
    ]
    with pytest.raises(HarnessError, match=r"window=\[115.0, 116.0, 117.0, 118.0\]"):
        parse_dense_events(
            '{"events":[{"start":114,"end":115,"caption":"Before the window."}]}',
            timeline,
        )


def test_dense_events_accept_markdown_json_fence_but_not_prose():
    payload = {"events": [{"start": 0, "end": 1, "caption": "A door opens."}]}
    inner = json.dumps(payload)
    expected = [{"start": 0.0, "end": 1.0, "caption": "A door opens."}]
    timeline = [0.0, 1.0]
    for text in (
        f"```json\n{inner}\n```",
        f"```JSON\n{inner}\n```",
        f"```\n{inner}\n```",
        f"```json\n{inner}```",
        f"\n```json\n{inner}\n```\n",
    ):
        assert parse_dense_events(text, timeline) == expected
    with pytest.raises(HarnessError, match="Invalid JSON"):
        parse_dense_events(f"Here is the JSON:\n```json\n{inner}\n```", timeline)
    with pytest.raises(HarnessError):
        parse_json(f"```json\n{inner}\n```")


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


def test_ingest_captions_overlapping_multi_frame_windows(cfg, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    output = tmp_path / "wiki" / "clip"
    cfg["ingest"].update({
        "sample_interval_sec": 5.0,
        "caption_window_frames": 3,
        "caption_stride_frames": 2,
    })
    monkeypatch.setattr("harness.ingest.probe_durations", lambda _path: (21.0, 21.0))
    monkeypatch.setattr(
        "harness.ingest.extract_frame",
        lambda _video, _timestamp, path, _cfg: path.write_bytes(b"jpeg"),
    )
    monkeypatch.setattr("harness.ingest.media_command", lambda _cmd: "ffmpeg version test")

    class Captioner:
        def __init__(self):
            self.windows = []

        def caption(self, paths):
            self.windows.append([path.name for path in paths])
            return "A chronological sequence."

    captioner = Captioner()
    metadata = ingest_video(video, "clip", output, cfg, captioner=captioner)
    assert captioner.windows == [
        ["000001.jpg", "000002.jpg", "000003.jpg"],
        ["000003.jpg", "000004.jpg", "000005.jpg"],
        ["000005.jpg"],
    ]
    entries = read_jsonl(output / "frames.jsonl")
    assert [(row["start_timestamp"], row["end_timestamp"]) for row in entries] == [
        (0.0, 10.0), (10.0, 20.0), (20.0, 20.0),
    ]
    assert metadata["ingest_config"]["caption_window_frames"] == 3
    wiki = (output / "wiki.md").read_text()
    assert "Number of caption windows: 3" in wiki
    assert "Caption stride: 2 sampled frames" in wiki


def test_ingest_writes_validated_dense_events(cfg, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    output = tmp_path / "wiki" / "clip"
    cfg["ingest"].update({
        "sample_interval_sec": 5.0,
        "caption_mode": "dense",
        "caption_window_frames": 3,
        "caption_stride_frames": 2,
    })
    cfg["ingest"]["vlm"]["prompt"] = "{{FRAME_TIMESTAMPS}}"
    monkeypatch.setattr("harness.ingest.probe_durations", lambda _path: (21.0, 21.0))
    monkeypatch.setattr(
        "harness.ingest.extract_frame",
        lambda _video, _timestamp, path, _cfg: path.write_bytes(b"jpeg"),
    )
    monkeypatch.setattr("harness.ingest.media_command", lambda _cmd: "ffmpeg version test")

    class Captioner:
        def __init__(self):
            self.windows = []

        def caption(self, paths, *, timestamps):
            self.windows.append(([path.name for path in paths], timestamps))
            if len(timestamps) == 1:
                return '{"events":[]}'
            return json.dumps({"events": [{
                "start": timestamps[0],
                "end": timestamps[1],
                "caption": "A visible action changes.",
            }]})

    captioner = Captioner()
    ingest_video(video, "clip", output, cfg, captioner=captioner)
    assert captioner.windows == [
        (["000001.jpg", "000002.jpg", "000003.jpg"], [0.0, 5.0, 10.0]),
        (["000003.jpg", "000004.jpg", "000005.jpg"], [10.0, 15.0, 20.0]),
        (["000005.jpg"], [20.0]),
    ]
    entries = read_jsonl(output / "frames.jsonl")
    assert set(entries[0]) == {
        "window_id", "start_timestamp", "end_timestamp", "frames", "events",
    }
    assert entries[0]["events"] == [{
        "start": 0.0, "end": 5.0, "caption": "A visible action changes.",
    }]
    assert entries[-1]["events"] == []
    wiki = (output / "wiki.md").read_text()
    assert "Caption mode: dense" in wiki
    assert "Number of dense events: 2" in wiki
    assert "#### Event 0.0s–5.0s" in wiki


def test_default_window_settings_match_legacy_ingest_hash(cfg):
    legacy = {**cfg["ingest"]}
    legacy.pop("caption_mode")
    legacy.pop("caption_window_frames")
    legacy.pop("caption_stride_frames")
    assert ingest_content_hash({"ingest": legacy}) == ingest_content_hash(cfg)


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
