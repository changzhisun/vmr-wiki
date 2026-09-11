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
from harness.ingest import (caption_windows, compact_dense_events, dense_events,
                            dense_target_timestamps, ingest_video, probe_duration, probe_durations,
                            parse_dense_events, render_wiki, sample_times)
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


def test_default_config_uses_bidirectional_captions():
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    ingest = config["ingest"]
    assert ingest["sample_interval_sec"] == 1.0
    assert ingest["caption_mode"] == "bidirectional"
    assert ingest["caption_window_frames"] == 5
    assert ingest["caption_stride_frames"] == 1
    assert ingest["caption_processing_version"] == 4
    assert ingest["vlm"]["max_tokens"] == 8192
    assert ingest["bidirectional"]["max_frames_per_call"] == 100
    assert ingest["pipeline_version"] == "vlm_bidirectional_v1"


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

    raw = yaml.safe_load(source.read_text())
    raw["ingest"]["caption_mode"] = "dense"
    raw["ingest"]["vlm"]["prompt"] = "Timeline {{FRAME_TIMESTAMPS}}"
    raw["ingest"]["caption_stride_frames"] = 3
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="must not exceed half"):
        load_config(path)

    raw = yaml.safe_load(source.read_text())
    raw["ingest"]["caption_processing_version"] = 3
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="unsupported"):
        load_config(path)


def test_caption_windows_use_full_anchored_tail_without_redundant_short_windows():
    frames = [{"frame_id": f"f{i}"} for i in range(5)]
    windows = caption_windows(frames, window_frames=3, stride_frames=2)
    assert [[frame["frame_id"] for frame in window] for window in windows] == [
        ["f0", "f1", "f2"], ["f2", "f3", "f4"],
    ]
    frames.append({"frame_id": "f5"})
    assert [[frame["frame_id"] for frame in window]
            for window in caption_windows(frames, 3, 2)] == [
        ["f0", "f1", "f2"], ["f2", "f3", "f4"], ["f3", "f4", "f5"],
    ]
    assert caption_windows(frames[:2], 5, 1) == [frames[:2]]
    with pytest.raises(HarnessError, match="caption_window_frames"):
        caption_windows(frames, window_frames=0, stride_frames=1)
    with pytest.raises(HarnessError, match="caption_stride_frames"):
        caption_windows(frames, window_frames=1, stride_frames=0)


def test_dense_targets_focus_centers_and_extend_only_video_boundaries():
    frames = [{"timestamp": float(i)} for i in range(8)]
    windows = caption_windows(frames, 5, 1)
    assert dense_target_timestamps(windows) == [
        [0.0, 1.0, 2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0, 7.0],
    ]
    singleton = [[{"timestamp": 0.0}], [{"timestamp": 1.0}]]
    assert dense_target_timestamps(singleton) == [[0.0], [1.0]]
    separated = [[{"timestamp": 0.0}, {"timestamp": 1.0}],
                 [{"timestamp": 3.0}, {"timestamp": 4.0}]]
    with pytest.raises(HarnessError, match="stride is too large"):
        dense_target_timestamps(separated)

    for window_size in range(2, 10):
        for stride in range(1, window_size // 2 + 1):
            for count in range(1, 25):
                source = [{"timestamp": float(i)} for i in range(count)]
                generated = caption_windows(source, window_size, stride)
                targets = dense_target_timestamps(generated)
                assert all(len(window) == min(count, window_size) for window in generated)
                assert targets[0][0] == 0.0
                assert targets[-1][-1] == count - 1
                assert all(target for target in targets)
                assert all(set(target) <= {frame["timestamp"] for frame in window}
                           for window, target in zip(generated, targets))
                assert all(left[-1] == right[0]
                           for left, right in zip(targets, targets[1:]))


def test_compact_dense_events_only_deduplicates_identical_ranges_and_sorts_timeline():
    entries = [
        {"events": [{"start": 10.0, "end": 11.0, "kind": "state", "caption": "A door is open."}]},
        {"events": [{"start": 5.0, "end": 6.0, "kind": "action", "caption": "A person walks."},
                    {"start": 10.0, "end": 11.0, "kind": "state", "caption": " a door  is OPEN. "},
                    {"start": 11.0, "end": 12.0, "kind": "state", "caption": " a door  is OPEN. "}]},
        {"events": [{"start": 12.0, "end": 13.0, "kind": "transition", "caption": "A door is open."}]},
    ]
    assert compact_dense_events(entries) == [
        {"start": 5.0, "end": 6.0, "kind": "action", "caption": "A person walks."},
        {"start": 10.0, "end": 11.0, "kind": "state", "caption": "A door is open."},
        {"start": 11.0, "end": 12.0, "kind": "state", "caption": " a door  is OPEN. "},
        {"start": 12.0, "end": 13.0, "kind": "transition", "caption": "A door is open."},
    ]


def test_dense_wiki_groups_compact_events_into_thirty_second_buckets():
    entries = [{"events": [
        {"start": 29.0, "end": 31.0, "kind": "action", "caption": "A person walks."},
        {"start": 30.0, "end": 30.0, "kind": "state", "caption": "A door is open."},
        {"start": 60.0, "end": 61.0, "kind": "transition", "caption": "A light turns on."},
    ]}]
    wiki = render_wiki(65.0, 1.0, entries, sampled_frame_count=65,
                       window_frames=5, stride_frames=1, caption_mode="dense")
    assert "### Segments overlapping 0s–30s" in wiki
    assert "### Segments overlapping 30s–60s" in wiki
    assert "### Segments overlapping 60s–65.0s" in wiki
    assert wiki.count("`29.0s–31.0s` **action** — A person walks.") == 2
    middle_bucket = wiki.split("### Segments overlapping 30s–60s", 1)[1].split(
        "### Segments overlapping 60s–65.0s", 1
    )[0]
    assert "`29.0s–31.0s`" in middle_bucket
    assert "`30.0s`" in middle_bucket


class RepairCaptioner:
    """Answer out of window until the rejection has been fed back ``after`` times."""

    def __init__(self, after: int):
        self.after = after
        self.corrections = []

    def caption(self, images, *, timestamps=None, target_timestamps=None, correction=None):
        self.corrections.append(correction)
        if len(self.corrections) > self.after:
            return json.dumps({"events": [
                {"start": timestamps[0], "end": timestamps[-1], "kind": "action",
                 "caption": "In window."}]})
        return json.dumps({"events": [
            {"start": timestamps[-1], "end": timestamps[-1] + 3, "kind": "action",
             "caption": "Past the window."}]})


def test_out_of_window_answer_is_repaired_within_budget():
    captioner = RepairCaptioner(after=1)
    assert dense_events(captioner, [Path("f.jpg")], [8.0, 9.0, 10.0, 11.0], 2) == [
        {"start": 8.0, "end": 11.0, "kind": "action", "caption": "In window."}]
    # The first attempt carries no correction; the repair carries the rejection.
    assert captioner.corrections[0] is None
    assert "outside its window" in captioner.corrections[1]


def test_frame_index_target_is_only_a_focus_and_events_may_cross_it():
    seen = []

    class Captioner:
        def caption(self, images, *, timestamps, target_timestamps, correction=None):
            seen.append((timestamps, target_timestamps))
            return json.dumps({"events": [{
                "start": 0, "end": 4, "kind": "action", "caption": "An action continues.",
            }]})

    result = dense_events(Captioner(), [Path("f.jpg")] * 5,
                          [10.0, 11.0, 12.0, 13.0, 14.0], 0,
                          timestamp_mode="frame_index", target_timestamps=[12.0, 13.0])
    assert seen == [([0, 1, 2, 3, 4], [2, 3])]
    assert result == [{"start": 10.0, "end": 14.0, "kind": "action",
                       "caption": "An action continues."}]


def test_repair_budget_is_bounded_and_then_fails():
    captioner = RepairCaptioner(after=99)
    with pytest.raises(HarnessError, match="outside its window"):
        dense_events(captioner, [Path("f.jpg")], [8.0, 9.0, 10.0, 11.0], 2)
    assert len(captioner.corrections) == 3

    immediate = RepairCaptioner(after=99)
    with pytest.raises(HarnessError, match="outside its window"):
        dense_events(immediate, [Path("f.jpg")], [8.0, 9.0, 10.0, 11.0], 0)
    assert immediate.corrections == [None]


def test_dense_events_are_strict_and_use_only_window_timestamps():
    text = json.dumps({"events": [
        {"start": 0, "end": 1, "kind": " Action ", "caption": " A person opens a door. "},
        {"start": 2, "end": 2, "kind": "state", "caption": "The person looks inside."},
    ]})
    assert parse_dense_events(text, [0.0, 1.0, 2.0]) == [
        {"start": 0.0, "end": 1.0, "kind": "action", "caption": "A person opens a door."},
        {"start": 2.0, "end": 2.0, "kind": "state", "caption": "The person looks inside."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0, "end": 2, "kind": "action", "caption": "A person walks."},
        {"start": 0, "end": 1, "kind": "action", "caption": "The person waves."},
    ]}), [0.0, 1.0, 2.0])
    with pytest.raises(HarnessError, match="outside its window"):
        parse_dense_events(
            '{"events":[{"start":0,"end":1.5,"kind":"action","caption":"Invented precision."}]}',
            [0.0, 1.0, 2.0],
        )
    with pytest.raises(HarnessError, match="chronological order"):
        parse_dense_events(json.dumps({"events": [
            {"start": 2, "end": 2, "kind": "state", "caption": "Later."},
            {"start": 1, "end": 1, "kind": "state", "caption": "Earlier."},
        ]}), [0.0, 1.0, 2.0])
    assert parse_dense_events('{"events":[],"summary":"extra"}', [0.0]) == []
    assert parse_dense_events(
        json.dumps([{"start": 0, "end": 0, "kind": "state", "caption": "Bare list."}]), [0.0]
    ) == [{"start": 0.0, "end": 0.0, "kind": "state", "caption": "Bare list."}]
    with pytest.raises(HarnessError, match="kind must be"):
        parse_dense_events(
            '{"events":[{"start":0,"end":0,"kind":"scene","caption":"Invalid."}]}',
            [0.0],
        )
    with pytest.raises(HarnessError, match="kind must be"):
        parse_dense_events(
            '{"events":[{"start":0,"end":0,"kind":1,"caption":"Invalid."}]}',
            [0.0],
        )
    with pytest.raises(HarnessError, match=r"keys=\[summary\]"):
        parse_dense_events('{"summary":"no events"}', [0.0])


def test_dense_events_map_frame_indices_onto_shifted_windows():
    timeline = [115.0, 116.0, 117.0, 118.0]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 0, "end": 1, "kind": "transition", "caption": "A person enters."},
        {"start": 2, "end": 3, "kind": "state", "caption": "The person stops."},
    ]}), timeline, "frame_index") == [
        {"start": 115.0, "end": 116.0, "kind": "transition", "caption": "A person enters."},
        {"start": 117.0, "end": 118.0, "kind": "state", "caption": "The person stops."},
    ]
    assert parse_dense_events(json.dumps({"events": [
        {"start": 115, "end": 118, "kind": "action", "caption": "Absolute times."},
    ]}), timeline) == [
        {"start": 115.0, "end": 118.0, "kind": "action", "caption": "Absolute times."},
    ]
    for start, end, mode in [(0, 4, "frame_index"), (115, 119, "absolute_seconds"),
                             (0, 2, "absolute_seconds")]:
        with pytest.raises(HarnessError, match="outside its window"):
            parse_dense_events(json.dumps({"events": [
                {"start": start, "end": end, "kind": "action", "caption": "Invalid coordinates."},
            ]}), timeline, mode)
    # Reproduction: do not mix absolute seconds and indices within one event.
    text = '{"events":[{"start":0,"end":2,"kind":"action","caption":"An action."}]}'
    with pytest.raises(HarnessError, match="outside its window"):
        parse_dense_events(text, [2.0, 3.0, 4.0, 5.0])
    assert parse_dense_events(text, [2.0, 3.0, 4.0, 5.0], "frame_index") == [
        {"start": 2.0, "end": 4.0, "kind": "action", "caption": "An action."}]
    with pytest.raises(HarnessError, match=r"window=\[115.0, 116.0, 117.0, 118.0\]"):
        parse_dense_events(
            '{"events":[{"start":114,"end":115,"kind":"action","caption":"Before the window."}]}',
            timeline,
        )


def test_dense_events_accept_markdown_json_fence_but_not_prose():
    payload = {"events": [{"start": 0, "end": 1, "kind": "transition",
                            "caption": "A door opens."}]}
    inner = json.dumps(payload)
    expected = [{"start": 0.0, "end": 1.0, "kind": "transition",
                 "caption": "A door opens."}]
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
    ]
    entries = read_jsonl(output / "frames.jsonl")
    assert [(row["start_timestamp"], row["end_timestamp"]) for row in entries] == [
        (0.0, 10.0), (10.0, 20.0),
    ]
    assert metadata["ingest_config"]["caption_window_frames"] == 3
    wiki = (output / "wiki.md").read_text()
    assert "Number of caption windows: 2" in wiki
    assert "Caption stride: 2 sampled frames" in wiki


def test_ingest_writes_validated_dense_events(cfg, monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    output = tmp_path / "wiki" / "clip"
    cfg["ingest"].update({
        "sample_interval_sec": 5.0,
        "caption_mode": "dense",
        "caption_window_frames": 3,
        "caption_stride_frames": 1,
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

        def caption(self, paths, *, timestamps, target_timestamps):
            self.windows.append(([path.name for path in paths], timestamps, target_timestamps))
            return json.dumps({"events": [{
                "start": timestamps[0],
                "end": timestamps[-1],
                "kind": "action",
                "caption": "A visible action changes.",
            }]})

    captioner = Captioner()
    ingest_video(video, "clip", output, cfg, captioner=captioner)
    assert captioner.windows == [
        (["000001.jpg", "000002.jpg", "000003.jpg"],
         [0.0, 5.0, 10.0], [0.0, 5.0, 10.0]),
        (["000002.jpg", "000003.jpg", "000004.jpg"],
         [5.0, 10.0, 15.0], [10.0, 15.0]),
        (["000003.jpg", "000004.jpg", "000005.jpg"],
         [10.0, 15.0, 20.0], [15.0, 20.0]),
    ]
    entries = read_jsonl(output / "frames.jsonl")
    assert set(entries[0]) == {
        "window_id", "start_timestamp", "end_timestamp", "target_start_timestamp",
        "target_end_timestamp", "frames", "events",
    }
    assert entries[0]["events"] == [{
        "start": 0.0, "end": 10.0, "kind": "action",
        "caption": "A visible action changes.",
    }]
    # The target is a prompt focus only; full-window event boundaries remain valid.
    assert entries[1]["target_start_timestamp"] == 10.0
    assert entries[1]["events"][0]["start"] == 5.0
    wiki = (output / "wiki.md").read_text()
    assert "Caption mode: dense" in wiki
    assert "Number of dense events: 3" in wiki
    assert "Number of displayed segments: 3" in wiki
    assert "`0.0s–10.0s` **action** — A visible action changes." in wiki
    assert "`5.0s–15.0s` **action** — A visible action changes." in wiki
    assert "`10.0s–20.0s` **action** — A visible action changes." in wiki
    assert "`0.0s–20.0s`" not in wiki
    assert "Frames:" not in wiki and "### Window" not in wiki


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

    def fake_ingest(vid, video, path, output, cfg, *, captioner, runner, cancel_event):
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


@pytest.mark.parametrize("jobs", [1, 2])
def test_batch_video_failure_does_not_skip_healthy_videos(cfg, tmp_path, monkeypatch, jobs):
    from harness.ingest_all import _run_sequential
    from harness.common import read_json
    parent = tmp_path / "batch" / "videos"
    tasks = [(vid, {"duration": 3}, tmp_path / f"{vid}.mp4", parent / vid) for vid in ("bad", "good")]
    done = []
    class Bar:
        def update(self):
            pass
    def fake_ingest(path, vid, output, config, **kwargs):
        if vid == "bad":
            raise HarnessError("Invalid model response")
        done.append(vid)
        return {"duration": 3}
    monkeypatch.setattr("harness.ingest_all.ingest_video", fake_ingest)
    with pytest.raises(HarnessError, match="1 of 2 video"):
        if jobs == 1:
            _run_sequential(tasks, cfg, None, Bar(), threading.Event())
        else:
            _run_parallel(tasks, cfg, None, jobs, Bar(), threading.Event())
    assert done == ["good"]
    report = read_json(parent.parent / ".ingest-failures/bad.json")
    assert report["status"] == "failed" and report["error"] == "Invalid model response"


@pytest.mark.parametrize("jobs", [1, 2])
def test_fatal_configuration_stops_batch_before_more_submissions(cfg, tmp_path, monkeypatch, jobs):
    from harness.ingest_all import _run_sequential
    from harness.bidirectional_io import RequestFailed
    from harness.vlm_transport import FatalVLMError
    calls = []
    stopped = threading.Event()
    tasks = [(str(i), {"duration": 3}, tmp_path / f"{i}.mp4", tmp_path / f"out/{i}") for i in range(20)]
    class Bar:
        def update(self):
            pass
    def fake_ingest(path, vid, output, config, **kwargs):
        calls.append(vid)
        if vid != '0':
            assert kwargs['cancel_event'].wait(2)
            stopped.set()
            raise HarnessError("VLM request cancelled")
        try:
            raise FatalVLMError("VLM request failed with HTTP 401")
        except FatalVLMError as exc:
            raise RequestFailed(str(exc)) from exc
    monkeypatch.setattr("harness.ingest_all.ingest_video", fake_ingest)
    cancel = threading.Event()
    with pytest.raises(HarnessError, match="circuit opened.*HTTP 401"):
        if jobs == 1:
            _run_sequential(tasks, cfg, None, Bar(), cancel)
        else:
            _run_parallel(tasks, cfg, None, jobs, Bar(), cancel)
    assert cancel.is_set() and len(calls) <= jobs
    if len(calls) > 1:
        assert stopped.is_set()
    reports = list((tmp_path / '.ingest-failures').glob('*.json'))
    assert [p.name for p in reports] == ['0.json']


def test_consecutive_failure_circuit_resets_after_success(cfg, tmp_path, monkeypatch):
    from harness.ingest_all import _run_sequential
    cfg['ingest']['consecutive_failure_limit'] = 2
    calls = []
    tasks = [(str(i), {}, tmp_path / str(i), tmp_path / f"out/{i}") for i in range(10)]
    class Bar:
        def update(self):
            pass
    def fake_one(vid, *args, **kwargs):
        calls.append(vid)
        if vid != '1':
            raise HarnessError("malformed response")
        return {}, False
    monkeypatch.setattr("harness.ingest_all._ingest_one", fake_one)
    with pytest.raises(HarnessError, match="2 consecutive videos failed"):
        _run_sequential(tasks, cfg, None, Bar(), threading.Event())
    assert calls == ['0', '1', '2', '3']


def test_cancelled_video_does_not_create_failure_report(cfg, tmp_path, monkeypatch):
    from harness.ingest_all import _ingest_one
    cancelled = threading.Event()
    def fake_ingest(*args, **kwargs):
        cancelled.set()
        raise HarnessError("VLM request cancelled")
    monkeypatch.setattr("harness.ingest_all.ingest_video", fake_ingest)
    with pytest.raises(HarnessError, match="cancelled"):
        _ingest_one("v", {"duration": 3}, tmp_path / 'v.mp4', tmp_path / 'videos/v', cfg, cancel_event=cancelled)
    assert not (tmp_path / '.ingest-failures/v.json').exists()
