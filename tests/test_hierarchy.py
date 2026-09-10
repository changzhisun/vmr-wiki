import copy
import json
from pathlib import Path
import threading
import subprocess

import pytest
import yaml

from harness.common import HarnessError, ingest_content_hash, read_json, read_jsonl
from harness.config import load_config
from harness.freeze import freeze_wiki, verify_wiki
from harness.hierarchy import (SCHEMA, flatten, parse_merge, parse_split, validate_tree)
from harness.hierarchy_config import settings
from harness.ingest import ingest_video
from harness.sampling import analyze_changes, context_bounds, frame_timestamps, mixed_times, uniform_times
from harness.workspace import query_workspace


def hierarchical(cfg, **options):
    cfg["ingest"]["caption_mode"] = "hierarchical"
    cfg["ingest"]["hierarchy"] = settings({"hierarchy": {
        "max_frames": 8, "min_segment_sec": 0.1, "min_sample_interval_sec": 0.1, **options}})
    return cfg


def segment(start, end, title="Visible activity"):
    return {**copy.deepcopy(SCHEMA), "start": start, "end": end, "title": title}


class TreeCaptioner:
    def __init__(self, *, fail_at=None, terminal=False, merge=True, repair=False):
        self.calls = []
        self.fail_at, self.terminal, self.merge, self.repair = fail_at, terminal, merge, repair

    def caption(self, images, *, prompt_override, correction=None):
        spec = json.JSONDecoder().raw_decode(prompt_override.split("Request:\n", 1)[1])[0]
        assert 1 <= len(images) <= 100
        assert all(image.read_bytes().startswith(b"\xff\xd8") for image in images)
        assert spec["timestamps"] == sorted(set(spec["timestamps"]))
        assert len(spec["timestamps"]) == len(images)
        self.calls.append(spec)
        if self.fail_at == len(self.calls):
            raise HarnessError("fixture transport interruption")
        if self.repair and correction is None:
            return '{"wrong":true}'
        if spec["operation"] == "split":
            if self.terminal and spec["level"] != "chapter":
                return '{"terminal":true,"nodes":[]}'
            start, end = spec["target"]
            mid = (start + end) / 2
            return json.dumps({"terminal": False, "nodes": [segment(start, mid), segment(mid, end)]})
        nodes = spec["nodes"]
        groups = [nodes] if self.merge else [[node] for node in nodes]
        return json.dumps({"groups": [
            {"node_ids": [node["node_id"] for node in group],
             **{key: SCHEMA[key] for key in SCHEMA.keys() - {"start", "end"}}}
            for group in groups]})


def video_and_output(cfg):
    video = Path(cfg["paths"]["datasets"]).parent / "videos/video.mp4"
    output = Path(cfg["paths"]["wiki"]) / "qvhighlights/videos/video"
    return video, output


def test_uniform_and_mixed_sampling_are_bounded_and_cover_changes():
    cfg = settings({"hierarchy": {"max_frames": 20}})
    scores = [{"timestamp": 42.25, "scene": 0.9, "motion": 0.5},
              {"timestamp": 77.7, "scene": 0.0, "motion": 0.8}]
    stamps = mixed_times(0, 100, 100, scores, cfg)
    assert len(stamps) == 20 and stamps == sorted(set(stamps))
    assert 42.25 in stamps and 41.75 in stamps and 77.7 in stamps
    assert stamps[0] == 0 and stamps[-1] < 100
    assert mixed_times(0, 100, 100, scores, cfg, global_scan=True) == uniform_times(0, 100, 20, 100)
    for length in (0.000001, 0.1, 1, 100, 10000):
        stamps = mixed_times(0, length, length, [], settings({}))
        assert 1 <= len(stamps) <= 100
        assert all(0 <= stamp < length for stamp in stamps)


def test_context_overlap_does_not_change_semantic_partition():
    nodes = [segment(0, 10), segment(10, 30), segment(30, 40)]
    contexts = [context_bounds(nodes, i, 0, 40, 0.15) for i in range(3)]
    assert contexts == [(0, 10.75), (9.25, 30.75), (29.25, 40)]
    assert contexts[0][1] - contexts[1][0] == 1.5
    assert [n["start"] for n in nodes] == [0, 10, 30]


@pytest.mark.parametrize("change", [
    lambda nodes: nodes[0].update(start=-1),
    lambda nodes: nodes[0].update(end=6),
    lambda nodes: nodes[1].update(start=6),
    lambda nodes: nodes[1].update(end=9),
    lambda nodes: nodes[0].update(confidence=True),
    lambda nodes: nodes[0].update(confidence=1.1),
    lambda nodes: nodes[0].update(actors="person"),
    lambda nodes: nodes[0].update(title=" "),
    lambda nodes: nodes[0].update(node_id="model-controlled"),
])
def test_split_contract_rejects_invalid_temporal_and_semantic_fields(change):
    nodes = [segment(0, 5), segment(5, 10)]
    change(nodes)
    with pytest.raises(HarnessError):
        parse_split(json.dumps({"terminal": False, "nodes": nodes}), 0, 10, 12)


def test_split_requires_global_coverage_but_allows_semantic_stop():
    terminal = '{"terminal":true,"nodes":[]}'
    assert parse_split(terminal, 0, 10, 12) == []
    with pytest.raises(HarnessError):
        parse_split(terminal, 0, 10, 12, global_scan=True)
    with pytest.raises(HarnessError):
        parse_split('{"terminal":false,"nodes":[]}', 0, 10, 12)


@pytest.mark.parametrize("ids", [["b", "a"], ["a", "a"], ["a"], ["a", "other"], []])
def test_merge_rejects_nonadjacent_duplicate_missing_or_unknown_ids(ids):
    siblings = [{"node_id": "a"}, {"node_id": "b"}]
    group = {key: SCHEMA[key] for key in SCHEMA.keys() - {"start", "end"}}
    with pytest.raises(HarnessError):
        parse_merge(json.dumps({"groups": [{"node_ids": ids, **group}]}), siblings)


def test_recursive_ingest_merges_bottom_up_preserves_evidence_and_freezes(prepared):
    cfg, _ = prepared
    hierarchical(cfg)
    video, output = video_and_output(cfg)
    captioner = TreeCaptioner()
    meta = ingest_video(video, "video", output, cfg, captioner=captioner)
    rows = read_jsonl(output / "nodes.jsonl")
    observations = read_jsonl(output / "observations.jsonl")
    frames = read_jsonl(output / "frames.jsonl")
    assert {r["level"] for r in rows} == {"chapter", "scene", "event", "action"}
    validate_tree(rows, 3.0)
    assert len([r for r in rows if r["parent_id"] is None]) == 1
    assert len(observations) > len(rows)
    by_id = {r["node_id"]: r for r in observations}
    assert all(row["parent_id"] in by_id for row in observations if row["level"] != "chapter")
    assert all(source in by_id for row in rows for source in row.get("source_node_ids", []))
    frame_ids = {r["frame_id"] for r in frames}
    assert all(set(r["evidence_frame_ids"]) <= frame_ids for r in rows + observations)
    assert all(len(call["timestamps"]) <= 8 for call in captioner.calls)
    assert captioner.calls[0]["level"] == "chapter"
    assert captioner.calls[-1]["operation"] == "merge"
    assert captioner.calls[-1]["level"] == "chapter"
    assert any(call["context"] != call["target"] for call in captioner.calls
               if call["operation"] == "split")
    assert meta["telemetry"]["node_count"] == len(rows)
    assert "## Chapters" in (output / "wiki.md").read_text()
    freeze_wiki(output)
    seal = verify_wiki(output)
    assert {"nodes.jsonl", "observations.jsonl", "sampling.jsonl"} <= seal["files"].keys()
    query = {"query_id": "q1", "video_id": "video", "split": "train", "query": "Find activity"}
    task = {**query, "max_predictions": 5}
    with query_workspace(query, task, output, {"AGENTS.md": "fixture", "query_prompt.md": "fixture"},
                         Path(cfg["paths"]["runs"])) as workspace:
        assert set(p.name for p in (workspace / "wiki").iterdir()) == {
            "wiki.md", "frames.jsonl", "nodes.jsonl", "observations.jsonl", "frames"}
        assert read_jsonl(workspace / "wiki/nodes.jsonl") == rows
    count = len(captioner.calls)
    assert ingest_video(video, "video", output, cfg, captioner=captioner) == meta
    assert len(captioner.calls) == count


def test_hierarchy_resume_reuses_requests_and_retains_failed_audit(prepared):
    cfg, _ = prepared
    hierarchical(cfg, max_depth=2)
    video, output = video_and_output(cfg)
    first = TreeCaptioner(fail_at=3)
    with pytest.raises(HarnessError, match="interruption"):
        ingest_video(video, "video", output, cfg, captioner=first)
    assert not output.exists()
    second = TreeCaptioner()
    meta = ingest_video(video, "video", output, cfg, captioner=second)
    assert second.calls[0] == first.calls[2]
    assert meta["telemetry"]["reused_windows"] == 2
    assert meta["telemetry"]["reused_frames"] > 0
    assert meta["telemetry"]["reused_analysis"]
    assert meta["telemetry"]["rejected_attempts"] == 1
    assert not list(output.parent.parent.glob('.ingest-checkpoints/video/*/identity.json'))


@pytest.mark.parametrize("options,terminal,reason", [
    ({"min_segment_sec": 10}, False, "short_interval"),
    ({"max_depth": 1}, False, "max_depth"),
    ({}, True, "semantically_indivisible"),
])
def test_hierarchy_stops_at_duration_depth_or_semantic_leaf(prepared, options, terminal, reason):
    cfg, _ = prepared
    hierarchical(cfg, merge_adjacent=False, **options)
    video, output = video_and_output(cfg)
    captioner = TreeCaptioner(terminal=terminal)
    ingest_video(video, "video", output, cfg, captioner=captioner)
    rows = read_jsonl(output / "nodes.jsonl")
    assert len(rows) == 2
    assert all(row["stop_reason"] == reason for row in rows)


def test_schema_repair_and_request_budget(prepared):
    cfg, _ = prepared
    hierarchical(cfg, max_depth=1, merge_adjacent=False, max_requests=1)
    video, output = video_and_output(cfg)
    meta = ingest_video(video, "video", output, cfg, captioner=TreeCaptioner(repair=True))
    assert meta["telemetry"]["rejected_attempts"] == 1
    assert meta["telemetry"]["caption_attempts"] == 2
    cfg["ingest"]["hierarchy"]["max_depth"] = 4
    with pytest.raises(HarnessError, match="max_requests"):
        ingest_video(video, "video2", output.with_name("video2"), cfg, captioner=TreeCaptioner())
    assert not output.with_name("video2").exists()


def test_change_analysis_can_cancel_and_detects_static_scene(prepared):
    cfg, _ = prepared
    video, _ = video_and_output(cfg)
    scores = analyze_changes(video, 3, settings({}), lambda: None)
    assert scores and all(row["scene"] == row["motion"] == 0 for row in scores)
    def cancel():
        raise HarnessError("cancelled")
    with pytest.raises(HarnessError, match="cancelled"):
        analyze_changes(video, 3, settings({}), cancel)


def test_visual_cut_detection_and_source_timestamps(prepared, tmp_path):
    # Real decode test: a hard black-to-white cut should score in both signals.
    video = tmp_path / "cut.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=black:s=80x60:r=10:d=1",
                    "-f", "lavfi", "-i", "color=white:s=80x60:r=10:d=1", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "mpeg4", "-y", str(video)], check=True)
    stamps = frame_timestamps(video, 2)
    assert stamps == pytest.approx([i / 10 for i in range(20)])
    scores = analyze_changes(video, 2, settings({}), lambda: None)
    peak = max(scores, key=lambda row: row["scene"])
    assert abs(peak["timestamp"] - 1) <= 0.5
    assert peak["scene"] > 0.8 and peak["motion"] > 0.8


def test_node_budget_does_not_publish_partial_tree(prepared):
    cfg, _ = prepared
    hierarchical(cfg, max_nodes=1, max_depth=1)
    video, output = video_and_output(cfg)
    with pytest.raises(HarnessError, match="max_nodes"):
        ingest_video(video, "video", output, cfg, captioner=TreeCaptioner())
    assert not output.exists()


def test_corrupt_hierarchy_checkpoint_rejected(prepared):
    cfg, _ = prepared
    hierarchical(cfg)
    video, output = video_and_output(cfg)
    with pytest.raises(HarnessError):
        ingest_video(video, "video", output, cfg, captioner=TreeCaptioner(fail_at=2))
    path = next(output.parent.parent.glob('.ingest-checkpoints/video/*/hierarchy/split_root.json'))
    data = read_json(path)
    data["data"]["raw_response"] = "changed"
    path.write_text(json.dumps(data))
    with pytest.raises(HarnessError, match="Checkpoint changed"):
        ingest_video(video, "video", output, cfg, captioner=TreeCaptioner())


def test_pre_cancelled_hierarchy_does_not_publish(prepared):
    cfg, _ = prepared
    hierarchical(cfg)
    video, output = video_and_output(cfg)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(HarnessError, match="cancelled"):
        ingest_video(video, "video", output, cfg, captioner=TreeCaptioner(), cancel_event=cancel)
    assert not output.exists()


@pytest.mark.parametrize("key,value", [("max_frames", 101), ("max_depth", 33),
                                        ("overlap_ratio", 0.6), ("merge_adjacent", "yes"),
                                        ("analysis_fps", 0), ("scene_threshold", 2)])
def test_hierarchy_config_rejects_invalid_settings(tmp_path, key, value):
    raw = yaml.safe_load((Path(__file__).parents[1] / "config.yaml").read_text())
    # The bidirectional pipeline owns the new settings; preserve the old
    # hierarchy tests' field names where they have a direct equivalent.
    mapping = {"max_frames": ("max_frames_per_call",), "max_depth": ("topdown", "max_depth"),
               "overlap_ratio": ("bottomup", "overlap_ratio"),
               "merge_adjacent": None, "analysis_fps": None, "scene_threshold": None}
    path_keys = mapping[key]
    if path_keys is None:
        pytest.skip("setting belongs only to the legacy hierarchical pipeline")
    if len(path_keys) == 1:
        raw["ingest"]["bidirectional"][path_keys[0]] = value
    else:
        raw["ingest"]["bidirectional"][path_keys[0]][path_keys[1]] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError):
        load_config(path)


def test_hierarchy_content_hash_and_legacy_compatibility(cfg):
    before = ingest_content_hash(cfg)
    cfg["ingest"]["hierarchy"] = {"max_frames": 50}
    assert ingest_content_hash(cfg) == before  # Legacy identity unaffected.
    hierarchical(cfg)
    before = ingest_content_hash(cfg)
    cfg["ingest"]["hierarchy"]["overlap_ratio"] = 0.2
    assert ingest_content_hash(cfg) != before
    before = ingest_content_hash(cfg)
    cfg["ingest"]["vlm"]["base_url"] = "https://different.example/v1"
    assert ingest_content_hash(cfg) == before
    cfg["ingest"]["hierarchy_processing_version"] = 99
    assert ingest_content_hash(cfg) != before


def test_final_tree_validation_rejects_broken_parent_links():
    forest = [{"node_id": "c", "level": "chapter", **segment(0, 10), "children": [
        {"node_id": "e", "level": "event", **segment(0, 10), "children": []}]}]
    with pytest.raises(HarnessError, match="parent"):
        validate_tree(flatten(forest), 10)
