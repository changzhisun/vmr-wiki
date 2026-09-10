from copy import deepcopy
import json
from pathlib import Path

import pytest

from harness.bidirectional_config import settings
from harness.bidirectional_io import batches, windows
from harness.common import HarnessError, read_jsonl
from harness.freeze import freeze_wiki, verify_wiki
from harness.ingest import ingest_video
from harness.temporal_graph import NODE_EXAMPLE, TemporalGraph, normalize_node, read_node
from harness.workspace import query_workspace


def node(start=0, end=3, **updates):
    row = deepcopy(NODE_EXAMPLE)
    row.update(start=start, end=end, boundary_uncertainty={"start": [start, start], "end": [end, end]},
               confidence=dict.fromkeys(("semantic", "boundary", "hierarchy"), "medium"))
    row.update(updates)
    return row


def config(cfg, **overrides):
    cfg["ingest"]["caption_mode"] = "bidirectional"
    cfg["ingest"]["bidirectional"] = settings({"bidirectional": overrides})
    return cfg


class BidirectionalVLM:
    """No network. Top-down deliberately misses a short event found blindly."""
    def __init__(self, fail_at=None, omit=False, malformed=False):
        self.calls = []
        self.fail_at, self.omit, self.malformed = fail_at, omit, malformed

    def complete(self, prompt, images=()):
        if prompt.startswith("Repair only JSON"):
            self.calls.append(("format_repair", {}, len(images)))
            return json.dumps({"nodes": [node(title="Coarse phase")]})
        role = prompt.split("Role: ", 1)[1].split("\n", 1)[0]
        data = json.loads(prompt.split("\nINPUT:\n", 1)[1])
        self.calls.append((role, data, len(images)))
        assert len(images) <= 100
        if images:
            assert len(data["frame_timestamps"]) == len(images)
        if len(self.calls) == self.fail_at:
            raise HarnessError("fixture interruption")
        if role == "top_down":
            if self.malformed:
                self.malformed = False
                return '{"nodes": '
            return json.dumps({"nodes": [node(data["start"], data["end"], title="Coarse phase")]})
        if role == "bottom_up":
            return json.dumps({"nodes": [node(1, 2, type="action", title="Brief tomato transfer",
                                                state_before="tomato on board", state_after="tomato in pan")]})
        if role in ("reconciliation", "bottomup_organization") and not self.omit:
            obs = next(r for r in data["records"] if r["kind"] == "observation")
            parent = next((r for r in data["records"] if r["kind"] == "node"), None)
            new = {**node(1, 2, title=obs["title"], type="action"),
                   "parent_id": parent["node_id"] if parent else None, "granularity": 1 if parent else 0}
            return json.dumps({"operations": [{"op": "INSERT", "node_ids": [],
                "observation_ids": [obs["observation_id"]], "new_node": new, "reason": "Independent short action omitted by H0"}],
                "conflicts": []})
        if role == "boundary_refinement":
            target = data["target"]
            return json.dumps({"visible": True, "start": target["start"], "end": target["end"],
                               "boundary_uncertainty": target["boundary_uncertainty"], "before": None, "after": None})
        if role in ("targeted_review", "coverage_review"):
            return json.dumps({"operations": [], "conflicts": [], "refuted_observation_ids": [], "resolved": False})
        if role == "independent_check":
            return json.dumps({"nodes": [node(data["start"], data["end"])]})
        if role == "confidence_comparison":
            return json.dumps({"confidence": dict.fromkeys(("semantic", "boundary", "hierarchy"), "low"),
                               "reason": "Ambiguous", "conflict": True})
        if role == "local_dedup":
            return json.dumps({"comparisons": [{"observation_id": r["observation_id"], "relation": "different_event",
                                               "reason": "distinct"} for r in data["candidates"]]})
        return '{"operations":[],"conflicts":[]}'


def locations(cfg):
    video = Path(cfg["paths"]["datasets"]).parent / "videos/video.mp4"
    output = Path(cfg["paths"]["wiki"]) / "qvhighlights/videos/video"
    return video, output


def test_blind_sweep_recovers_short_event_and_freezes(prepared, monkeypatch):
    cfg, _ = prepared
    config(cfg)
    def forbidden(*args, **kwargs):
        pytest.fail("Non-VLM visual/embedding analysis called")
    monkeypatch.setattr("harness.sampling.analyze_changes", forbidden)
    monkeypatch.setattr("harness.sampling.mixed_times", forbidden)
    monkeypatch.setattr("harness.embed_wiki.EmbeddingClient.embed", forbidden)
    video, output = locations(cfg)
    vlm = BidirectionalVLM()
    meta = ingest_video(video, "video", output, cfg, captioner=vlm)
    assert meta["pipeline_version"] == "vlm_bidirectional_v1" and meta["schema_version"] == 2
    h0, final = read_jsonl(output / "topdown_nodes.jsonl"), read_jsonl(output / "nodes.jsonl")
    assert len(h0) == 1 and len(final) == 2
    action = next(r for r in final if r["type"] == "action")
    assert action["title"] == "Brief tomato transfer" and action["parent_id"] == h0[0]["node_id"]
    blind = next(data for role, data, count in vlm.calls if role == "bottom_up")
    assert set(blind) == {"start", "end", "frame_timestamps"}
    assert next(count for role, data, count in vlm.calls if role == "top_down") == 30
    assert "Coarse phase" not in json.dumps(blind)
    coverage = read_jsonl(output / "coverage.jsonl")
    assert all(row["bottomup_seen"] for row in coverage)
    assert all(ids for row in coverage for ids in row["observation_support"].values())
    freeze_wiki(output)
    seal = verify_wiki(output)
    assert "reconciliation.jsonl" in seal["files"]
    query = {"query_id": "q1", "video_id": "video", "query": "Find tomato transfer", "split": "train"}
    with query_workspace(query, {**query, "max_predictions": 5}, output,
                         {"AGENTS.md": "fixture", "query_prompt.md": "fixture"}, Path(cfg["paths"]["runs"])) as workspace:
        assert (workspace / "wiki/coverage.jsonl").exists()
        assert (workspace / "wiki/bottomup_observations.jsonl").exists()
        assert not (workspace / "wiki/caption_audit.jsonl").exists()
    calls = len(vlm.calls)
    assert ingest_video(video, "video", output, cfg, captioner=vlm) == meta
    assert len(vlm.calls) == calls


def test_unexplained_observation_is_retained_as_unresolved_candidate(prepared):
    cfg, _ = prepared
    config(cfg)
    video, output = locations(cfg)
    meta = ingest_video(video, "video", output, cfg, captioner=BidirectionalVLM(omit=True))
    assert meta["review_status"] == "unresolved"
    final = read_jsonl(output / "nodes.jsonl")
    candidate = next(r for r in final if "unexplained_bottomup_observation" in r["issues"])
    assert candidate["review_status"] == "unresolved"
    assert candidate["confidence"]["semantic"] == "low"
    assert "UNRESOLVED" in (output / "wiki.md").read_text()


def test_resume_reuses_completed_passes_and_preserves_failures(prepared):
    cfg, _ = prepared
    config(cfg)
    video, output = locations(cfg)
    first = BidirectionalVLM(fail_at=3)
    with pytest.raises(HarnessError, match="fixture interruption"):
        ingest_video(video, "video", output, cfg, captioner=first)
    assert not output.exists()
    second = BidirectionalVLM()
    meta = ingest_video(video, "video", output, cfg, captioner=second)
    assert second.calls[0][0] == "reconciliation"
    assert meta["telemetry"]["reused_windows"] >= 2
    assert meta["telemetry"]["rejected_attempts"] >= 1
    assert len(read_jsonl(output / "nodes.jsonl")) == 2


def test_format_repair_is_text_only(prepared):
    cfg, _ = prepared
    config(cfg)
    video, output = locations(cfg)
    vlm = BidirectionalVLM(malformed=True)
    ingest_video(video, "video", output, cfg, captioner=vlm)
    assert any(role == "format_repair" and count == 0 for role, data, count in vlm.calls)


@pytest.mark.parametrize("duration,length,overlap", [(0.1, 45, .25), (100, 45, .25), (3600, 30, .25), (86400, 120, .15)])
def test_window_schedule_covers_timeline(duration, length, overlap):
    schedule = list(windows(duration, length, overlap))
    assert schedule[0][0] == 0 and schedule[-1][1] == duration
    assert all(b[0] <= a[1] for a, b in zip(schedule, schedule[1:]))
    assert all(0 <= start < end <= duration for start, end in schedule)


def test_batches_preserve_every_record_and_enforce_limits():
    source = [{"id": i, "text": "a" * 10} for i in range(1000)]
    result = list(batches(iter(source), 12, 500))
    assert [r for batch in result for r in batch] == source
    assert all(len(batch) <= 12 for batch in result)
    with pytest.raises(HarnessError, match="One semantic record"):
        list(batches([{"text": "x" * 1000}], 10, 20))


@pytest.fixture
def graph(tmp_path):
    graph = TemporalGraph(tmp_path / "graph.sqlite", 100, 100)
    obs = normalize_node(node(0, 100), 100)
    obs.update(observation_id="o1", evidence={"pass": ["bottom_up"], "observation_ids": ["o1"],
                                             "frame_ids": ["f1"], "frame_timestamps": [1.0]})
    graph.observe(obs, "bottom_up")
    root = {**node(0, 100), "node_id": "root", "parent_id": None, "evidence": obs["evidence"]}
    graph.put(root)
    graph.db.commit()
    yield graph
    graph.close()


def apply(graph, ops, **kwargs):
    return graph.apply(ops, expected_version=graph.version, request_id="test/request", **kwargs)


def test_edits_support_overlap_gaps_multisupport_and_atomic_rollback(graph):
    apply(graph, [{"op": "INSERT", "node_ids": [], "observation_ids": ["o1"], "ref": "$a", "reason": "evidence",
                   "new_node": node(10, 20, granularity=1, parent_id="root", type="state_change")},
                  {"op": "INSERT", "node_ids": [], "observation_ids": ["o1"], "reason": "overlap",
                   "new_node": node(15, 30, granularity=2, parent_id="$a", type="chapter")}])
    rows = list(graph.rows())
    assert len(rows) == 3 and len(graph.db.execute("SELECT * FROM support WHERE obs='o1'").fetchall()) == 3
    graph.validate()  # No fixed ontology or mandatory child coverage.
    assert list(graph.issues())  # Out-of-parent temporal extent is a review issue, not silent clipping.
    before = graph.version
    with pytest.raises(HarnessError):
        apply(graph, [{"op": "RELABEL", "node_ids": ["root"], "reason": "edit", "updates": {"title": "changed"}},
                      {"op": "REPARENT", "node_ids": ["root"], "parent_id": "missing", "reason": "broken"}])
    assert graph.version == before and graph.get("root")["title"] != "changed"


def test_merge_split_shift_keep_delete_and_stale_edits(graph):
    logs = apply(graph, [{"op": "SPLIT", "node_ids": ["root"], "reason": "split", "new_nodes": [
        node(0, 40, granularity=1), node(50, 100, granularity=1)]}])
    ids = [i for i in logs[0]["affected_ids"] if i != "root"]
    apply(graph, [{"op": "KEEP", "node_ids": ids, "observation_ids": ["o1"], "reason": "shared evidence"}])
    logs = apply(graph, [{"op": "MERGE", "node_ids": ids, "reason": "same activity",
                         "new_node": node(0, 100, granularity=1)}])
    merged = logs[0]["affected_ids"][0]
    assert graph.get(merged)["source_node_ids"] == ids
    with pytest.raises(HarnessError, match="requires local"):
        apply(graph, [{"op": "SHIFT", "node_ids": [merged], "reason": "candidate", "updates": {
            "start": 1, "end": 99, "boundary_uncertainty": {"start": [1, 1], "end": [99, 99]}}}])
    apply(graph, [{"op": "SHIFT", "node_ids": [merged], "reason": "visible", "updates": {
        "start": 1, "end": 99, "boundary_uncertainty": {"start": [1, 1], "end": [99, 99]}}}], visual=True)
    with pytest.raises(HarnessError, match="visual"):
        apply(graph, [{"op": "DELETE", "node_ids": [merged], "reason": "wrong"}])
    apply(graph, [{"op": "DELETE", "node_ids": ["root"], "reason": "wrong parent"}], visual=True)
    assert graph.get(merged)["parent_id"] is None
    with pytest.raises(HarnessError, match="stale"):
        graph.apply([], expected_version="old", request_id="x")


def test_legacy_confidence_is_not_promoted():
    old = {"level": "event", "confidence": 0.99}
    adapted = read_node(old)
    assert adapted["granularity"] == 2 and adapted["type"] == "event"
    assert adapted["legacy_confidence"] == .99 and adapted["confidence"]["semantic"] == "low"
    assert old == {"level": "event", "confidence": .99}


def test_normalize_node_clamps_copied_example_uncertainty():
    raw = node(12, 18)
    raw["boundary_uncertainty"] = {"start": [0.0, 0.5], "end": [0.5, 1.0]}
    out = normalize_node(raw, 31.5)
    start_lo, start_hi = out["boundary_uncertainty"]["start"]
    end_lo, end_hi = out["boundary_uncertainty"]["end"]
    assert start_lo <= 12 <= start_hi <= 31.5
    assert end_lo <= 18 <= end_hi <= 31.5
    overflow = node(0, 32)
    overflow["end"] = 32
    clamped = normalize_node(overflow, 31.5)
    assert clamped["end"] == 31.5


def test_split_accepts_singular_node_id(graph):
    apply(graph, [{"op": "SPLIT", "node_id": "root", "reason": "split", "new_nodes": [
        node(0, 40, granularity=1), node(50, 100, granularity=1)]}])
    assert len(list(graph.rows())) == 3


class ShapeMistakeVLM(BidirectionalVLM):
    """Reproduce the production schema mistakes that previously failed ingest."""

    def complete(self, prompt, images=()):
        if prompt.startswith("Repair only JSON"):
            return super().complete(prompt, images)
        role = prompt.split("Role: ", 1)[1].split("\n", 1)[0]
        data = json.loads(prompt.split("\nINPUT:\n", 1)[1])
        if role == "top_down":
            self.calls.append((role, data, len(images)))
            n = node(data["start"], data["end"], title="Coarse phase")
            n["boundary_uncertainty"] = {"start": [0.0, 0.5], "end": [0.5, 1.0]}
            return json.dumps({"nodes": [n], "comment": "extra key"})
        if role in ("targeted_review", "coverage_review"):
            self.calls.append((role, data, len(images)))
            return json.dumps({"operations": [], "conflicts": []})
        if role in ("reconciliation", "bottomup_organization"):
            self.calls.append((role, data, len(images)))
            obs = next(r for r in data["records"] if r["kind"] == "observation")
            parent = next((r for r in data["records"] if r["kind"] == "node"), None)
            new = {**node(1, 2, title=obs["title"], type="action"),
                   "parent_id": parent["node_id"] if parent else None, "granularity": 1 if parent else 0}
            return json.dumps({"operations": [
                {"op": "INSERT", "node_ids": [], "observation_ids": [obs["observation_id"]],
                 "new_node": new, "reason": "Independent short action omitted by H0"},
                {"op": "SPLIT", "node_id": parent["node_id"] if parent else "n_missing",
                 "observation_ids": [], "new_nodes": [node(0, 1)], "reason": "incomplete split"},
                {"op": "KEEP", "node_ids": ["n_not_in_batch"], "observation_ids": [obs["observation_id"]],
                 "reason": "outside batch"}], "conflicts": []})
        return super().complete(prompt, images)


def test_common_model_shape_mistakes_still_publish(prepared):
    cfg, _ = prepared
    config(cfg)
    video, output = locations(cfg)
    ingest_video(video, "video", output, cfg, captioner=ShapeMistakeVLM())
    assert output.exists()
    final = read_jsonl(output / "nodes.jsonl")
    assert any(r["title"] == "Brief tomato transfer" for r in final)


class GarbageReviewVLM(BidirectionalVLM):
    def complete(self, prompt, images=()):
        if prompt.startswith("Repair only JSON"):
            return super().complete(prompt, images)
        role = prompt.split("Role: ", 1)[1].split("\n", 1)[0]
        if role in ("targeted_review", "coverage_review"):
            self.calls.append((role, {}, len(images)))
            return json.dumps({"operations": "not-a-list", "conflicts": []})
        return super().complete(prompt, images)


def test_unusable_visual_review_degrades_instead_of_failing(prepared):
    cfg, _ = prepared
    config(cfg)
    video, output = locations(cfg)
    ingest_video(video, "video", output, cfg, captioner=GarbageReviewVLM(omit=True))
    assert output.exists()
    final = read_jsonl(output / "nodes.jsonl")
    assert any("unexplained_bottomup_observation" in r.get("issues", []) for r in final)


@pytest.mark.parametrize("broken", [
    {"nodes": [{"start": 0, "end": 0}]},
    {"nodes": [node(evidence=None)]},
    {"nodes": [node(confidence={"semantic": [], "boundary": "low", "hierarchy": "low"})]},
    {"nodes": [node(history=None)]},
    {"nodes": [node(parent_id=[])]},
])
def test_schema_reask_recovers_bad_model_fields(prepared, broken):
    cfg, _ = prepared
    config(cfg)
    class BadFields(BidirectionalVLM):
        def __init__(self):
            super().__init__()
            self.bad_sent = False
            self.corrected = False
        def complete(self, prompt, images=()):
            if not self.bad_sent:
                self.bad_sent = True
                return json.dumps(broken)
            if "Your structured answer was rejected" in prompt:
                self.corrected = True
                prompt = prompt.split("\nYour structured answer was rejected", 1)[0]
            return super().complete(prompt, images)
    client = BadFields()
    video, output = locations(cfg)
    ingest_video(video, "video", output, cfg, captioner=client)
    assert client.corrected
    audits = read_jsonl(output / "caption_audit.jsonl")
    assert any(a["kind"] == "schema_rejection" for row in audits for a in row["attempts"])


def test_truncation_reasks_concisely_without_accepting_partial_json(prepared):
    from harness.vlm_transport import ResponseRejected
    cfg, _ = prepared
    config(cfg)
    class Truncated(BidirectionalVLM):
        first = True
        def complete(self, prompt, images=()):
            if self.first:
                self.first = False
                raise ResponseRejected("finish_reason='length'")
            if "Your structured answer was rejected" in prompt:
                assert "complete concise JSON" in prompt
                prompt = prompt.split("\nYour structured answer was rejected", 1)[0]
            return super().complete(prompt, images)
    video, output = locations(cfg)
    ingest_video(video, "video", output, cfg, captioner=Truncated())
    assert output.exists()


def test_permanent_invalid_json_is_bounded_and_resumable(prepared):
    cfg, _ = prepared
    config(cfg)
    class AlwaysBroken:
        calls = 0
        def complete(self, prompt, images=()):
            self.calls += 1
            return 'not JSON'
    client = AlwaysBroken()
    video, output = locations(cfg)
    with pytest.raises(HarnessError, match="repair budget exhausted"):
        ingest_video(video, "video", output, cfg, captioner=client)
    repairs = cfg["ingest"]["caption_max_repairs"]
    assert client.calls == 1 + 2 * repairs
    assert not output.exists()
    checkpoint_root = output.parent.parent / ".ingest-checkpoints"
    assert list(checkpoint_root.rglob("*.json"))
    ingest_video(video, "video", output, cfg, captioner=BidirectionalVLM())
    assert output.exists()


def test_format_repair_transport_failure_has_no_extra_semantic_retries(prepared):
    cfg, _ = prepared
    config(cfg)
    class BrokenRepair:
        calls = 0
        def complete(self, prompt, images=()):
            self.calls += 1
            if self.calls == 1:
                return 'broken JSON'
            raise HarnessError("VLM request failed with HTTP 401")
    client = BrokenRepair()
    video, output = locations(cfg)
    with pytest.raises(HarnessError, match="HTTP 401"):
        ingest_video(video, "video", output, cfg, captioner=client)
    assert client.calls == 2
    assert not output.exists()


@pytest.mark.parametrize('bug', [AttributeError, IndexError])
def test_parser_programming_errors_are_not_model_repairs(prepared, monkeypatch, bug):
    cfg, _ = prepared
    config(cfg)
    def broken_parser(*args, **kwargs):
        raise bug('parser bug')
    monkeypatch.setattr('harness.bidirectional.BidirectionalBuilder.parse_nodes', broken_parser)
    video, output = locations(cfg)
    vlm = BidirectionalVLM()
    with pytest.raises(bug, match='parser bug'):
        ingest_video(video, 'video', output, cfg, captioner=vlm)
    assert len(vlm.calls) == 1
    assert not output.exists()
