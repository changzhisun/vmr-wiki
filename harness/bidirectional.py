"""Bidirectional temporal parsing using one VLM and deterministic time logic."""
from __future__ import annotations

from copy import deepcopy
import json
import logging

from harness.bidirectional_config import PIPELINE_VERSION, SCHEMA_VERSION, budgets, settings
from harness.bidirectional_io import FrameIndex, RequestJournal, batches, stream_jsonl, windows
from harness.common import HarnessError, canonical, file_hash, now, object_hash, write_json
from harness.freeze import remove_tree, tree_hashes
from harness.temporal_graph import (NODE_EXAMPLE, OPERATIONS, SEMANTICS, TemporalGraph, cite_ids,
                                    combine_evidence, normalize_node, strings)

LOG = logging.getLogger(__name__)
NODE_SCHEMA = json.dumps(NODE_EXAMPLE)
EDIT_INSTRUCTION = """Compare sources without assuming top-down is correct. Preserve supported short moments.
Return JSON with "operations" and "conflicts" arrays (either may be empty). Extra keys are ignored.
Visual review must also include "refuted_observation_ids" (array, possibly empty) and "resolved" (boolean).
Allowed operations only: KEEP, INSERT, DELETE, SPLIT, MERGE, SHIFT, RELABEL, REPARENT.
Every operation has op, node_ids (list of existing IDs; [] if none), observation_ids (list of supplied evidence IDs), reason.
Cite only IDs present in this INPUT records list. Temporary INSERT refs may use "$name".
KEEP associates observations with one or more existing nodes.
INSERT has new_node with all node-schema fields and parent_id (existing ID or null).
SPLIT has node_ids with exactly one existing parent ID and new_nodes with at least two children;
keep the original as their broader parent.
MERGE has peer node_ids with equal granularity/parent and new_node with combined semantics;
its time range is the union envelope, inherited children and original evidence are preserved.
SHIFT has updates containing start, end, boundary_uncertainty; requires visual boundary review.
RELABEL has updates containing only semantic fields (including type, retrieval_text, confidence).
REPARENT has node_ids with exactly one ID, parent_id, and optional relations; keep child granularity above parent.
INSERT may assign a temporary ref such as "$parent" for subsequent operations in this batch.
DELETE always needs new visual evidence. Never delete because the other pass omitted a moment.
For visual contradictions (e.g. salt vs sugar) use conflicts with node_ids, observation_ids, reason;
do not guess or silently settle them from text. A conflict must cite at least one supplied reference.
Overlaps and gaps are legal when semantically meaningful. Do not force a time partition or fixed ontology.
Unknown facts stay unknown; no audio, external lookup, embeddings, detector, or other model is available.
boundary_uncertainty.start/end are [lo, hi] in seconds and must satisfy 0 <= lo <= that endpoint <= hi <= video duration.
Never copy example timestamps; they are shape only.
Node schema: """ + NODE_SCHEMA


def is_unusable_response(exc):
    return isinstance(exc, HarnessError) and "response repair budget exhausted" in str(exc)


def view(node):
    # Evidence/history remain in the disk index. Text comparison receives the
    # semantic record and stable references, not unbounded frame/history arrays.
    keys = SEMANTICS | {"node_id", "observation_id", "parent_id", "granularity", "start", "end",
                       "boundary_uncertainty", "review_status", "issues", "group_id"}
    return {key: value for key, value in node.items() if key in keys}


class BidirectionalBuilder:
    def __init__(self, video, duration, cfg, checkpoint, staging, client, check):
        self.duration, self.cfg, self.checkpoint, self.staging, self.check = duration, cfg, checkpoint, staging, check
        self.options = settings(cfg["ingest"])
        self.limits = budgets(self.options, duration)
        self.graph = TemporalGraph(checkpoint.root / "working.sqlite", duration, self.limits["max_nodes"])
        self.db = self.graph.db
        self.db.executescript("""CREATE TABLE IF NOT EXISTS groups(obs TEXT PRIMARY KEY,leader TEXT);
            DELETE FROM groups;
            CREATE TABLE IF NOT EXISTS queue(id TEXT PRIMARY KEY,reason TEXT,done INTEGER DEFAULT 0);
            DELETE FROM queue;
            CREATE TABLE IF NOT EXISTS boundary_queue(id TEXT PRIMARY KEY,done INTEGER DEFAULT 0);
            DELETE FROM boundary_queue;
            CREATE TABLE IF NOT EXISTS conflicts(id TEXT PRIMARY KEY,data TEXT);
            DELETE FROM conflicts;""")
        try:
            self.frames = FrameIndex(self.db, video, duration, checkpoint, staging, cfg["ingest"], check)
        except BaseException:
            self.graph.close()
            raise
        self.journal = RequestJournal(client, checkpoint, self.db, cfg, self.options, self.limits, check)
        self.cap = self.options["max_frames_per_call"]
        self.batch = self.options["reconciliation"]

    def packs(self, rows):
        return batches(rows, self.batch["max_records"], self.batch["max_chars"] - 6000)

    def visual(self, role, start, end, data, instruction, parser, *, replicate=None):
        self.journal.ensure_budget()
        frames = self.frames.images(start, end, self.cap)
        payload = {"start": start, "end": end, "frame_timestamps": [f["timestamp"] for f in frames], **data}
        result, request_id = self.journal.request(role, payload, instruction, parser, frames=frames, replicate=replicate)
        return result, request_id, self.frames.evidence(role, frames)

    def parse_nodes(self, payload, start, end, *, limit=None, require=False, parent=None):
        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            raise HarnessError("Response must contain a nodes array")
        if require and not payload["nodes"]:
            raise HarnessError("Global scan requires at least one coarse phase")
        if limit is not None and len(payload["nodes"]) > limit:
            raise HarnessError("Too many children in one response")
        rows = []
        for raw in payload["nodes"]:
            node = normalize_node(raw, self.duration)
            if not start <= node["start"] < node["end"] <= end:
                raise HarnessError("Node lies outside the request target")
            if parent and (node["start"], node["end"], node["title"], node["summary"]) == (
                    parent["start"], parent["end"], parent["title"], parent["summary"]):
                raise HarnessError("Recursive subdivision made no progress")
            rows.append(node)
        if [r["start"] for r in rows] != sorted(r["start"] for r in rows):
            raise HarnessError("Nodes must be chronological")
        return rows

    def observe(self, node, stage, request_id, index, evidence):
        row = deepcopy(node)
        obs_id = "o_" + object_hash([stage, request_id, index])[:24]
        row.pop("node_id", None)
        row.update(observation_id=obs_id, request_id=request_id)
        # Restrict references to the relevant frames, with bracketing evidence
        # retained when a semantic interval lies between two sparse observations.
        selected = [i for i, stamp in enumerate(evidence["frame_timestamps"])
                    if row["start"] <= stamp <= row["end"]]
        if not selected and evidence["frame_timestamps"]:
            selected = sorted(range(len(evidence["frame_timestamps"])), key=lambda i: min(
                abs(evidence["frame_timestamps"][i] - row["start"]),
                abs(evidence["frame_timestamps"][i] - row["end"])))[:2]
        row["evidence"] = {"pass": [stage], "observation_ids": [obs_id],
                           "frame_ids": [evidence["frame_ids"][i] for i in selected],
                           "frame_timestamps": [evidence["frame_timestamps"][i] for i in selected]}
        row["history"] = [{"pass": stage, "operation": "created", "request_id": request_id}]
        self.graph.observe(row, stage)
        return row

    def topdown(self, start, end, depth=0, parent=None, short_used=False):
        opts = self.options["topdown"]
        data = {"granularity": depth, "max_children": opts["max_children_per_node"]}
        if parent:
            data["parent_context"] = view(parent)
        instruction = ("Identify coarse phases." if parent is None else "Decompose this semantic interval.") + (
            " Return {\"nodes\":[...]} with chronological nodes, each using the schema below. "
            "Set needs_refinement and semantic_complexity based on visible evidence; multiple_actions=true "
            "only for several distinct sequential actions. boundary_uncertainty.start/end must satisfy "
            "0 <= lo <= the endpoint <= hi <= video duration; never copy example timestamps. "
            "Granularity is a temporal scale, never a fixed ontology. Overlaps/gaps are allowed; static regions "
            "may remain covered by a broader parent. An indivisible parent may return no children. "
            "The global scan must return at least one coarse phase.\n" + NODE_SCHEMA)
        nodes, req, evidence = self.visual("top_down", start, end, data, instruction, lambda payload: self.parse_nodes(
            payload, start, end, limit=opts["max_children_per_node"], require=parent is None, parent=parent))
        for index, raw in enumerate(nodes):
            raw["granularity"] = depth
            raw["parent_id"] = parent["node_id"] if parent else None
            obs = self.observe(raw, "top_down", req, index, evidence)
            node = {**obs, "node_id": "n_" + object_hash(obs["observation_id"])[:24]}
            node.pop("observation_id")
            self.graph.put(node)
            self.db.commit()
            short = node["end"] - node["start"] <= opts["min_segment_duration_sec"]
            can_split = node["needs_refinement"] and (not short or (node["multiple_actions"] and not short_used))
            if depth + 1 < opts["max_depth"] and can_split:
                self.topdown(node["start"], node["end"], depth + 1, node, short_used or short)

    def bottomup(self):
        opts = self.options["bottomup"]
        for window_index, (start, end) in enumerate(windows(self.duration, opts["window_sec"], opts["overlap_ratio"])):
            # Blind means a new request with no previous answers, H0 labels,
            # ancestors, observations, or model session in this input.
            data = {}
            if not opts["independent_observation"]:
                # Experimental contextual control. Separate bounded requests
                # carry all overlapping H0 context if it exceeds one batch.
                contexts = self.packs(view(n) for n in self.graph.rows(start=start, end=end))
            else:
                contexts = iter([None])
            called = False
            for context in contexts:
                called = True
                data = {} if context is None else {"topdown_context": context}
                self.bottomup_window(start, end, window_index, data)
            if not called:
                self.bottomup_window(start, end, window_index, {"topdown_context": []})

    def bottomup_window(self, start, end, window_index, data):
        nodes, req, evidence = self.visual("bottom_up", start, end, data,
            "Independently enumerate ALL significant visible local actions, events and state changes in time order. "
            "Pay special attention to brief actions, entry/exit, object transfers and state changes. "
            "Do not invent activity in quiet frames; an empty array is valid. Do not guess unseen details. "
            "Return {\"nodes\":[...]} using this schema; granularity is provisional, not an ontology.\n" + NODE_SCHEMA,
            lambda payload: self.parse_nodes(payload, start, end))
        for index, node in enumerate(nodes):
            node["window_index"] = window_index
            obs = self.observe(node, "bottom_up", req, index, evidence)
            self.db.execute("INSERT INTO groups VALUES(?,?)", (obs["observation_id"], obs["observation_id"]))
        self.db.commit()

    def deduplicate(self):
        # Only time-near observations from adjacent overlapping windows are
        # candidates. Semantics are decided by the VLM, never by overlap alone.
        for left in self.graph.rows(observations=True, stage="bottom_up"):
            candidates = (right for right in self.graph.rows(observations=True, stage="bottom_up",
                            start=max(0, left["start"] - 2), end=min(self.duration, left["end"] + 2))
                          if right["window_index"] == left["window_index"] + 1)
            for pack in self.packs(view(row) for row in candidates):
                allowed = {r["observation_id"] for r in pack}
                def parse(payload):
                    if not isinstance(payload, dict) or set(payload) != {"comparisons"}:
                        raise HarnessError("Expected comparisons")
                    comparisons = payload["comparisons"]
                    if not isinstance(comparisons, list):
                        raise HarnessError("comparisons must be a list")
                    seen = []
                    for item in comparisons:
                        if (not isinstance(item, dict) or item.get("observation_id") not in allowed or
                                item.get("relation") not in {"same_event", "continuation", "different_event"} or
                                not isinstance(item.get("reason"), str) or not item["reason"].strip()):
                            raise HarnessError("Invalid local comparison")
                        seen.append(item["observation_id"])
                    if len(seen) != len(set(seen)) or set(seen) != allowed:
                        raise HarnessError("Compare every candidate exactly once")
                    return comparisons
                comparisons, req = self.journal.request("local_dedup", {"left": view(left), "candidates": pack},
                    "Compare each candidate to the left observation. Decide same_event, continuation, or different_event. "
                    "Repeated but separate occurrences are different_event. Return {\"comparisons\":[{\"observation_id\":...,"
                    "\"relation\":...,\"reason\":...}]}. Never assume time overlap means the same event.", parse)
                for item in comparisons:
                    pair = [left["observation_id"], item["observation_id"]]
                    if item["relation"] == "same_event":
                        leaders = [self.db.execute("SELECT leader FROM groups WHERE obs=?", (o,)).fetchone()[0] for o in pair]
                        leader = min(leaders)
                        for old in leaders:
                            self.db.execute("UPDATE groups SET leader=? WHERE leader=?", (leader, old))
                    self.db.execute("INSERT INTO edits(data) VALUES(?)", (canonical({
                        "request_id": req, "kind": "observation_dedup", "relation": item["relation"],
                        "operation": {"op": "MERGE" if item["relation"] == "same_event" else "KEEP",
                                      "observation_ids": pair, "reason": item["reason"]}}).decode(),))
                self.db.commit()

    def scoped_records(self, start, end, include_observations=True):
        for node in self.graph.rows(start=start, end=end):
            yield {"kind": "node", **view(node)}
        if include_observations:
            for obs in self.graph.rows(start=start, end=end, observations=True, stage="bottom_up"):
                leader = self.db.execute("SELECT leader FROM groups WHERE obs=?", (obs["observation_id"],)).fetchone()
                yield {"kind": "observation", **view(obs), "group_id": leader[0]}

    def with_context(self, pack):
        # Every target retains ancestor and immediate neighbor references. If
        # the closure is too large, split target packs further, never truncate.
        seen = {row["node_id"] for row in pack if "node_id" in row}
        result = list(pack)
        for original in list(pack):
            parent_id = original.get("parent_id")
            while parent_id:
                parent = self.graph.get(parent_id)
                if parent_id not in seen:
                    result.append({"kind": "context", **view(parent)})
                    seen.add(parent_id)
                parent_id = parent["parent_id"]
        for edge in (min(r["start"] for r in pack), max(r["end"] for r in pack)):
            for order, comparator in (("DESC", "<="), ("ASC", ">=")):
                row = self.db.execute(f"SELECT id FROM nodes WHERE start{comparator}? ORDER BY start {order},id LIMIT 1",
                                      (edge,)).fetchone()
                if row and row[0] not in seen:
                    result.append({"kind": "context", **view(self.graph.get(row[0]))})
                    seen.add(row[0])
        if (len(result) > self.batch["max_records"] or
                len(json.dumps(result, ensure_ascii=False)) > self.batch["max_chars"] - 6000):
            if len(pack) == 1:
                raise HarnessError("One target and required context exceed text batch limits")
            mid = len(pack) // 2
            yield from self.with_context(pack[:mid])
            yield from self.with_context(pack[mid:])
        else:
            yield result

    def parse_edits(self, payload, records, role, *, visual=False, evidence=None):
        if not isinstance(payload, dict):
            raise HarnessError("Expected operations and conflicts arrays")
        operations = payload.get("operations") or []
        conflicts = payload.get("conflicts") or []
        if not isinstance(operations, list) or not isinstance(conflicts, list):
            raise HarnessError("Expected operations and conflicts arrays")
        known_nodes = {r["node_id"] for r in records if "node_id" in r}
        known_obs = {r["observation_id"] for r in records if "observation_id" in r}

        def bound(item, key, known):
            try:
                refs = cite_ids(item, key)
            except HarnessError:
                return []
            if key == "node_ids":
                return [n for n in refs if n in known or n.startswith("$")]
            return [n for n in refs if n in known]

        def keep(item, *, operation=False):
            if not isinstance(item, dict):
                raise HarnessError("Edit/conflict must be an object")
            item = {**item, "node_ids": bound(item, "node_ids", known_nodes),
                    "observation_ids": bound(item, "observation_ids", known_obs)}
            kind, nodes, obs = item.get("op"), item["node_ids"], item["observation_ids"]
            if operation:
                if kind == "SPLIT" and (len(nodes) != 1 or not isinstance(item.get("new_nodes"), list)
                                        or len(item["new_nodes"]) < 2):
                    return None
                if kind == "MERGE" and len(nodes) < 2:
                    return None
                if kind in {"DELETE", "SHIFT", "RELABEL", "REPARENT"} and len(nodes) != 1:
                    return None
                if kind == "KEEP" and not nodes:
                    return None
                if kind == "INSERT":
                    if not isinstance(item.get("new_node"), dict) or not (obs or visual):
                        return None
                elif kind not in OPERATIONS:
                    return None
                elif not nodes and not obs:
                    return None
            elif not nodes and not obs:
                return None
            if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                raise HarnessError("Every discrepancy needs a reason")
            return item

        payload = {"operations": [item for item in (keep(op, operation=True) for op in operations) if item],
                   "conflicts": [item for item in (keep(conflict) for conflict in conflicts) if item]}
        self.graph.apply(payload["operations"], expected_version=self.graph.version,
                         request_id=role + "/validation", visual=True, visual_evidence=evidence, dry_run=True)
        return payload

    def needs_visual(self, operation):
        if operation["op"] in {"DELETE", "SHIFT"} or operation.get("requires_visual_review"):
            return True
        if operation["op"] == "RELABEL":
            node = self.graph.get(operation["node_ids"][0])
            for key in ("actors", "objects", "state_before", "state_after"):
                if key in operation["updates"] and node[key] and operation["updates"][key] != node[key]:
                    return True
        return False

    def queue_conflict(self, conflict):
        key = object_hash(conflict)
        self.db.execute("INSERT OR IGNORE INTO conflicts VALUES(?,?)", (key, canonical(conflict).decode()))
        self.db.commit()

    def apply_response(self, response, req, *, visual=False, evidence=None):
        safe = []
        for operation in response["operations"]:
            if not visual and self.needs_visual(operation):
                self.queue_conflict({"node_ids": operation.get("node_ids", []),
                                     "observation_ids": operation.get("observation_ids", []),
                                     "reason": operation["reason"], "proposed_operation": operation})
            else:
                safe.append(operation)
        # No partially applied script: dependent operations stay together for
        # visual review when one operation requires it.
        if len(safe) != len(response["operations"]):
            ids = list(dict.fromkeys(n for op in response["operations"] for n in op.get("node_ids", []) if not n.startswith("$")))
            obs = list(dict.fromkeys(o for op in response["operations"] for o in op.get("observation_ids", [])))
            self.queue_conflict({"node_ids": ids, "observation_ids": obs, "reason": "Atomic script requires visual adjudication"})
            safe = []
        logs = self.graph.apply(safe, expected_version=self.graph.version, request_id=req,
                                visual=visual, visual_evidence=evidence)
        for log in logs:
            if log["operation"]["op"] in {"INSERT", "SPLIT", "SHIFT"}:
                for node_id in log["affected_ids"]:
                    self.db.execute("INSERT OR IGNORE INTO boundary_queue(id) VALUES(?)", (node_id,))
        for conflict in response["conflicts"]:
            self.queue_conflict(conflict)
        self.db.commit()

    def reconcile(self, role="reconciliation", include_observations=True):
        for start, end in windows(self.duration, self.batch["window_sec"]):
            for pack in self.packs(self.scoped_records(start, end, include_observations)):
                for records in self.with_context(pack):
                    # Refresh node records after earlier batches edited them.
                    fresh = []
                    for record in records:
                        if "node_id" in record:
                            try:
                                record = {"kind": record["kind"], **view(self.graph.get(record["node_id"]))}
                            except HarnessError:
                                continue
                        fresh.append(record)
                    if not fresh:
                        continue
                    payload = {"start": start, "end": end, "graph_version": self.graph.version, "records": fresh}
                    instruction = EDIT_INSTRUCTION
                    if role == "consistency_review":
                        instruction += ("\nReview state continuity, parent semantics, time scale differences, "
                                        "duplicates, meaningful overlaps/gaps and evidence. New visual claims require conflicts.")
                    if role == "bottomup_organization":
                        instruction += "\nOrganize these independent observations into useful temporal hierarchy."
                    try:
                        response, req = self.journal.request(role, payload, instruction,
                            lambda value: self.parse_edits(value, fresh, role))
                    except HarnessError as exc:
                        if not is_unusable_response(exc):
                            raise
                        LOG.warning("%s: skipping unusable batch: %s", role, exc)
                        continue
                    self.apply_response(response, req)

    def link_duplicates(self):
        # Only propagate a same_event judgment, never a mere temporal overlap
        # or continuation judgment. Keep every raw observation separately.
        for row in self.db.execute("SELECT obs,leader FROM groups ORDER BY obs"):
            obs_id, leader = row
            matches = self.db.execute("""SELECT DISTINCT s.node FROM groups g JOIN support s ON s.obs=g.obs
                WHERE g.leader=? ORDER BY s.node""", (leader,)).fetchall()
            for match in matches:
                node = self.graph.get(match[0])
                if obs_id not in node["evidence"]["observation_ids"]:
                    obs = self.graph.observation(obs_id)
                    node["evidence"] = combine_evidence(node["evidence"], obs["evidence"])
                    node["history"].append({"pass": "local_dedup", "operation": "support_association", "observation_id": obs_id})
                    self.graph.put(node)
        self.db.commit()

    def mark_unresolved(self, node_id, reason):
        try:
            node = self.graph.get(node_id)
        except HarnessError:
            return
        node["review_status"] = "unresolved"
        node["issues"] = list(dict.fromkeys(node["issues"] + [reason]))
        self.graph.put(node)
        self.db.commit()

    def conflict_records(self, conflict):
        records = []
        for node_id in conflict.get("node_ids", []):
            try:
                records.append({"kind": "node", **view(self.graph.get(node_id))})
            except HarnessError:
                pass
        for obs in conflict.get("observation_ids", []):
            records.append({"kind": "observation", **view(self.graph.observation(obs))})
        return records

    def review_conflict(self, conflict, *, role="targeted_review", max_rounds=None):
        records = self.conflict_records(conflict)
        if not records:
            return
        count = max_rounds or self.options["confidence"]["max_review_rounds"]
        for record_pack in self.packs(records):
            start = max(0, min(r["start"] for r in record_pack) - 4)
            end = min(self.duration, max(r["end"] for r in record_pack) + 4)
            for round_index in range(count):
                def parse(value):
                    if not isinstance(value, dict):
                        raise HarnessError("Visual review needs edits, refuted_observation_ids and resolved boolean")
                    resolved = value.get("resolved", False)
                    if isinstance(resolved, str) and resolved.strip().lower() in {"true", "false"}:
                        resolved = resolved.strip().lower() == "true"
                    if type(resolved) is not bool:
                        raise HarnessError("resolved must be boolean")
                    refs = value.get("refuted_observation_ids") or []
                    if isinstance(refs, str):
                        refs = [refs] if refs.strip() else []
                    try:
                        refs = strings(refs, "refuted_observation_ids")
                    except HarnessError:
                        refs = []
                    available = {r["observation_id"] for r in record_pack if "observation_id" in r}
                    refs = [obs_id for obs_id in refs if obs_id in available]
                    edited = self.parse_edits(value, record_pack, role, visual=True)
                    return {**edited, "refuted_observation_ids": refs, "resolved": resolved}
                try:
                    result, req, evidence = self.visual(role, start, end,
                        {"records": record_pack, "discrepancy": conflict["reason"], "round": round_index,
                         "graph_version": self.graph.version},
                        EDIT_INSTRUCTION + "\nReinspect the images to adjudicate the discrepancy. "
                        "Return operations, conflicts, refuted_observation_ids (array) and resolved (boolean). "
                        "Refute only observations demonstrably false from these frames, never merely absent "
                        "from sparse samples. Empty edits are allowed if unresolved.", parse)
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning("%s: unusable response retained as unresolved: %s", role, exc)
                    for record in record_pack:
                        if "node_id" in record:
                            self.mark_unresolved(record["node_id"], conflict["reason"])
                    break
                self.apply_response(result, req, visual=True, evidence=evidence)
                for obs_id in result["refuted_observation_ids"]:
                    self.db.execute("INSERT OR REPLACE INTO refuted VALUES(?,?,?)", (obs_id, req, conflict["reason"]))
                if result["resolved"] and not result["conflicts"]:
                    for record in record_pack:
                        if "node_id" in record:
                            try:
                                node = self.graph.get(record["node_id"])
                            except HarnessError:
                                continue
                            node["review_status"] = "resolved"
                            node["issues"] = []
                            node["history"].append({"pass": role, "operation": "review_resolved", "request_id": req})
                            self.graph.put(node)
                    self.db.commit()
                    break
                for record in record_pack:
                    if "node_id" in record:
                        self.mark_unresolved(record["node_id"], conflict["reason"])
                self.db.commit()

    def targeted_reviews(self):
        # Snapshot by ID on disk: newly reported disagreements do not create
        # an unbounded recursive review loop.
        self.db.execute("DROP TABLE IF EXISTS pending_conflicts")
        self.db.execute("CREATE TEMP TABLE pending_conflicts AS SELECT * FROM conflicts")
        self.db.execute("DELETE FROM conflicts")
        for row in self.db.execute("SELECT data FROM pending_conflicts ORDER BY id"):
            self.review_conflict(json.loads(row[0]))
        for row in self.db.execute("SELECT data FROM conflicts"):
            conflict = json.loads(row[0])
            for node_id in conflict.get("node_ids", []):
                self.mark_unresolved(node_id, conflict["reason"])

    def boundary(self, node_id):
        try:
            node = self.graph.get(node_id)
        except HarnessError:
            return
        options = self.options["boundary_refinement"]
        expand = (node["end"] - node["start"]) * options["context_expand_ratio"]
        intervals = [(max(0, node["start"] - expand), min(self.duration, node["end"] + expand), "both")]
        for round_index in range(options["max_rounds"]):
            next_intervals = []
            for start, end, endpoint in intervals:
                def parse(value):
                    if not isinstance(value, dict) or not {"visible", "start", "end", "boundary_uncertainty", "before", "after"} <= set(value):
                        raise HarnessError("Boundary answer requires visible, start/end, uncertainty, before/after")
                    if type(value["visible"]) is not bool:
                        raise HarnessError("visible must be boolean")
                    if value["visible"]:
                        updates = {k: value[k] for k in ("start", "end", "boundary_uncertainty")}
                        normalize_node({**node, **updates}, self.duration)
                        for key in ("before", "after"):
                            if value[key] is not None and (type(value[key]) not in (int, float) or not start <= value[key] <= end):
                                raise HarnessError("Before/after evidence must be visible timestamps or null")
                        checked = ("start", "end") if endpoint == "both" else (endpoint,)
                        if any(not start <= value[k] <= end for k in checked):
                            raise HarnessError("Refined endpoint lies outside its inspected interval")
                    return value
                try:
                    result, req, evidence = self.visual("boundary_refinement", start, end,
                        {"target": view(node), "endpoint": endpoint, "round": round_index},
                        "Find earliest clear beginning, latest ongoing timestamp and immediately before/after evidence. "
                        "Return {visible:boolean,start:seconds,end:seconds,boundary_uncertainty:{start:[lo,hi],end:[lo,hi]},"
                        "before:timestamp_or_null,after:timestamp_or_null}. If an endpoint is not visible, retain its "
                        "estimate and uncertainty. When inspecting only one endpoint keep the other unchanged. "
                        "Never invent frame-accurate timing from sparse frames. "
                        "boundary_uncertainty must satisfy 0 <= lo <= the endpoint <= hi <= video duration.", parse)
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning("boundary_refinement: unusable response retained as unresolved: %s", exc)
                    self.mark_unresolved(node_id, "boundary_response_unusable")
                    continue
                if not result["visible"]:
                    self.mark_unresolved(node_id, "boundary_not_visible")
                    continue
                updates = {k: result[k] for k in ("start", "end", "boundary_uncertainty")}
                if endpoint != "both":
                    other = "end" if endpoint == "start" else "start"
                    updates[other] = node[other]
                    updates["boundary_uncertainty"][other] = node["boundary_uncertainty"][other]
                self.graph.apply([{"op": "SHIFT", "node_ids": [node_id], "observation_ids": [],
                                   "updates": updates, "reason": "Locally reinspected boundary"}],
                                 expected_version=self.graph.version, request_id=req, visual=True, visual_evidence=evidence)
                node = self.graph.get(node_id)
                for key in ("start", "end") if endpoint == "both" else (endpoint,):
                    lo, hi = node["boundary_uncertainty"][key]
                    if hi - lo > options["uncertainty_sec"]:
                        radius = options["endpoint_context_sec"]
                        next_intervals.append((max(0, node[key] - radius), min(self.duration, node[key] + radius), key))
            intervals = next_intervals
            if not intervals:
                break

    def boundaries(self):
        if not self.options["boundary_refinement"]["enabled"]:
            return
        for node in self.graph.rows():
            if node["confidence"]["boundary"] == "low" or any(
                    hi - lo > self.options["boundary_refinement"]["uncertainty_sec"]
                    for lo, hi in node["boundary_uncertainty"].values()):
                self.db.execute("INSERT OR IGNORE INTO boundary_queue(id) VALUES(?)", (node["node_id"],))
        for row in self.db.execute("SELECT id FROM boundary_queue WHERE done=0 ORDER BY id"):
            self.boundary(row[0])
            self.db.execute("UPDATE boundary_queue SET done=1 WHERE id=?", (row[0],))
        self.db.commit()

    def self_consistency(self):
        if not self.options["confidence"]["self_consistency_for_low_confidence"]:
            return
        for node in self.graph.rows():
            if "low" not in node["confidence"].values():
                continue
            answers, evidence_sets = [], []
            try:
                for replica in (0, 1):
                    answer, req, evidence = self.visual("independent_check", node["start"], node["end"],
                        {"target_description": node["title"]}, "Independently describe this target from the frames. "
                        "Do not assume the target description is correct. Return {\"nodes\":[...]} using the node schema.\n" + NODE_SCHEMA,
                        lambda payload: self.parse_nodes(payload, node["start"], node["end"]), replicate=replica)
                    answers.append(answer)
                    evidence_sets.append(evidence)
            except HarnessError as exc:
                if not is_unusable_response(exc):
                    raise
                LOG.warning("independent_check: skipping unusable replica: %s", exc)
                self.mark_unresolved(node["node_id"], "independent_check_unusable")
                continue
            def parse(payload):
                if not isinstance(payload, dict) or not {"confidence", "reason", "conflict"} <= set(payload):
                    raise HarnessError("Comparison requires confidence, reason and conflict")
                normalize_node({**node, "confidence": payload["confidence"]}, self.duration)
                if type(payload["conflict"]) is not bool or not isinstance(payload["reason"], str):
                    raise HarnessError("Invalid confidence comparison")
                return payload
            # Geometry is descriptive evidence, not a semantic matcher.
            iou = None
            differences = {}
            if len(answers[0]) == len(answers[1]) == 1:
                a, b = answers[0][0], answers[1][0]
                intersection = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
                iou = intersection / (a["end"] - a["start"] + b["end"] - b["start"] - intersection)
                differences = {key: sorted(set(a[key]) ^ set(b[key])) for key in ("actors", "actions", "objects")}
            payload = {"target": view(node), "answers": [[view(n) for n in answer] for answer in answers],
                       "time_iou": iou, "literal_entity_differences": differences,
                       "parent": view(self.graph.get(node["parent_id"])) if node["parent_id"] else None}
            try:
                result, req = self.journal.request("confidence_comparison", payload,
                    "Compare the two independent visual answers. Return {confidence:{semantic:high|medium|low,"
                    "boundary:high|medium|low,hierarchy:high|medium|low},reason:string,conflict:boolean}. "
                    "Agreement is not truth or calibrated probability. Assess each dimension separately; IoU or "
                    "equal wording alone cannot promote semantic or hierarchy confidence. Missing evidence remains low.", parse)
            except HarnessError as exc:
                if not is_unusable_response(exc):
                    raise
                LOG.warning("confidence_comparison: skipping unusable comparison: %s", exc)
                self.mark_unresolved(node["node_id"], "confidence_comparison_unusable")
                continue
            node["confidence"] = result["confidence"]
            node["evidence"] = combine_evidence(node["evidence"], *evidence_sets)
            node["history"].append({"pass": "confidence_comparison", "operation": "confidence_review",
                                    "request_id": req, "time_iou": iou, "entity_differences": differences,
                                    "reason": result["reason"]})
            self.graph.put(node)
            self.db.commit()
            if result["conflict"] or "low" in result["confidence"].values():
                self.queue_conflict({"node_ids": [node["node_id"]], "observation_ids": [], "reason": result["reason"]})

    def coverage_review(self):
        self.link_duplicates()
        # Each unsupported observation gets one final independent visual check.
        for obs in self.graph.unsupported():
            overlapping = [n["node_id"] for n in self.graph.rows(start=obs["start"], end=obs["end"])]
            self.review_conflict({"node_ids": overlapping, "observation_ids": [obs["observation_id"]],
                                  "reason": "Bottom-up observation has no corresponding final node"},
                                 role="coverage_review", max_rounds=1)
            supported = self.db.execute("SELECT 1 FROM support WHERE obs=?", (obs["observation_id"],)).fetchone()
            refuted = self.db.execute("SELECT 1 FROM refuted WHERE obs=?", (obs["observation_id"],)).fetchone()
            if not supported and not refuted:
                node = deepcopy(obs)
                node.update(node_id="n_unresolved_" + object_hash(obs["observation_id"])[:20], parent_id=None,
                            granularity=0, review_status="unresolved", issues=["unexplained_bottomup_observation"],
                            confidence=dict.fromkeys(("semantic", "boundary", "hierarchy"), "low"))
                node.pop("observation_id", None)
                node["history"].append({"pass": "coverage_review", "operation": "INSERT", "reason": "Preserve unresolved evidence"})
                self.graph.put(node)
                self.db.execute("INSERT INTO edits(data) VALUES(?)", (canonical({"kind": "coverage_fallback",
                    "operation": {"op": "INSERT", "observation_ids": [obs["observation_id"]]},
                    "affected_ids": [node["node_id"]], "review_status": "unresolved"}).decode(),))
                self.db.commit()
        self.link_duplicates()

    def coverage_rows(self):
        self.db.execute("DROP TABLE IF EXISTS coverage_edges")
        self.db.execute("CREATE TEMP TABLE coverage_edges(t REAL PRIMARY KEY)")
        self.db.executemany("INSERT OR IGNORE INTO coverage_edges VALUES(?)", [(0,), (self.duration,)])
        for table in ("nodes", "observations"):
            for field in ("start", "end"):
                self.db.execute(f"INSERT OR IGNORE INTO coverage_edges SELECT {field} FROM {table}")
        for row in self.db.execute("SELECT data FROM sampling"):
            sample = json.loads(row[0])
            self.db.executemany("INSERT OR IGNORE INTO coverage_edges VALUES(?)", [(sample["start"],), (sample["end"],)])
        previous = None
        for edge in self.db.execute("SELECT t FROM coverage_edges ORDER BY t"):
            end = edge[0]
            if previous is None:
                previous = end
                continue
            start = previous
            previous = end
            nodes = [n for n in self.graph.rows(start=start, end=end)]
            observations = [o for o in self.graph.rows(start=start, end=end, observations=True, stage="bottom_up")]
            seen = {"top_down": False, "bottom_up": False}
            for record in self.db.execute("SELECT data FROM sampling"):
                sample = json.loads(record[0])
                if sample["pass"] in seen and sample["start"] <= start and sample["end"] >= end:
                    seen[sample["pass"]] = True
            mappings = {obs["observation_id"]: [r[0] for r in self.db.execute(
                "SELECT node FROM support WHERE obs=? ORDER BY node", (obs["observation_id"],))] for obs in observations}
            refuted = [o["observation_id"] for o in observations if self.db.execute(
                "SELECT 1 FROM refuted WHERE obs=?", (o["observation_id"],)).fetchone()]
            unresolved = any(n["review_status"] == "unresolved" for n in nodes)
            yield {"start": start, "end": end, "topdown_seen": seen["top_down"], "bottomup_seen": seen["bottom_up"],
                   "topdown_node_ids": [r[0] for r in self.db.execute("SELECT json_extract(data,'$.node_id') FROM facts "
                       "WHERE key LIKE 'h0:%' AND json_extract(data,'$.start')<? AND json_extract(data,'$.end')>?", (end, start))],
                   "bottomup_observation_ids": list(mappings), "observation_support": mappings,
                   "refuted_observation_ids": refuted, "final_node_count": len(nodes),
                   "final_node_ids": [n["node_id"] for n in nodes], "review_status": "unresolved" if unresolved else "resolved",
                   "seen_definition": "successful sampled-window coverage, not exhaustive event recall"}

    def build(self):
        opts = self.options["bottomup"]
        schedule = list(windows(self.duration, opts["window_sec"], opts["overlap_ratio"])) if opts["enabled"] else []
        preflight = {"bottomup_windows": len(schedule), "bottomup_image_inputs": sum(
            len(self.frames.times(a, b, self.cap)) for a, b in schedule), **self.limits}
        LOG.info("Bidirectional preflight: %s", preflight)
        if len(schedule) > self.limits["max_requests"]:
            raise HarnessError("Budget cannot cover the mandatory bottom-up sweep")
        write_json(self.staging / "preflight.json", preflight)
        if self.options["topdown"]["enabled"]:
            self.topdown(0, self.duration)
        for node in self.graph.exported():
            self.db.execute("INSERT INTO facts VALUES(?,?)", ("h0:" + node["node_id"], canonical(node).decode()))
        stream_jsonl(self.staging / "topdown_nodes.jsonl", self.graph.exported())
        if opts["enabled"]:
            self.bottomup()
            self.deduplicate()
            self.reconcile("reconciliation" if self.options["topdown"]["enabled"] else "bottomup_organization")
            self.link_duplicates()
            self.targeted_reviews()
        self.boundaries()
        self.reconcile("consistency_review", include_observations=False)
        # Separately inspect every cross-reference and primary parent edge,
        # including those spanning different text/time batches.
        for node in self.graph.rows():
            related = ([node["parent_id"]] if node["parent_id"] else []) + [r["target_id"] for r in node["relations"]]
            for target in dict.fromkeys(related):
                records = [view(node), view(self.graph.get(target))]
                try:
                    response, req = self.journal.request("relation_review", {"records": records, "graph_version": self.graph.version},
                        EDIT_INSTRUCTION + "\nReview this cross-reference or parent-child edge, especially state and temporal consistency.",
                        lambda payload: self.parse_edits(payload, records, "relation_review"))
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning("relation_review: skipping unusable edge: %s", exc)
                    continue
                self.apply_response(response, req)
        for node_id, reason in self.graph.issues():
            self.mark_unresolved(node_id, reason)
            self.queue_conflict({"node_ids": [node_id], "observation_ids": [], "reason": reason})
        self.self_consistency()
        self.targeted_reviews()
        if opts["enabled"]:
            self.coverage_review()
        self.boundaries()  # New review-created nodes only; completed queues do not repeat.
        for node_id, reason in self.graph.issues():
            self.mark_unresolved(node_id, reason)
        self.graph.validate()
        if opts["enabled"] and next(self.graph.unsupported(), None) is not None:
            raise HarnessError("Unaccounted bottom-up evidence cannot be published")
        stream_jsonl(self.staging / "nodes.jsonl", self.graph.exported())
        stream_jsonl(self.staging / "observations.jsonl", self.graph.rows(observations=True))
        stream_jsonl(self.staging / "bottomup_observations.jsonl", self.graph.rows(observations=True, stage="bottom_up"))
        stream_jsonl(self.staging / "frames.jsonl", (json.loads(r[0]) for r in self.db.execute("SELECT data FROM frames ORDER BY timestamp")))
        stream_jsonl(self.staging / "sampling.jsonl", (json.loads(r[0]) for r in self.db.execute("SELECT data FROM sampling ORDER BY seq")))
        stream_jsonl(self.staging / "reconciliation.jsonl", (json.loads(r[0]) for r in self.db.execute("SELECT data FROM edits ORDER BY seq")))
        stream_jsonl(self.staging / "caption_audit.jsonl", self.journal.audits())
        stream_jsonl(self.staging / "coverage.jsonl", self.coverage_rows())
        self.render()
        stats = self.journal.telemetry()
        unresolved = self.db.execute("SELECT COUNT(*) FROM nodes WHERE json_extract(data,'$.review_status')='unresolved'").fetchone()[0]
        return {**stats, "preflight": preflight, "extraction_sec_this_run": self.frames.extraction_sec,
                "extraction_sec": self.db.execute("SELECT COALESCE(SUM(elapsed),0) FROM frames").fetchone()[0],
                "reused_frames": self.frames.reused_frames, "unresolved_nodes": unresolved,
                "node_count": self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
                "frame_count": self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
                "inserted_nodes": self.db.execute("SELECT COUNT(*) FROM edits WHERE json_extract(data,'$.operation.op')='INSERT'").fetchone()[0],
                "refuted_observations": self.db.execute("SELECT COUNT(*) FROM refuted").fetchone()[0]}

    def render(self):
        def prose(value):
            return " ".join(str(value).split()).replace("<", "&lt;").replace(">", "&gt;")
        with (self.staging / "wiki.md").open("w", encoding="utf-8") as stream:
            stream.write(f"# Video\n\n## Metadata\n\n- Duration: {self.duration} seconds\n- Caption mode: bidirectional\n"
                         "- Schema version: 2\n- nodes.jsonl is the main graph; bottomup_observations.jsonl retains blind evidence.\n"
                         "- Coverage means successfully processed sampling windows, not guaranteed event recall.\n"
                         "- Confidence labels are evidence assessments, not probabilities.\n\n## Navigation\n\n")
            for node in self.graph.rows():
                if node["parent_id"] is None:
                    stream.write(f"- `{node['node_id']}` {node['start']:g}–{node['end']:g}s: {prose(node['title'])}\n")
            stream.write("\n## Temporal hierarchy\n\n")
            # Recursive traversal streams one node at a time; only IDs of the
            # current ancestry are held, not the full long-video tree.
            def visit(node):
                heading = "#" * min(6, node["granularity"] + 3)
                stream.write(f"{heading} {node['start']:g}–{node['end']:g}s · {prose(node['title'])} `{node['node_id']}`\n\n")
                if node["review_status"] == "unresolved":
                    stream.write("**UNRESOLVED — candidate requiring evidence inspection.** " + prose(", ".join(node["issues"])) + "\n\n")
                stream.write(prose(node["summary"]) + "\n\n")
                stream.write(f"- Granularity: {node['granularity']}; type: {node['type']}; parent: {node['parent_id']}\n")
                for key in ("actors", "actions", "objects", "state_before", "state_after", "confidence", "boundary_uncertainty", "relations"):
                    stream.write(f"- {key}: {prose(node[key])}\n")
                stream.write("- Evidence: see this node's observation/frame IDs in nodes.jsonl and frames.jsonl.\n\n")
                for child in self.db.execute("SELECT id FROM nodes WHERE parent=? ORDER BY start,end,id", (node["node_id"],)):
                    visit(self.graph.get(child[0]))
            for row in self.db.execute("SELECT id FROM nodes WHERE parent IS NULL ORDER BY start,end,id"):
                visit(self.graph.get(row[0]))


def publish_bidirectional(video, video_id, output, staging, checkpoint, client, cfg,
                          duration, video_stream_duration, source_hash, content_hash, ffmpeg_version, check):
    builder = BidirectionalBuilder(video, video_stream_duration, cfg, checkpoint, staging, client, check)
    try:
        telemetry = builder.build()
    finally:
        builder.graph.close()
    if file_hash(video) != source_hash:
        raise HarnessError("Source video changed during ingest")
    check()
    metadata = {"version": 1, "video_id": video_id, "duration": video_stream_duration,
                "container_duration": duration, "video_stream_duration": video_stream_duration,
                "source_sha256": source_hash, "ingest_config": cfg["ingest"], "ingest_config_hash": content_hash,
                "created_at": now(), "ffmpeg_version": ffmpeg_version,
                "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "review_status": "unresolved" if telemetry["unresolved_nodes"] else "resolved",
                "telemetry": telemetry, "content_hashes": tree_hashes(staging)}
    write_json(staging / "ingest.json", metadata)
    staging.rename(output)
    try:
        remove_tree(checkpoint.root)
    except OSError:
        LOG.warning("Published Wiki; could not remove checkpoint: %s", checkpoint.root)
    return metadata
