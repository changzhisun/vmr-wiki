"""Versioned flat temporal graphs and transactional, evidence-linked edits."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sqlite3

from harness.common import HarnessError, canonical, nonempty, number, object_hash

TYPES = {"chapter", "scene", "event", "action", "state_change", "transition", "dialogue", "other"}
CONFIDENCE = {"high", "medium", "low"}
OPERATIONS = {"KEEP", "INSERT", "DELETE", "SPLIT", "MERGE", "SHIFT", "RELABEL", "REPARENT"}
SEMANTICS = {"type", "title", "summary", "actors", "actions", "objects", "state_before", "state_after",
             "retrieval_text", "confidence", "needs_refinement", "semantic_complexity", "multiple_actions"}
NODE_EXAMPLE = {
    "granularity": 0, "type": "event", "start": 0.0, "end": 1.0,
    "title": "Visible event", "summary": "Describe only visible evidence.",
    "actors": [], "actions": [], "objects": [], "state_before": "", "state_after": "",
    "confidence": {"semantic": "medium", "boundary": "low", "hierarchy": "medium"},
    "boundary_uncertainty": {"start": [0.0, 0.5], "end": [0.5, 1.0]},
    "needs_refinement": False, "semantic_complexity": "low", "multiple_actions": False,
    "retrieval_text": ["Visible event"],
}


def strings(value, name):
    if not isinstance(value, list):
        raise HarnessError(f"{name} must be a list")
    return list(dict.fromkeys(nonempty(item, name).strip() for item in value))


def normalize_node(raw, duration):
    if not isinstance(raw, dict):
        raise HarnessError("Node must be an object")
    row = deepcopy(raw)
    try:
        start, end = number(row["start"], "start", 0), number(row["end"], "end", 0)
        if not start < end <= duration:
            raise HarnessError("Node must satisfy 0 <= start < end <= video duration")
        row.update(start=start, end=end)
        if type(row["granularity"]) is not int or row["granularity"] < 0 or row["type"] not in TYPES:
            raise HarnessError("Invalid granularity or semantic type")
        for key in ("title", "summary"):
            row[key] = nonempty(row[key], key).strip()
        for key in ("actors", "actions", "objects"):
            row[key] = strings(row.get(key, []), key)
        for key in ("state_before", "state_after"):
            if not isinstance(row.get(key, ""), str):
                raise HarnessError(f"{key} must be a string")
            row.setdefault(key, "")
        conf = row["confidence"]
        if (not isinstance(conf, dict) or set(conf) != {"semantic", "boundary", "hierarchy"} or
                any(value not in CONFIDENCE for value in conf.values())):
            raise HarnessError("confidence must contain semantic/boundary/hierarchy discrete labels")
        uncertainty = row["boundary_uncertainty"]
        if not isinstance(uncertainty, dict) or set(uncertainty) != {"start", "end"}:
            raise HarnessError("boundary_uncertainty must contain start and end ranges")
        for key in ("start", "end"):
            bounds = uncertainty[key]
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise HarnessError("Boundary uncertainty must be a pair of seconds")
            lo, hi = number(bounds[0], "uncertainty", 0), number(bounds[1], "uncertainty", 0)
            if not lo <= row[key] <= hi <= duration:
                raise HarnessError("Uncertainty must bracket its boundary within the video")
        for key in ("needs_refinement", "multiple_actions"):
            row.setdefault(key, False)
            if type(row[key]) is not bool:
                raise HarnessError(f"{key} must be boolean")
        row.setdefault("semantic_complexity", "low")
        if row["semantic_complexity"] not in CONFIDENCE:
            raise HarnessError("Invalid semantic_complexity")
        row["retrieval_text"] = strings(row.get("retrieval_text", [row["summary"]]), "retrieval_text")
        row.setdefault("parent_id", None)
        if row["parent_id"] is not None:
            nonempty(row["parent_id"], "parent_id")
        row.setdefault("relations", [])
        if not isinstance(row["relations"], list):
            raise HarnessError("relations must be a list")
        for relation in row["relations"]:
            if (not isinstance(relation, dict) or set(relation) != {"type", "target_id"} or
                    relation["type"] not in {"part_of", "related", "continuation", "same_event"}):
                raise HarnessError("Invalid relation")
            nonempty(relation["target_id"], "relation target")
        row.setdefault("history", [])
        if not isinstance(row["history"], list) or any(not isinstance(item, dict) for item in row["history"]):
            raise HarnessError("history must be a list of records")
        row.setdefault("evidence", {"pass": [], "observation_ids": [], "frame_ids": [], "frame_timestamps": []})
        if not isinstance(row["evidence"], dict):
            raise HarnessError("evidence must be an object")
        for key in ("pass", "observation_ids", "frame_ids"):
            row["evidence"][key] = strings(row["evidence"].get(key, []), "evidence." + key)
        row["evidence"]["frame_timestamps"] = sorted(set(
            number(stamp, "evidence timestamp", 0) for stamp in row["evidence"].get("frame_timestamps", [])))
        if any(stamp >= duration for stamp in row["evidence"]["frame_timestamps"]):
            raise HarnessError("Evidence timestamp outside video")
        row.setdefault("review_status", "unresolved" if "low" in conf.values() else "resolved")
        if row["review_status"] not in {"resolved", "unresolved"}:
            raise HarnessError("Invalid review_status")
        row.setdefault("issues", [])
        row["issues"] = strings(row["issues"], "issues")
        return row
    except (KeyError, TypeError) as exc:
        raise HarnessError(f"Missing or malformed node field: {exc}") from exc


def read_node(raw):
    """Non-mutating legacy adapter; an uncalibrated numeric score stays legacy."""
    row = deepcopy(raw)
    if "granularity" not in row:
        levels = ("chapter", "scene", "event", "action")
        row["granularity"] = levels.index(row["level"])
        row["type"] = row["level"]
        row["legacy_confidence"] = row["confidence"]
        row["confidence"] = dict.fromkeys(("semantic", "boundary", "hierarchy"), "low")
    return row


def combine_evidence(*items):
    result = {"pass": [], "observation_ids": [], "frame_ids": [], "frame_timestamps": []}
    for item in items:
        for key in result:
            result[key].extend(item.get(key, []))
    return {key: sorted(set(value)) if key == "frame_timestamps" else list(dict.fromkeys(value))
            for key, value in result.items()}


class TemporalGraph:
    def __init__(self, path, duration, max_nodes):
        self.duration, self.max_nodes = duration, max_nodes
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS nodes(id TEXT PRIMARY KEY,parent TEXT,start REAL,end REAL,g INTEGER,data TEXT);
            CREATE INDEX IF NOT EXISTS node_time ON nodes(start,end);
            CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY,stage TEXT,start REAL,end REAL,data TEXT);
            CREATE INDEX IF NOT EXISTS obs_time ON observations(stage,start,end);
            CREATE TABLE IF NOT EXISTS support(obs TEXT,node TEXT,PRIMARY KEY(obs,node));
            CREATE TABLE IF NOT EXISTS refuted(obs TEXT PRIMARY KEY,request TEXT,reason TEXT);
            CREATE TABLE IF NOT EXISTS edits(seq INTEGER PRIMARY KEY,data TEXT);
            CREATE TABLE IF NOT EXISTS facts(key TEXT PRIMARY KEY,data TEXT);
        """)
        # This is a derived disk index. Rebuild deterministically from sealed
        # request checkpoints on resume; never trust an interrupted DB as input.
        for table in ("nodes", "observations", "support", "refuted", "edits", "facts"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()

    def close(self):
        self.db.close()

    @property
    def version(self):
        digest = hashlib.sha256()
        for row in self.db.execute("SELECT data FROM nodes ORDER BY id"):
            digest.update(row[0].encode())
        for row in self.db.execute("SELECT * FROM refuted ORDER BY obs"):
            digest.update(canonical(tuple(row)))
        return digest.hexdigest()

    def get(self, node_id):
        row = self.db.execute("SELECT data FROM nodes WHERE id=?", (node_id,)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown node: {node_id}")
        return json.loads(row[0])

    def observation(self, observation_id):
        row = self.db.execute("SELECT data FROM observations WHERE id=?", (observation_id,)).fetchone()
        if row is None:
            raise HarnessError(f"Unknown observation: {observation_id}")
        return json.loads(row[0])

    def observe(self, row, stage):
        self.db.execute("INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?)",
                        (row["observation_id"], stage, row["start"], row["end"], canonical(row).decode()))
        self.db.commit()

    def rows(self, *, start=None, end=None, stage=None, observations=False):
        table = "observations" if observations else "nodes"
        clauses, params = [], []
        if start is not None:
            clauses.append("end>?")
            params.append(start)
        if end is not None:
            clauses.append("start<?")
            params.append(end)
        if stage is not None:
            clauses.append("stage=?")
            params.append(stage)
        sql = f"SELECT data FROM {table}" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        for row in self.db.execute(sql + " ORDER BY start,end,id", params):
            yield json.loads(row[0])

    def put(self, row):
        row = normalize_node(row, self.duration)
        node_id = nonempty(row["node_id"], "node_id")
        if (self.db.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone() is None and
                self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] >= self.max_nodes):
            raise HarnessError("Bidirectional max_nodes exhausted")
        self.db.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?)", (
            node_id, row["parent_id"], row["start"], row["end"], row["granularity"], canonical(row).decode()))
        self.db.execute("DELETE FROM support WHERE node=?", (node_id,))
        for obs in row["evidence"]["observation_ids"]:
            self.observation(obs)
            self.db.execute("INSERT OR IGNORE INTO support VALUES(?,?)", (obs, node_id))

    def validate(self):
        for row in self.rows():
            normalize_node(row, self.duration)
            if row["parent_id"] is not None:
                parent = self.get(row["parent_id"])
                if parent["granularity"] >= row["granularity"]:
                    raise HarnessError("Child granularity must exceed parent granularity")
            for rel in row["relations"]:
                self.get(rel["target_id"])
            todo = [(row["node_id"], frozenset())]
            while todo:
                current, ancestors = todo.pop()
                if current in ancestors:
                    raise HarnessError("Cyclic primary/part_of hierarchy")
                node = self.get(current)
                parents = ([node["parent_id"]] if node["parent_id"] else []) + [
                    rel["target_id"] for rel in node["relations"] if rel["type"] == "part_of"]
                todo.extend((parent, ancestors | {current}) for parent in parents)

    def exported(self):
        for row in self.rows():
            row["children"] = [item[0] for item in self.db.execute(
                "SELECT id FROM nodes WHERE parent=? ORDER BY start,end,id", (row["node_id"],))]
            yield row

    def unsupported(self):
        for row in self.db.execute("""SELECT o.data FROM observations o WHERE stage='bottom_up'
            AND NOT EXISTS(SELECT 1 FROM support s WHERE s.obs=o.id)
            AND NOT EXISTS(SELECT 1 FROM refuted r WHERE r.obs=o.id) ORDER BY o.start,o.id"""):
            yield json.loads(row[0])

    def issues(self):
        for row in self.rows():
            if row["parent_id"]:
                parent = self.get(row["parent_id"])
                if row["start"] < parent["start"] or row["end"] > parent["end"]:
                    yield row["node_id"], "child_outside_parent"

    def apply(self, operations, *, expected_version, request_id, visual=False, visual_evidence=None,
              dry_run=False):
        if expected_version != self.version:
            raise HarnessError("Edit script graph version is stale")
        if not isinstance(operations, list):
            raise HarnessError("operations must be a list")
        self.db.commit()
        self.db.execute("SAVEPOINT edits")
        aliases, logs = {}, []
        try:
            for index, original in enumerate(operations):
                op = deepcopy(original)
                if not isinstance(op, dict) or op.get("op") not in OPERATIONS:
                    raise HarnessError("Unsupported edit operation")
                reason = nonempty(op.get("reason"), "operation reason")
                kind = op["op"]
                ids = strings(op.get("node_ids", []), "node_ids")
                ids = [aliases.get(node_id, node_id) for node_id in ids]
                nodes = [self.get(node_id) for node_id in ids]
                obs_ids = strings(op.get("observation_ids", []), "observation_ids")
                observations = [self.observation(obs) for obs in obs_ids]
                if not nodes and not observations and not visual:
                    raise HarnessError("Every operation needs node or observation evidence")
                evidence = combine_evidence(*(o["evidence"] for o in observations),
                                            visual_evidence or {})
                evidence["observation_ids"] = list(dict.fromkeys(evidence["observation_ids"] + obs_ids))
                history = {"pass": request_id.split("/", 1)[0], "operation": kind,
                           "request_id": request_id, "reason": reason}
                affected = []

                def save(node):
                    node["evidence"] = combine_evidence(node["evidence"], evidence)
                    node["history"].append(history)
                    self.put(node)
                    affected.append(node["node_id"])

                def create(raw, suffix="", parent=None):
                    if not isinstance(raw, dict):
                        raise HarnessError("new_node must be an object")
                    node = normalize_node(raw, self.duration)
                    node["node_id"] = "n_" + object_hash([request_id, index, suffix])[:24]
                    node["parent_id"] = aliases.get(node["parent_id"], node["parent_id"])
                    if parent is not None:
                        node["parent_id"] = parent
                    # Evidence/lineage cannot be supplied by generated prose.
                    node["evidence"], node["history"] = combine_evidence(evidence), []
                    save(node)
                    return node

                if kind == "KEEP":
                    if not nodes:
                        raise HarnessError("KEEP needs existing nodes")
                    for node in nodes:
                        save(node)
                elif kind == "INSERT":
                    if not observations and not visual:
                        raise HarnessError("INSERT needs observation or new visual evidence")
                    created = create(op.get("new_node"))
                    if "ref" in op:
                        ref = nonempty(op["ref"], "ref")
                        if not ref.startswith("$") or ref in aliases:
                            raise HarnessError("Temporary references must be unique $names")
                        aliases[ref] = created["node_id"]
                elif kind == "SPLIT":
                    if len(nodes) != 1 or not isinstance(op.get("new_nodes"), list) or len(op["new_nodes"]) < 2:
                        raise HarnessError("SPLIT needs one parent and at least two children")
                    evidence = combine_evidence(evidence, nodes[0]["evidence"])
                    for child_index, raw in enumerate(op["new_nodes"]):
                        child = create(raw, str(child_index), nodes[0]["node_id"])
                        if not nodes[0]["start"] <= child["start"] < child["end"] <= nodes[0]["end"]:
                            raise HarnessError("SPLIT children must lie inside the source node")
                    save(nodes[0])
                elif kind == "MERGE":
                    if len(nodes) < 2 or len({(n["parent_id"], n["granularity"]) for n in nodes}) != 1:
                        raise HarnessError("MERGE needs peers with one parent and granularity")
                    evidence = combine_evidence(evidence, *(n["evidence"] for n in nodes))
                    raw = deepcopy(op.get("new_node"))
                    if not isinstance(raw, dict):
                        raise HarnessError("MERGE needs new_node")
                    raw.update(start=min(n["start"] for n in nodes), end=max(n["end"] for n in nodes),
                               parent_id=nodes[0]["parent_id"], granularity=nodes[0]["granularity"])
                    created = create(raw)
                    created["source_node_ids"] = ids
                    self.put(created)
                    for other in self.rows():
                        if other["node_id"] in ids:
                            continue
                        changed = False
                        if other["parent_id"] in ids:
                            other["parent_id"] = created["node_id"]
                            changed = True
                        for rel in other["relations"]:
                            if rel["target_id"] in ids:
                                rel["target_id"] = created["node_id"]
                                changed = True
                        if changed:
                            save(other)
                    for node_id in ids:
                        self.db.execute("DELETE FROM nodes WHERE id=?", (node_id,))
                        self.db.execute("DELETE FROM support WHERE node=?", (node_id,))
                elif kind == "DELETE":
                    if not visual or not nodes:
                        raise HarnessError("DELETE requires targeted visual review")
                    for node in nodes:
                        for other in self.rows():
                            if other["node_id"] in ids:
                                continue
                            if other["parent_id"] == node["node_id"]:
                                other["parent_id"] = node["parent_id"]
                            other["relations"] = [r for r in other["relations"] if r["target_id"] not in ids]
                            self.put(other)
                        self.db.execute("DELETE FROM nodes WHERE id=?", (node["node_id"],))
                        self.db.execute("DELETE FROM support WHERE node=?", (node["node_id"],))
                    affected.extend(ids)
                elif kind in {"SHIFT", "RELABEL", "REPARENT"}:
                    if len(nodes) != 1:
                        raise HarnessError(f"{kind} needs exactly one node")
                    node = nodes[0]
                    if kind == "REPARENT":
                        parent = aliases.get(op.get("parent_id"), op.get("parent_id"))
                        cursor = parent
                        while cursor:
                            if cursor == node["node_id"]:
                                raise HarnessError("Cyclic reparent operation")
                            cursor = self.get(cursor)["parent_id"]
                        node["parent_id"] = parent
                        if parent:
                            node["granularity"] = max(node["granularity"], self.get(parent)["granularity"] + 1)
                        if "relations" in op:
                            node["relations"] = op["relations"]
                    else:
                        updates = op.get("updates")
                        allowed = {"start", "end", "boundary_uncertainty"} if kind == "SHIFT" else SEMANTICS
                        if not isinstance(updates, dict) or not updates or set(updates) - allowed:
                            raise HarnessError(f"Invalid {kind} fields")
                        if kind == "SHIFT" and not visual:
                            raise HarnessError("SHIFT requires local boundary evidence")
                        node.update(updates)
                    save(node)
                    if kind == "REPARENT":
                        queue = [(node["node_id"], node["granularity"])]
                        while queue:
                            parent_id, granularity = queue.pop()
                            for record in self.db.execute("SELECT id FROM nodes WHERE parent=?", (parent_id,)):
                                child = self.get(record[0])
                                child["granularity"] = max(child["granularity"], granularity + 1)
                                save(child)
                                queue.append((child["node_id"], child["granularity"]))
                logs.append({"operation": op, "affected_ids": affected, "before_nodes": nodes,
                             "request_id": request_id, "visual": visual})
            self.validate()
            after = self.version
            if dry_run:
                self.db.execute("ROLLBACK TO edits")
                self.db.execute("RELEASE edits")
                return logs
            for log in logs:
                log.update(before_version=expected_version, after_version=after)
                self.db.execute("INSERT INTO edits(data) VALUES(?)", (canonical(log).decode(),))
            self.db.execute("RELEASE edits")
            self.db.commit()
            return logs
        except BaseException:
            self.db.execute("ROLLBACK TO edits")
            self.db.execute("RELEASE edits")
            raise
