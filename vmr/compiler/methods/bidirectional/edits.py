"""Pure temporal edit reducer. No filesystem or SQLite dependencies."""

from dataclasses import dataclass
from copy import deepcopy
import hashlib
from .model import (
    normalize_node,
    nonempty,
    HarnessError,
    canonical,
    OPERATIONS,
    SEMANTICS,
    cite_ids,
    combine_evidence,
    object_hash,
)


@dataclass
class GraphState:
    nodes: dict
    observations: dict
    refuted: tuple
    duration: float
    max_nodes: int

    @property
    def version(self):
        digest = hashlib.sha256()
        for key in sorted(self.nodes):
            digest.update(canonical(self.nodes[key]))
        for row in sorted(self.refuted):
            digest.update(canonical(row))
        return digest.hexdigest()

    def get(self, key):
        if key not in self.nodes:
            raise HarnessError(f"Unknown node: {key}")
        return deepcopy(self.nodes[key])

    def observation(self, key):
        if key not in self.observations:
            raise HarnessError(f"Unknown observation: {key}")
        return deepcopy(self.observations[key])

    def rows(self):
        return sorted(
            (deepcopy(n) for n in self.nodes.values()),
            key=lambda n: (n["start"], n["end"], n["node_id"]),
        )

    def put(self, row):
        row = normalize_node(row, self.duration)
        key = nonempty(row["node_id"], "node_id")
        if key not in self.nodes and len(self.nodes) >= self.max_nodes:
            raise HarnessError("Bidirectional max_nodes exhausted")
        for obs in row["evidence"]["observation_ids"]:
            self.observation(obs)
        self.nodes[key] = row

    def validate(self):
        for row in self.rows():
            normalize_node(row, self.duration)
            if row["parent_id"] is not None:
                parent = self.get(row["parent_id"])
                if parent["granularity"] >= row["granularity"]:
                    raise HarnessError(
                        "Child granularity must exceed parent granularity"
                    )
            for rel in row["relations"]:
                self.get(rel["target_id"])
            todo = [(row["node_id"], frozenset())]
            while todo:
                current, ancestors = todo.pop()
                if current in ancestors:
                    raise HarnessError("Cyclic primary/part_of hierarchy")
                node = self.get(current)
                parents = ([node["parent_id"]] if node["parent_id"] else []) + [
                    rel["target_id"]
                    for rel in node["relations"]
                    if rel["type"] == "part_of"
                ]
                todo.extend((parent, ancestors | {current}) for parent in parents)


def apply_edit(
    state,
    operations,
    *,
    expected_version,
    request_id,
    visual=False,
    visual_evidence=None,
):
    if expected_version != state.version:
        raise HarnessError("Edit script graph version is stale")
    if not isinstance(operations, list):
        raise HarnessError("operations must be a list")
    self = deepcopy(state)
    aliases, logs = {}, []
    for index, original in enumerate(operations):
        op = deepcopy(original)
        if not isinstance(op, dict) or op.get("op") not in OPERATIONS:
            raise HarnessError("Unsupported edit operation")
        reason = nonempty(op.get("reason"), "operation reason")
        kind = op["op"]
        ids = cite_ids(op, "node_ids")
        ids = [aliases.get(node_id, node_id) for node_id in ids]
        nodes = [self.get(node_id) for node_id in ids]
        obs_ids = cite_ids(op, "observation_ids")
        observations = [self.observation(obs) for obs in obs_ids]
        if not nodes and not observations and not visual:
            raise HarnessError("Every operation needs node or observation evidence")
        evidence = combine_evidence(
            *(o["evidence"] for o in observations), visual_evidence or {}
        )
        evidence["observation_ids"] = list(
            dict.fromkeys(evidence["observation_ids"] + obs_ids)
        )
        history = {
            "pass": request_id.split("/", 1)[0],
            "operation": kind,
            "request_id": request_id,
            "reason": reason,
        }
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
            if (
                len(nodes) != 1
                or not isinstance(op.get("new_nodes"), list)
                or len(op["new_nodes"]) < 2
            ):
                raise HarnessError("SPLIT needs one parent and at least two children")
            evidence = combine_evidence(evidence, nodes[0]["evidence"])
            for child_index, raw in enumerate(op["new_nodes"]):
                child = create(raw, str(child_index), nodes[0]["node_id"])
                if (
                    not nodes[0]["start"]
                    <= child["start"]
                    < child["end"]
                    <= nodes[0]["end"]
                ):
                    raise HarnessError("SPLIT children must lie inside the source node")
            save(nodes[0])
        elif kind == "MERGE":
            if (
                len(nodes) < 2
                or len({(n["parent_id"], n["granularity"]) for n in nodes}) != 1
            ):
                raise HarnessError("MERGE needs peers with one parent and granularity")
            evidence = combine_evidence(evidence, *(n["evidence"] for n in nodes))
            raw = deepcopy(op.get("new_node"))
            if not isinstance(raw, dict):
                raise HarnessError("MERGE needs new_node")
            raw.update(
                start=min(n["start"] for n in nodes),
                end=max(n["end"] for n in nodes),
                parent_id=nodes[0]["parent_id"],
                granularity=nodes[0]["granularity"],
            )
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
                self.nodes.pop(node_id, None)
                pass
        elif kind == "DELETE":
            if not visual or not nodes:
                raise HarnessError("DELETE requires targeted visual review")
            for node in nodes:
                for other in self.rows():
                    if other["node_id"] in ids:
                        continue
                    if other["parent_id"] == node["node_id"]:
                        other["parent_id"] = node["parent_id"]
                    other["relations"] = [
                        r for r in other["relations"] if r["target_id"] not in ids
                    ]
                    self.put(other)
                self.nodes.pop(node["node_id"], None)
                pass
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
                    node["granularity"] = max(
                        node["granularity"], self.get(parent)["granularity"] + 1
                    )
                if "relations" in op:
                    node["relations"] = op["relations"]
            else:
                updates = op.get("updates")
                allowed = (
                    {"start", "end", "boundary_uncertainty"}
                    if kind == "SHIFT"
                    else SEMANTICS
                )
                if (
                    not isinstance(updates, dict)
                    or not updates
                    or set(updates) - allowed
                ):
                    raise HarnessError(f"Invalid {kind} fields")
                if kind == "SHIFT" and not visual:
                    raise HarnessError("SHIFT requires local boundary evidence")
                node.update(updates)
            save(node)
            if kind == "REPARENT":
                queue = [(node["node_id"], node["granularity"])]
                while queue:
                    parent_id, granularity = queue.pop()
                    for record in [
                        (n["node_id"],)
                        for n in self.rows()
                        if n["parent_id"] == parent_id
                    ]:
                        child = self.get(record[0])
                        child["granularity"] = max(
                            child["granularity"], granularity + 1
                        )
                        save(child)
                        queue.append((child["node_id"], child["granularity"]))
        logs.append(
            {
                "operation": op,
                "affected_ids": affected,
                "before_nodes": nodes,
                "request_id": request_id,
                "visual": visual,
            }
        )
    self.validate()
    for log in logs:
        log.update(before_version=expected_version, after_version=self.version)
    return self, logs


def validate_edit(state, operations, **kwargs):
    apply_edit(state, operations, **kwargs)
