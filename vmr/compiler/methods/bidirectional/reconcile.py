"""Reconcile stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class ReconcileStage:
    def scoped_records(self, start, end, include_observations=True):
        for node in self.graph.rows(start=start, end=end):
            yield {"kind": "node", **view(node)}
        if include_observations:
            for obs in self.graph.rows(
                start=start, end=end, observations=True, stage="bottom_up"
            ):
                leader = self.db.execute(
                    "SELECT leader FROM groups WHERE obs=?", (obs["observation_id"],)
                ).fetchone()
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
                row = self.db.execute(
                    f"SELECT id FROM nodes WHERE start{comparator}? ORDER BY start {order},id LIMIT 1",
                    (edge,),
                ).fetchone()
                if row and row[0] not in seen:
                    result.append({"kind": "context", **view(self.graph.get(row[0]))})
                    seen.add(row[0])
        if (
            len(result) > self.batch["max_records"]
            or len(json.dumps(result, ensure_ascii=False))
            > self.batch["max_chars"] - 6000
        ):
            if len(pack) == 1:
                raise HarnessError(
                    "One target and required context exceed text batch limits"
                )
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
            item = {
                **item,
                "node_ids": bound(item, "node_ids", known_nodes),
                "observation_ids": bound(item, "observation_ids", known_obs),
            }
            kind, nodes, obs = item.get("op"), item["node_ids"], item["observation_ids"]
            if operation:
                if kind == "SPLIT" and (
                    len(nodes) != 1
                    or not isinstance(item.get("new_nodes"), list)
                    or len(item["new_nodes"]) < 2
                ):
                    return None
                if kind == "MERGE" and len(nodes) < 2:
                    return None
                if (
                    kind in {"DELETE", "SHIFT", "RELABEL", "REPARENT"}
                    and len(nodes) != 1
                ):
                    return None
                if kind == "KEEP" and not nodes:
                    return None
                if kind == "INSERT":
                    if not isinstance(item.get("new_node"), dict) or not (
                        obs or visual
                    ):
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

        payload = {
            "operations": [
                item for item in (keep(op, operation=True) for op in operations) if item
            ],
            "conflicts": [
                item for item in (keep(conflict) for conflict in conflicts) if item
            ],
        }
        self.graph.apply(
            payload["operations"],
            expected_version=self.graph.version,
            request_id=role + "/validation",
            visual=True,
            visual_evidence=evidence,
            dry_run=True,
        )
        return payload

    def needs_visual(self, operation):
        if operation["op"] in {"DELETE", "SHIFT"} or operation.get(
            "requires_visual_review"
        ):
            return True
        if operation["op"] == "RELABEL":
            node = self.graph.get(operation["node_ids"][0])
            for key in ("actors", "objects", "state_before", "state_after"):
                if (
                    key in operation["updates"]
                    and node[key]
                    and operation["updates"][key] != node[key]
                ):
                    return True
        return False

    def queue_conflict(self, conflict):
        key = object_hash(conflict)
        self.db.execute(
            "INSERT OR IGNORE INTO conflicts VALUES(?,?)",
            (key, canonical(conflict).decode()),
        )
        self.db.commit()

    def apply_response(self, response, req, *, visual=False, evidence=None):
        safe = []
        for operation in response["operations"]:
            if not visual and self.needs_visual(operation):
                self.queue_conflict(
                    {
                        "node_ids": operation.get("node_ids", []),
                        "observation_ids": operation.get("observation_ids", []),
                        "reason": operation["reason"],
                        "proposed_operation": operation,
                    }
                )
            else:
                safe.append(operation)
        # No partially applied script: dependent operations stay together for
        # visual review when one operation requires it.
        if len(safe) != len(response["operations"]):
            ids = list(
                dict.fromkeys(
                    n
                    for op in response["operations"]
                    for n in op.get("node_ids", [])
                    if not n.startswith("$")
                )
            )
            obs = list(
                dict.fromkeys(
                    o
                    for op in response["operations"]
                    for o in op.get("observation_ids", [])
                )
            )
            self.queue_conflict(
                {
                    "node_ids": ids,
                    "observation_ids": obs,
                    "reason": "Atomic script requires visual adjudication",
                }
            )
            safe = []
        logs = self.graph.apply(
            safe,
            expected_version=self.graph.version,
            request_id=req,
            visual=visual,
            visual_evidence=evidence,
        )
        for log in logs:
            if log["operation"]["op"] in {"INSERT", "SPLIT", "SHIFT"}:
                for node_id in log["affected_ids"]:
                    self.db.execute(
                        "INSERT OR IGNORE INTO boundary_queue(id) VALUES(?)", (node_id,)
                    )
        for conflict in response["conflicts"]:
            self.queue_conflict(conflict)
        self.db.commit()

    def reconcile(self, role="reconciliation", include_observations=True):
        for start, end in windows(self.duration, self.batch["window_sec"]):
            for pack in self.packs(
                self.scoped_records(start, end, include_observations)
            ):
                for records in self.with_context(pack):
                    # Refresh node records after earlier batches edited them.
                    fresh = []
                    for record in records:
                        if "node_id" in record:
                            try:
                                record = {
                                    "kind": record["kind"],
                                    **view(self.graph.get(record["node_id"])),
                                }
                            except HarnessError:
                                continue
                        fresh.append(record)
                    if not fresh:
                        continue
                    payload = {
                        "start": start,
                        "end": end,
                        "graph_version": self.graph.version,
                        "records": fresh,
                    }
                    instruction = EDIT_INSTRUCTION
                    if role == "consistency_review":
                        instruction += (
                            "\nReview state continuity, parent semantics, time scale differences, "
                            "duplicates, meaningful overlaps/gaps and evidence. New visual claims require conflicts."
                        )
                    if role == "bottomup_organization":
                        instruction += "\nOrganize these independent observations into useful temporal hierarchy."
                    try:
                        response, req = self.journal.request(
                            role,
                            payload,
                            instruction,
                            lambda value: self.parse_edits(value, fresh, role),
                        )
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
            matches = self.db.execute(
                """SELECT DISTINCT s.node FROM groups g JOIN support s ON s.obs=g.obs
                WHERE g.leader=? ORDER BY s.node""",
                (leader,),
            ).fetchall()
            for match in matches:
                node = self.graph.get(match[0])
                if obs_id not in node["evidence"]["observation_ids"]:
                    obs = self.graph.observation(obs_id)
                    node["evidence"] = combine_evidence(
                        node["evidence"], obs["evidence"]
                    )
                    node["history"].append(
                        {
                            "pass": "local_dedup",
                            "operation": "support_association",
                            "observation_id": obs_id,
                        }
                    )
                    self.graph.put(node)
        self.db.commit()
