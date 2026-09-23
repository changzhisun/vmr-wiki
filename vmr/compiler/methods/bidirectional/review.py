"""Review stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class ReviewStage:
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
                        raise HarnessError(
                            "Visual review needs edits, refuted_observation_ids and resolved boolean"
                        )
                    resolved = value.get("resolved", False)
                    if isinstance(resolved, str) and resolved.strip().lower() in {
                        "true",
                        "false",
                    }:
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
                    available = {
                        r["observation_id"]
                        for r in record_pack
                        if "observation_id" in r
                    }
                    refs = [obs_id for obs_id in refs if obs_id in available]
                    edited = self.parse_edits(value, record_pack, role, visual=True)
                    return {
                        **edited,
                        "refuted_observation_ids": refs,
                        "resolved": resolved,
                    }

                try:
                    result, req, evidence = self.visual(
                        role,
                        start,
                        end,
                        {
                            "records": record_pack,
                            "discrepancy": conflict["reason"],
                            "round": round_index,
                            "graph_version": self.graph.version,
                        },
                        EDIT_INSTRUCTION
                        + "\nReinspect the images to adjudicate the discrepancy. "
                        "Return operations, conflicts, refuted_observation_ids (array) and resolved (boolean). "
                        "Refute only observations demonstrably false from these frames, never merely absent "
                        "from sparse samples. Empty edits are allowed if unresolved.",
                        parse,
                    )
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning(
                        "%s: unusable response retained as unresolved: %s", role, exc
                    )
                    for record in record_pack:
                        if "node_id" in record:
                            self.mark_unresolved(record["node_id"], conflict["reason"])
                    break
                self.apply_response(result, req, visual=True, evidence=evidence)
                for obs_id in result["refuted_observation_ids"]:
                    self.db.execute(
                        "INSERT OR REPLACE INTO refuted VALUES(?,?,?)",
                        (obs_id, req, conflict["reason"]),
                    )
                if result["resolved"] and not result["conflicts"]:
                    for record in record_pack:
                        if "node_id" in record:
                            try:
                                node = self.graph.get(record["node_id"])
                            except HarnessError:
                                continue
                            node["review_status"] = "resolved"
                            node["issues"] = []
                            node["history"].append(
                                {
                                    "pass": role,
                                    "operation": "review_resolved",
                                    "request_id": req,
                                }
                            )
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
        self.db.execute(
            "CREATE TEMP TABLE pending_conflicts AS SELECT * FROM conflicts"
        )
        self.db.execute("DELETE FROM conflicts")
        for row in self.db.execute("SELECT data FROM pending_conflicts ORDER BY id"):
            self.review_conflict(json.loads(row[0]))
        for row in self.db.execute("SELECT data FROM conflicts"):
            conflict = json.loads(row[0])
            for node_id in conflict.get("node_ids", []):
                self.mark_unresolved(node_id, conflict["reason"])

    def self_consistency(self):
        if not self.options["confidence"]["self_consistency_for_low_confidence"]:
            return
        for node in self.graph.rows():
            if "low" not in node["confidence"].values():
                continue
            answers, evidence_sets = [], []
            try:
                for replica in (0, 1):
                    answer, req, evidence = self.visual(
                        "independent_check",
                        node["start"],
                        node["end"],
                        {"target_description": node["title"]},
                        "Independently describe this target from the frames. "
                        'Do not assume the target description is correct. Return {"nodes":[...]} using the node schema.\n'
                        + NODE_SCHEMA,
                        lambda payload: self.parse_nodes(
                            payload, node["start"], node["end"]
                        ),
                        replicate=replica,
                    )
                    answers.append(answer)
                    evidence_sets.append(evidence)
            except HarnessError as exc:
                if not is_unusable_response(exc):
                    raise
                LOG.warning("independent_check: skipping unusable replica: %s", exc)
                self.mark_unresolved(node["node_id"], "independent_check_unusable")
                continue

            def parse(payload):
                if not isinstance(payload, dict) or not {
                    "confidence",
                    "reason",
                    "conflict",
                } <= set(payload):
                    raise HarnessError(
                        "Comparison requires confidence, reason and conflict"
                    )
                normalize_node(
                    {**node, "confidence": payload["confidence"]}, self.duration
                )
                if type(payload["conflict"]) is not bool or not isinstance(
                    payload["reason"], str
                ):
                    raise HarnessError("Invalid confidence comparison")
                return payload

            # Geometry is descriptive evidence, not a semantic matcher.
            iou = None
            differences = {}
            if len(answers[0]) == len(answers[1]) == 1:
                a, b = answers[0][0], answers[1][0]
                intersection = max(
                    0, min(a["end"], b["end"]) - max(a["start"], b["start"])
                )
                iou = intersection / (
                    a["end"] - a["start"] + b["end"] - b["start"] - intersection
                )
                differences = {
                    key: sorted(set(a[key]) ^ set(b[key]))
                    for key in ("actors", "actions", "objects")
                }
            payload = {
                "target": view(node),
                "answers": [[view(n) for n in answer] for answer in answers],
                "time_iou": iou,
                "literal_entity_differences": differences,
                "parent": view(self.graph.get(node["parent_id"]))
                if node["parent_id"]
                else None,
            }
            try:
                result, req = self.journal.request(
                    "confidence_comparison",
                    payload,
                    "Compare the two independent visual answers. Return {confidence:{semantic:high|medium|low,"
                    "boundary:high|medium|low,hierarchy:high|medium|low},reason:string,conflict:boolean}. "
                    "Agreement is not truth or calibrated probability. Assess each dimension separately; IoU or "
                    "equal wording alone cannot promote semantic or hierarchy confidence. Missing evidence remains low.",
                    parse,
                )
            except HarnessError as exc:
                if not is_unusable_response(exc):
                    raise
                LOG.warning(
                    "confidence_comparison: skipping unusable comparison: %s", exc
                )
                self.mark_unresolved(node["node_id"], "confidence_comparison_unusable")
                continue
            node["confidence"] = result["confidence"]
            node["evidence"] = combine_evidence(node["evidence"], *evidence_sets)
            node["history"].append(
                {
                    "pass": "confidence_comparison",
                    "operation": "confidence_review",
                    "request_id": req,
                    "time_iou": iou,
                    "entity_differences": differences,
                    "reason": result["reason"],
                }
            )
            self.graph.put(node)
            self.db.commit()
            if result["conflict"] or "low" in result["confidence"].values():
                self.queue_conflict(
                    {
                        "node_ids": [node["node_id"]],
                        "observation_ids": [],
                        "reason": result["reason"],
                    }
                )

    def coverage_review(self):
        self.link_duplicates()
        # Each unsupported observation gets one final independent visual check.
        for obs in self.graph.unsupported():
            overlapping = [
                n["node_id"]
                for n in self.graph.rows(start=obs["start"], end=obs["end"])
            ]
            self.review_conflict(
                {
                    "node_ids": overlapping,
                    "observation_ids": [obs["observation_id"]],
                    "reason": "Bottom-up observation has no corresponding final node",
                },
                role="coverage_review",
                max_rounds=1,
            )
            supported = self.db.execute(
                "SELECT 1 FROM support WHERE obs=?", (obs["observation_id"],)
            ).fetchone()
            refuted = self.db.execute(
                "SELECT 1 FROM refuted WHERE obs=?", (obs["observation_id"],)
            ).fetchone()
            if not supported and not refuted:
                node = deepcopy(obs)
                node.update(
                    node_id="n_unresolved_" + object_hash(obs["observation_id"])[:20],
                    parent_id=None,
                    granularity=0,
                    review_status="unresolved",
                    issues=["unexplained_bottomup_observation"],
                    confidence=dict.fromkeys(
                        ("semantic", "boundary", "hierarchy"), "low"
                    ),
                )
                node.pop("observation_id", None)
                node["history"].append(
                    {
                        "pass": "coverage_review",
                        "operation": "INSERT",
                        "reason": "Preserve unresolved evidence",
                    }
                )
                self.graph.put(node)
                self.db.execute(
                    "INSERT INTO edits(data) VALUES(?)",
                    (
                        canonical(
                            {
                                "kind": "coverage_fallback",
                                "operation": {
                                    "op": "INSERT",
                                    "observation_ids": [obs["observation_id"]],
                                },
                                "affected_ids": [node["node_id"]],
                                "review_status": "unresolved",
                            }
                        ).decode(),
                    ),
                )
                self.db.commit()
        self.link_duplicates()

    def coverage_rows(self):
        self.db.execute("DROP TABLE IF EXISTS coverage_edges")
        self.db.execute("CREATE TEMP TABLE coverage_edges(t REAL PRIMARY KEY)")
        self.db.executemany(
            "INSERT OR IGNORE INTO coverage_edges VALUES(?)", [(0,), (self.duration,)]
        )
        for table in ("nodes", "observations"):
            for field in ("start", "end"):
                self.db.execute(
                    f"INSERT OR IGNORE INTO coverage_edges SELECT {field} FROM {table}"
                )
        for row in self.db.execute("SELECT data FROM sampling"):
            sample = json.loads(row[0])
            self.db.executemany(
                "INSERT OR IGNORE INTO coverage_edges VALUES(?)",
                [(sample["start"],), (sample["end"],)],
            )
        previous = None
        for edge in self.db.execute("SELECT t FROM coverage_edges ORDER BY t"):
            end = edge[0]
            if previous is None:
                previous = end
                continue
            start = previous
            previous = end
            nodes = [n for n in self.graph.rows(start=start, end=end)]
            observations = [
                o
                for o in self.graph.rows(
                    start=start, end=end, observations=True, stage="bottom_up"
                )
            ]
            seen = {"top_down": False, "bottom_up": False}
            for record in self.db.execute("SELECT data FROM sampling"):
                sample = json.loads(record[0])
                if (
                    sample["pass"] in seen
                    and sample["start"] <= start
                    and sample["end"] >= end
                ):
                    seen[sample["pass"]] = True
            mappings = {
                obs["observation_id"]: [
                    r[0]
                    for r in self.db.execute(
                        "SELECT node FROM support WHERE obs=? ORDER BY node",
                        (obs["observation_id"],),
                    )
                ]
                for obs in observations
            }
            refuted = [
                o["observation_id"]
                for o in observations
                if self.db.execute(
                    "SELECT 1 FROM refuted WHERE obs=?", (o["observation_id"],)
                ).fetchone()
            ]
            unresolved = any(n["review_status"] == "unresolved" for n in nodes)
            yield {
                "start": start,
                "end": end,
                "topdown_seen": seen["top_down"],
                "bottomup_seen": seen["bottom_up"],
                "topdown_node_ids": [
                    r[0]
                    for r in self.db.execute(
                        "SELECT json_extract(data,'$.node_id') FROM facts "
                        "WHERE key LIKE 'h0:%' AND json_extract(data,'$.start')<? AND json_extract(data,'$.end')>?",
                        (end, start),
                    )
                ],
                "bottomup_observation_ids": list(mappings),
                "observation_support": mappings,
                "refuted_observation_ids": refuted,
                "final_node_count": len(nodes),
                "final_node_ids": [n["node_id"] for n in nodes],
                "review_status": "unresolved" if unresolved else "resolved",
                "seen_definition": "successful sampled-window coverage, not exhaustive event recall",
            }
