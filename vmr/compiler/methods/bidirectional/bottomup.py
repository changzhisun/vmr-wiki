"""Bottomup stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class BottomupStage:
    def bottomup(self):
        opts = self.options["bottomup"]
        for window_index, (start, end) in enumerate(
            windows(self.duration, opts["window_sec"], opts["overlap_ratio"])
        ):
            # Blind means a new request with no previous answers, H0 labels,
            # ancestors, observations, or model session in this input.
            data = {}
            if not opts["independent_observation"]:
                # Experimental contextual control. Separate bounded requests
                # carry all overlapping H0 context if it exceeds one batch.
                contexts = self.packs(
                    view(n) for n in self.graph.rows(start=start, end=end)
                )
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
        nodes, req, evidence = self.visual(
            "bottom_up",
            start,
            end,
            data,
            "Independently enumerate ALL significant visible local actions, events and state changes in time order. "
            "Pay special attention to brief actions, entry/exit, object transfers and state changes. "
            "Do not invent activity in quiet frames; an empty array is valid. Do not guess unseen details. "
            'Return {"nodes":[...]} using this schema; granularity is provisional, not an ontology.\n'
            + NODE_SCHEMA,
            lambda payload: self.parse_nodes(payload, start, end),
        )
        for index, node in enumerate(nodes):
            node["window_index"] = window_index
            obs = self.observe(node, "bottom_up", req, index, evidence)
            self.db.execute(
                "INSERT INTO groups VALUES(?,?)",
                (obs["observation_id"], obs["observation_id"]),
            )
        self.db.commit()

    def deduplicate(self):
        # Only time-near observations from adjacent overlapping windows are
        # candidates. Semantics are decided by the VLM, never by overlap alone.
        for left in self.graph.rows(observations=True, stage="bottom_up"):
            candidates = (
                right
                for right in self.graph.rows(
                    observations=True,
                    stage="bottom_up",
                    start=max(0, left["start"] - 2),
                    end=min(self.duration, left["end"] + 2),
                )
                if right["window_index"] == left["window_index"] + 1
            )
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
                        if (
                            not isinstance(item, dict)
                            or item.get("observation_id") not in allowed
                            or item.get("relation")
                            not in {"same_event", "continuation", "different_event"}
                            or not isinstance(item.get("reason"), str)
                            or not item["reason"].strip()
                        ):
                            raise HarnessError("Invalid local comparison")
                        seen.append(item["observation_id"])
                    if len(seen) != len(set(seen)) or set(seen) != allowed:
                        raise HarnessError("Compare every candidate exactly once")
                    return comparisons

                comparisons, req = self.journal.request(
                    "local_dedup",
                    {"left": view(left), "candidates": pack},
                    "Compare each candidate to the left observation. Decide same_event, continuation, or different_event. "
                    'Repeated but separate occurrences are different_event. Return {"comparisons":[{"observation_id":...,'
                    '"relation":...,"reason":...}]}. Never assume time overlap means the same event.',
                    parse,
                )
                for item in comparisons:
                    pair = [left["observation_id"], item["observation_id"]]
                    if item["relation"] == "same_event":
                        leaders = [
                            self.db.execute(
                                "SELECT leader FROM groups WHERE obs=?", (o,)
                            ).fetchone()[0]
                            for o in pair
                        ]
                        leader = min(leaders)
                        for old in leaders:
                            self.db.execute(
                                "UPDATE groups SET leader=? WHERE leader=?",
                                (leader, old),
                            )
                    self.db.execute(
                        "INSERT INTO edits(data) VALUES(?)",
                        (
                            canonical(
                                {
                                    "request_id": req,
                                    "kind": "observation_dedup",
                                    "relation": item["relation"],
                                    "operation": {
                                        "op": "MERGE"
                                        if item["relation"] == "same_event"
                                        else "KEEP",
                                        "observation_ids": pair,
                                        "reason": item["reason"],
                                    },
                                }
                            ).decode(),
                        ),
                    )
                self.db.commit()
