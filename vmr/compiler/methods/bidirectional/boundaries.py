"""Boundaries stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class BoundariesStage:
    def boundary(self, node_id):
        try:
            node = self.graph.get(node_id)
        except HarnessError:
            return
        options = self.options["boundary_refinement"]
        expand = (node["end"] - node["start"]) * options["context_expand_ratio"]
        intervals = [
            (
                max(0, node["start"] - expand),
                min(self.duration, node["end"] + expand),
                "both",
            )
        ]
        for round_index in range(options["max_rounds"]):
            next_intervals = []
            for start, end, endpoint in intervals:

                def parse(value):
                    if not isinstance(value, dict) or not {
                        "visible",
                        "start",
                        "end",
                        "boundary_uncertainty",
                        "before",
                        "after",
                    } <= set(value):
                        raise HarnessError(
                            "Boundary answer requires visible, start/end, uncertainty, before/after"
                        )
                    if type(value["visible"]) is not bool:
                        raise HarnessError("visible must be boolean")
                    if value["visible"]:
                        updates = {
                            k: value[k]
                            for k in ("start", "end", "boundary_uncertainty")
                        }
                        normalize_node({**node, **updates}, self.duration)
                        for key in ("before", "after"):
                            if value[key] is not None and (
                                type(value[key]) not in (int, float)
                                or not start <= value[key] <= end
                            ):
                                raise HarnessError(
                                    "Before/after evidence must be visible timestamps or null"
                                )
                        checked = (
                            ("start", "end") if endpoint == "both" else (endpoint,)
                        )
                        if any(not start <= value[k] <= end for k in checked):
                            raise HarnessError(
                                "Refined endpoint lies outside its inspected interval"
                            )
                    return value

                try:
                    result, req, evidence = self.visual(
                        "boundary_refinement",
                        start,
                        end,
                        {
                            "target": view(node),
                            "endpoint": endpoint,
                            "round": round_index,
                        },
                        "Find earliest clear beginning, latest ongoing timestamp and immediately before/after evidence. "
                        "Return {visible:boolean,start:seconds,end:seconds,boundary_uncertainty:{start:[lo,hi],end:[lo,hi]},"
                        "before:timestamp_or_null,after:timestamp_or_null}. If an endpoint is not visible, retain its "
                        "estimate and uncertainty. When inspecting only one endpoint keep the other unchanged. "
                        "Never invent frame-accurate timing from sparse frames. "
                        "boundary_uncertainty must satisfy 0 <= lo <= the endpoint <= hi <= video duration.",
                        parse,
                    )
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning(
                        "boundary_refinement: unusable response retained as unresolved: %s",
                        exc,
                    )
                    self.mark_unresolved(node_id, "boundary_response_unusable")
                    continue
                if not result["visible"]:
                    self.mark_unresolved(node_id, "boundary_not_visible")
                    continue
                updates = {
                    k: result[k] for k in ("start", "end", "boundary_uncertainty")
                }
                if endpoint != "both":
                    other = "end" if endpoint == "start" else "start"
                    updates[other] = node[other]
                    updates["boundary_uncertainty"][other] = node[
                        "boundary_uncertainty"
                    ][other]
                self.graph.apply(
                    [
                        {
                            "op": "SHIFT",
                            "node_ids": [node_id],
                            "observation_ids": [],
                            "updates": updates,
                            "reason": "Locally reinspected boundary",
                        }
                    ],
                    expected_version=self.graph.version,
                    request_id=req,
                    visual=True,
                    visual_evidence=evidence,
                )
                node = self.graph.get(node_id)
                for key in ("start", "end") if endpoint == "both" else (endpoint,):
                    lo, hi = node["boundary_uncertainty"][key]
                    if hi - lo > options["uncertainty_sec"]:
                        radius = options["endpoint_context_sec"]
                        next_intervals.append(
                            (
                                max(0, node[key] - radius),
                                min(self.duration, node[key] + radius),
                                key,
                            )
                        )
            intervals = next_intervals
            if not intervals:
                break

    def boundaries(self):
        if not self.options["boundary_refinement"]["enabled"]:
            return
        for node in self.graph.rows():
            if node["confidence"]["boundary"] == "low" or any(
                hi - lo > self.options["boundary_refinement"]["uncertainty_sec"]
                for lo, hi in node["boundary_uncertainty"].values()
            ):
                self.db.execute(
                    "INSERT OR IGNORE INTO boundary_queue(id) VALUES(?)",
                    (node["node_id"],),
                )
        for row in self.db.execute(
            "SELECT id FROM boundary_queue WHERE done=0 ORDER BY id"
        ):
            self.boundary(row[0])
            self.db.execute("UPDATE boundary_queue SET done=1 WHERE id=?", (row[0],))
        self.db.commit()
