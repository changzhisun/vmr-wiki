"""Topdown stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class TopdownStage:
    def topdown(self, start, end, depth=0, parent=None, short_used=False):
        opts = self.options["topdown"]
        data = {"granularity": depth, "max_children": opts["max_children_per_node"]}
        if parent:
            data["parent_context"] = view(parent)
        instruction = (
            "Identify coarse phases."
            if parent is None
            else "Decompose this semantic interval."
        ) + (
            ' Return {"nodes":[...]} with chronological nodes, each using the schema below. '
            "Set needs_refinement and semantic_complexity based on visible evidence; multiple_actions=true "
            "only for several distinct sequential actions. boundary_uncertainty.start/end must satisfy "
            "0 <= lo <= the endpoint <= hi <= video duration; never copy example timestamps. "
            "Granularity is a temporal scale, never a fixed ontology. Overlaps/gaps are allowed; static regions "
            "may remain covered by a broader parent. An indivisible parent may return no children. "
            "The global scan must return at least one coarse phase.\n" + NODE_SCHEMA
        )
        nodes, req, evidence = self.visual(
            "top_down",
            start,
            end,
            data,
            instruction,
            lambda payload: self.parse_nodes(
                payload,
                start,
                end,
                limit=opts["max_children_per_node"],
                require=parent is None,
                parent=parent,
            ),
        )
        for index, raw in enumerate(nodes):
            raw["granularity"] = depth
            raw["parent_id"] = parent["node_id"] if parent else None
            obs = self.observe(raw, "top_down", req, index, evidence)
            node = {**obs, "node_id": "n_" + object_hash(obs["observation_id"])[:24]}
            node.pop("observation_id")
            self.graph.put(node)
            self.db.commit()
            short = node["end"] - node["start"] <= opts["min_segment_duration_sec"]
            can_split = node["needs_refinement"] and (
                not short or (node["multiple_actions"] and not short_used)
            )
            if depth + 1 < opts["max_depth"] and can_split:
                self.topdown(
                    node["start"], node["end"], depth + 1, node, short_used or short
                )
