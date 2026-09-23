"""Render stage; shares the explicit builder context."""

from .support import *  # noqa: F403


class RenderStage:
    def render(self):
        def prose(value):
            return (
                " ".join(str(value).split()).replace("<", "&lt;").replace(">", "&gt;")
            )

        with (self.staging / "wiki.md").open("w", encoding="utf-8") as stream:
            stream.write(
                f"# Video\n\n## Metadata\n\n- Duration: {self.duration} seconds\n- Caption mode: bidirectional\n"
                "- Schema version: 2\n- nodes.jsonl is the main graph; bottomup_observations.jsonl retains blind evidence.\n"
                "- Coverage means successfully processed sampling windows, not guaranteed event recall.\n"
                "- Confidence labels are evidence assessments, not probabilities.\n\n## Navigation\n\n"
            )
            for node in self.graph.rows():
                if node["parent_id"] is None:
                    stream.write(
                        f"- `{node['node_id']}` {node['start']:g}–{node['end']:g}s: {prose(node['title'])}\n"
                    )
            stream.write("\n## Temporal hierarchy\n\n")

            # Recursive traversal streams one node at a time; only IDs of the
            # current ancestry are held, not the full long-video tree.
            def visit(node):
                heading = "#" * min(6, node["granularity"] + 3)
                stream.write(
                    f"{heading} {node['start']:g}–{node['end']:g}s · {prose(node['title'])} `{node['node_id']}`\n\n"
                )
                if node["review_status"] == "unresolved":
                    stream.write(
                        "**UNRESOLVED — candidate requiring evidence inspection.** "
                        + prose(", ".join(node["issues"]))
                        + "\n\n"
                    )
                stream.write(prose(node["summary"]) + "\n\n")
                stream.write(
                    f"- Granularity: {node['granularity']}; type: {node['type']}; parent: {node['parent_id']}\n"
                )
                for key in (
                    "actors",
                    "actions",
                    "objects",
                    "state_before",
                    "state_after",
                    "confidence",
                    "boundary_uncertainty",
                    "relations",
                ):
                    stream.write(f"- {key}: {prose(node[key])}\n")
                stream.write(
                    "- Evidence: see this node's observation/frame IDs in nodes.jsonl and frames.jsonl.\n\n"
                )
                for child in self.db.execute(
                    "SELECT id FROM nodes WHERE parent=? ORDER BY start,end,id",
                    (node["node_id"],),
                ):
                    visit(self.graph.get(child[0]))

            for row in self.db.execute(
                "SELECT id FROM nodes WHERE parent IS NULL ORDER BY start,end,id"
            ):
                visit(self.graph.get(row[0]))
