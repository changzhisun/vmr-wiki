from .support import *  # noqa: F403
from .topdown import TopdownStage
from .bottomup import BottomupStage
from .reconcile import ReconcileStage
from .boundaries import BoundariesStage
from .review import ReviewStage
from .render import RenderStage


class BidirectionalBuilder(
    TopdownStage,
    BottomupStage,
    ReconcileStage,
    BoundariesStage,
    ReviewStage,
    RenderStage,
):
    def __init__(self, video, duration, cfg, checkpoint, staging, client, check):
        self.duration, self.cfg, self.checkpoint, self.staging, self.check = (
            duration,
            cfg,
            checkpoint,
            staging,
            check,
        )
        self.options = settings(cfg["ingest"])
        self.limits = budgets(self.options, duration)
        self.graph = TemporalGraph(
            checkpoint.root / "working.sqlite", duration, self.limits["max_nodes"]
        )
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
            self.frames = FrameIndex(
                self.db, video, duration, checkpoint, staging, cfg["ingest"], check
            )
        except BaseException:
            self.graph.close()
            raise
        self.journal = RequestJournal(
            client, checkpoint, self.db, cfg, self.options, self.limits, check
        )
        self.cap = self.options["max_frames_per_call"]
        self.batch = self.options["reconciliation"]

    def packs(self, rows):
        return batches(rows, self.batch["max_records"], self.batch["max_chars"] - 6000)

    def visual(self, role, start, end, data, instruction, parser, *, replicate=None):
        self.journal.ensure_budget()
        frames = self.frames.images(start, end, self.cap)
        payload = {
            "start": start,
            "end": end,
            "frame_timestamps": [f["timestamp"] for f in frames],
            **data,
        }
        result, request_id = self.journal.request(
            role, payload, instruction, parser, frames=frames, replicate=replicate
        )
        return result, request_id, self.frames.evidence(role, frames)

    def parse_nodes(
        self, payload, start, end, *, limit=None, require=False, parent=None
    ):
        if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
            raise HarnessError("Response must contain a nodes array")
        if require and not payload["nodes"]:
            raise HarnessError("Global scan requires at least one coarse phase")
        if limit is not None and len(payload["nodes"]) > limit:
            raise HarnessError("Too many children in one response")
        rows = []
        for raw in payload["nodes"]:
            node = normalize_node(raw, self.duration)
            clipped = clip_to_range(node["start"], node["end"], start, end)
            if clipped is None:
                LOG.warning(
                    "Dropped node %r [%g, %g]: no overlap with request window [%g, %g]",
                    node.get("title", ""),
                    node["start"],
                    node["end"],
                    start,
                    end,
                )
                continue
            if clipped != (node["start"], node["end"]):
                LOG.warning(
                    "Clipped node %r from [%g, %g] to request window [%g, %g] -> [%g, %g]",
                    node.get("title", ""),
                    node["start"],
                    node["end"],
                    start,
                    end,
                    clipped[0],
                    clipped[1],
                )
                node["start"], node["end"] = clipped
                node = normalize_node(node, self.duration)
            if parent and (
                node["start"],
                node["end"],
                node["title"],
                node["summary"],
            ) == (parent["start"], parent["end"], parent["title"], parent["summary"]):
                raise HarnessError("Recursive subdivision made no progress")
            rows.append(node)
        if require and not rows:
            raise HarnessError("Global scan requires at least one coarse phase")
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
        selected = [
            i
            for i, stamp in enumerate(evidence["frame_timestamps"])
            if row["start"] <= stamp <= row["end"]
        ]
        if not selected and evidence["frame_timestamps"]:
            selected = sorted(
                range(len(evidence["frame_timestamps"])),
                key=lambda i: min(
                    abs(evidence["frame_timestamps"][i] - row["start"]),
                    abs(evidence["frame_timestamps"][i] - row["end"]),
                ),
            )[:2]
        row["evidence"] = {
            "pass": [stage],
            "observation_ids": [obs_id],
            "frame_ids": [evidence["frame_ids"][i] for i in selected],
            "frame_timestamps": [evidence["frame_timestamps"][i] for i in selected],
        }
        row["history"] = [
            {"pass": stage, "operation": "created", "request_id": request_id}
        ]
        self.graph.observe(row, stage)
        return row

    def build(self):
        opts = self.options["bottomup"]
        schedule = (
            list(windows(self.duration, opts["window_sec"], opts["overlap_ratio"]))
            if opts["enabled"]
            else []
        )
        preflight = {
            "bottomup_windows": len(schedule),
            "bottomup_image_inputs": sum(
                len(self.frames.times(a, b, self.cap)) for a, b in schedule
            ),
            **self.limits,
        }
        LOG.info("Bidirectional preflight: %s", preflight)
        if len(schedule) > self.limits["max_requests"]:
            raise HarnessError("Budget cannot cover the mandatory bottom-up sweep")
        write_json(self.staging / "preflight.json", preflight)
        if self.options["topdown"]["enabled"]:
            self.topdown(0, self.duration)
        for node in self.graph.exported():
            self.db.execute(
                "INSERT INTO facts VALUES(?,?)",
                ("h0:" + node["node_id"], canonical(node).decode()),
            )
        stream_jsonl(self.staging / "topdown_nodes.jsonl", self.graph.exported())
        if opts["enabled"]:
            self.bottomup()
            self.deduplicate()
            self.reconcile(
                "reconciliation"
                if self.options["topdown"]["enabled"]
                else "bottomup_organization"
            )
            self.link_duplicates()
            self.targeted_reviews()
        self.boundaries()
        self.reconcile("consistency_review", include_observations=False)
        # Separately inspect every cross-reference and primary parent edge,
        # including those spanning different text/time batches.
        for node in self.graph.rows():
            related = ([node["parent_id"]] if node["parent_id"] else []) + [
                r["target_id"] for r in node["relations"]
            ]
            for target in dict.fromkeys(related):
                records = [view(node), view(self.graph.get(target))]
                try:
                    response, req = self.journal.request(
                        "relation_review",
                        {"records": records, "graph_version": self.graph.version},
                        EDIT_INSTRUCTION
                        + "\nReview this cross-reference or parent-child edge, especially state and temporal consistency.",
                        lambda payload: self.parse_edits(
                            payload, records, "relation_review"
                        ),
                    )
                except HarnessError as exc:
                    if not is_unusable_response(exc):
                        raise
                    LOG.warning("relation_review: skipping unusable edge: %s", exc)
                    continue
                self.apply_response(response, req)
        for node_id, reason in self.graph.issues():
            self.mark_unresolved(node_id, reason)
            self.queue_conflict(
                {"node_ids": [node_id], "observation_ids": [], "reason": reason}
            )
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
        stream_jsonl(
            self.staging / "observations.jsonl", self.graph.rows(observations=True)
        )
        stream_jsonl(
            self.staging / "bottomup_observations.jsonl",
            self.graph.rows(observations=True, stage="bottom_up"),
        )
        stream_jsonl(
            self.staging / "frames.jsonl",
            (
                json.loads(r[0])
                for r in self.db.execute("SELECT data FROM frames ORDER BY timestamp")
            ),
        )
        stream_jsonl(
            self.staging / "sampling.jsonl",
            (
                json.loads(r[0])
                for r in self.db.execute("SELECT data FROM sampling ORDER BY seq")
            ),
        )
        stream_jsonl(
            self.staging / "reconciliation.jsonl",
            (
                json.loads(r[0])
                for r in self.db.execute("SELECT data FROM edits ORDER BY seq")
            ),
        )
        stream_jsonl(self.staging / "caption_audit.jsonl", self.journal.audits())
        stream_jsonl(self.staging / "coverage.jsonl", self.coverage_rows())
        self.render()
        stats = self.journal.telemetry()
        unresolved = self.db.execute(
            "SELECT COUNT(*) FROM nodes WHERE json_extract(data,'$.review_status')='unresolved'"
        ).fetchone()[0]
        return {
            **stats,
            "preflight": preflight,
            "extraction_sec_this_run": self.frames.extraction_sec,
            "extraction_sec": self.db.execute(
                "SELECT COALESCE(SUM(elapsed),0) FROM frames"
            ).fetchone()[0],
            "reused_frames": self.frames.reused_frames,
            "unresolved_nodes": unresolved,
            "node_count": self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "frame_count": self.db.execute("SELECT COUNT(*) FROM frames").fetchone()[0],
            "inserted_nodes": self.db.execute(
                "SELECT COUNT(*) FROM edits WHERE json_extract(data,'$.operation.op')='INSERT'"
            ).fetchone()[0],
            "refuted_observations": self.db.execute(
                "SELECT COUNT(*) FROM refuted"
            ).fetchone()[0],
        }


def publish_bidirectional(
    video,
    video_id,
    output,
    staging,
    checkpoint,
    client,
    cfg,
    duration,
    video_stream_duration,
    source_hash,
    content_hash,
    ffmpeg_version,
    check,
):
    builder = BidirectionalBuilder(
        video, video_stream_duration, cfg, checkpoint, staging, client, check
    )
    try:
        telemetry = builder.build()
    finally:
        builder.graph.close()
    if file_hash(video) != source_hash:
        raise HarnessError("Source video changed during ingest")
    check()
    metadata = {
        "version": 1,
        "video_id": video_id,
        "duration": video_stream_duration,
        "container_duration": duration,
        "video_stream_duration": video_stream_duration,
        "source_sha256": source_hash,
        "ingest_config": cfg["ingest"],
        "ingest_config_hash": content_hash,
        "created_at": now(),
        "ffmpeg_version": ffmpeg_version,
        "schema_version": SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "review_status": "unresolved" if telemetry["unresolved_nodes"] else "resolved",
        "telemetry": telemetry,
        "content_hashes": tree_hashes(staging),
    }
    write_json(staging / cfg.get("build_record", "ingest.json"), metadata)
    staging.rename(output)
    try:
        remove_tree(checkpoint.root)
    except OSError:
        LOG.warning("Published Wiki; could not remove checkpoint: %s", checkpoint.root)
    return metadata
