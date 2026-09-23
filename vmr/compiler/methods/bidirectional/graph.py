import sqlite3
import hashlib
import json
from .model import *  # noqa: F403


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
        row = self.db.execute(
            "SELECT data FROM nodes WHERE id=?", (node_id,)
        ).fetchone()
        if row is None:
            raise HarnessError(f"Unknown node: {node_id}")
        return json.loads(row[0])

    def observation(self, observation_id):
        row = self.db.execute(
            "SELECT data FROM observations WHERE id=?", (observation_id,)
        ).fetchone()
        if row is None:
            raise HarnessError(f"Unknown observation: {observation_id}")
        return json.loads(row[0])

    def observe(self, row, stage):
        self.db.execute(
            "INSERT OR REPLACE INTO observations VALUES(?,?,?,?,?)",
            (
                row["observation_id"],
                stage,
                row["start"],
                row["end"],
                canonical(row).decode(),
            ),
        )
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
        sql = f"SELECT data FROM {table}" + (
            " WHERE " + " AND ".join(clauses) if clauses else ""
        )
        for row in self.db.execute(sql + " ORDER BY start,end,id", params):
            yield json.loads(row[0])

    def put(self, row):
        row = normalize_node(row, self.duration)
        node_id = nonempty(row["node_id"], "node_id")
        if (
            self.db.execute("SELECT 1 FROM nodes WHERE id=?", (node_id,)).fetchone()
            is None
            and self.db.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            >= self.max_nodes
        ):
            raise HarnessError("Bidirectional max_nodes exhausted")
        self.db.execute(
            "INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?)",
            (
                node_id,
                row["parent_id"],
                row["start"],
                row["end"],
                row["granularity"],
                canonical(row).decode(),
            ),
        )
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

    def exported(self):
        for row in self.rows():
            row["children"] = [
                item[0]
                for item in self.db.execute(
                    "SELECT id FROM nodes WHERE parent=? ORDER BY start,end,id",
                    (row["node_id"],),
                )
            ]
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

    def apply(
        self,
        operations,
        *,
        expected_version,
        request_id,
        visual=False,
        visual_evidence=None,
        dry_run=False,
    ):
        from .edits import GraphState, apply_edit

        state = GraphState(
            {n["node_id"]: n for n in self.rows()},
            {n["observation_id"]: n for n in self.rows(observations=True)},
            tuple(
                tuple(r) for r in self.db.execute("SELECT * FROM refuted ORDER BY obs")
            ),
            self.duration,
            self.max_nodes,
        )
        after, logs = apply_edit(
            state,
            operations,
            expected_version=expected_version,
            request_id=request_id,
            visual=visual,
            visual_evidence=visual_evidence,
        )
        if dry_run:
            return [
                {
                    k: v
                    for k, v in log.items()
                    if k not in ("before_version", "after_version")
                }
                for log in logs
            ]
        self.db.commit()
        self.db.execute("SAVEPOINT edits")
        try:
            self.db.execute("DELETE FROM support")
            self.db.execute("DELETE FROM nodes")
            for node in after.rows():
                self.put(node)
            for log in logs:
                self.db.execute(
                    "INSERT INTO edits(data) VALUES(?)", (canonical(log).decode(),)
                )
            self.db.execute("RELEASE edits")
            self.db.commit()
            return logs
        except BaseException:
            self.db.execute("ROLLBACK TO edits")
            self.db.execute("RELEASE edits")
            raise
