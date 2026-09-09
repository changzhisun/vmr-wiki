"""Private, atomic window checkpoints; never mounted into query workspaces."""
from pathlib import Path

from harness.common import HarnessError, file_hash, object_hash, read_json, write_json


class IngestCheckpoint:
    def __init__(self, output: Path, identity: dict):
        # Outside videos/: incomplete checkpoints cannot look like wiki outputs.
        self.root = output.parent.parent / ".ingest-checkpoints" / output.name / object_hash(identity)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        path = self.root / "identity.json"
        if path.exists() and read_json(path) != identity:
            raise HarnessError(f"Checkpoint identity mismatch: {path}")
        write_json(path, identity)

    def read(self, name: str):
        path = self.root / name
        if not path.exists():
            return None
        envelope = read_json(path)
        if not isinstance(envelope, dict) or "data" not in envelope:
            raise HarnessError(f"Checkpoint changed: {path}")
        data = envelope.get("data")
        if envelope.get("sha256") != object_hash(data):
            raise HarnessError(f"Checkpoint changed: {path}")
        return data

    def write(self, name: str, data):
        write_json(self.root / name, {"data": data, "sha256": object_hash(data)})

    def image(self, frame: dict) -> Path | None:
        relative = frame["frame"]
        path = self.root / relative
        record = self.read(relative + ".json")
        if record is None:
            return None
        if record["frame"] != frame or not path.is_file() or file_hash(path) != record["sha256"]:
            raise HarnessError(f"Checkpoint frame changed: {path}")
        return path

    def seal_image(self, frame: dict, elapsed_sec: float):
        self.write(frame["frame"] + ".json", {
            "frame": frame, "sha256": file_hash(self.root / frame["frame"]),
            "elapsed_sec": elapsed_sec,
        })

    def record_attempt(self, window_id: str, attempt: dict):
        directory = self.root / "attempts" / window_id
        directory.mkdir(parents=True, exist_ok=True)
        # Each video is guarded by the ingest lock; no concurrent writers here.
        count = max((int(path.stem) for path in directory.glob("*.json")), default=-1) + 1
        self.write(f"attempts/{window_id}/{count:06d}.json", attempt)

    def attempts(self, window_id: str) -> list[dict]:
        return [self.read(path.relative_to(self.root).as_posix())
                for path in sorted((self.root / "attempts" / window_id).glob("*.json"))]
