"""Experiment persistence and resume guard, separate from execution."""

from vmr.core.errors import HarnessError
from vmr.core.jsonio import read_json, write_json, atomic_text
from vmr.core.time import now
from .aliases import new_secret


class RunRepository:
    def __init__(self, root):
        self.root = root
        self.lock = root / ".experiment.lock"

    def enter(self, metadata, snapshot, templates):
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            with self.lock.open("x") as stream:
                stream.write(now())
        except FileExistsError as exc:
            raise HarnessError(
                "Experiment already running (or stale .experiment.lock)"
            ) from exc
        try:
            path = self.root / "experiment.json"
            if path.exists():
                saved = read_json(path)
                if saved.get("version") != 3 or not (
                    "artifacts" in saved or "video_hashes" in saved
                ):
                    raise HarnessError(
                        "Legacy experiment cannot resume; create a new experiment"
                    )
                metadata["alias_secret"] = saved["alias_secret"]
                if metadata != saved:
                    raise HarnessError(
                        "Experiment inputs/config/agent/code changed; use a new experiment"
                    )
                saved_snapshot = read_json(self.root / "query-config.json")
                if {
                    k: v for k, v in saved_snapshot.items() if k != "source_commit"
                } != {k: v for k, v in snapshot.items() if k != "source_commit"}:
                    raise HarnessError("Saved query configuration changed")
                for name, text in templates.items():
                    if (self.root / "templates" / name).read_text(
                        encoding="utf-8"
                    ) != text:
                        raise HarnessError("Saved experiment template changed")
            else:
                if any(p.name != self.lock.name for p in self.root.iterdir()):
                    raise HarnessError(
                        "Experiment directory contains untracked results"
                    )
                metadata["alias_secret"] = new_secret()
                write_json(path, metadata)
                write_json(self.root / "query-config.json", snapshot)
                for name, text in templates.items():
                    atomic_text(self.root / "templates" / name, text)
            for name in ("predictions", "run_metadata", "logs"):
                (self.root / name).mkdir(exist_ok=True)
        except BaseException:
            self.close()
            raise

    def close(self):
        self.lock.unlink(missing_ok=True)
