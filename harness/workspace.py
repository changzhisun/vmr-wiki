from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from harness.common import HarnessError, file_hash, write_json
from harness.freeze import make_readonly, remove_tree, tree_hashes, verify_wiki
from harness.validate import validate_query


@contextmanager
def query_workspace(query: dict, wiki: Path, templates: dict[str, str],
                    runs: Path, max_predictions: int):
    validate_query(query)
    seal = verify_wiki(wiki)
    if seal["video_id"] != query["video_id"]:
        raise HarnessError("Wiki does not belong to current query")
    runs.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"{query['query_id']}-", dir=runs)).resolve()
    try:
        target = root / "wiki"
        target.mkdir()
        for name in ("wiki.md", "frames.jsonl"):
            shutil.copyfile(wiki / name, target / name)
        shutil.copytree(wiki / "frames", target / "frames")
        expected = {key: value for key, value in seal["files"].items()
                    if key in ("wiki.md", "frames.jsonl") or key.startswith("frames/")}
        if tree_hashes(target) != expected:
            raise HarnessError("Wiki changed while creating workspace")
        make_readonly(target)
        (root / "AGENTS.md").write_text(templates["AGENTS.md"], encoding="utf-8")
        write_json(root / "task.json", {**query, "max_predictions": max_predictions})
        for name in ("AGENTS.md", "task.json"):
            (root / name).chmod(0o444)
        (root / "output").mkdir(mode=0o777)
        (root / "output").chmod(0o777)  # container runs as an unprivileged UID
        root.chmod(0o555)
        before = {name: file_hash(root / name) for name in ("AGENTS.md", "task.json")}
        yield root
        if tree_hashes(target) != expected or any(file_hash(root / k) != v for k, v in before.items()):
            raise HarnessError("Agent changed an immutable input")
        verify_wiki(wiki)
    finally:
        remove_tree(root)

