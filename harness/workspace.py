from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from harness.common import HarnessError, RunFailure, file_hash, write_json
from harness.freeze import make_readonly, remove_tree, tree_hashes, verify_wiki
from harness.validate import validate_query


WIKI_TITLE = "# Video"


def require_anonymous_wiki(wiki_md: Path) -> None:
    """A wiki titled with its real video id is a lookup key into public data."""
    title = wiki_md.read_text(encoding="utf-8").split("\n", 1)[0].strip()
    if title != WIKI_TITLE:
        raise HarnessError(
            f"Wiki names its own video in the title ({title!r}); the agent would receive a "
            "public dataset identifier. This wiki predates identity-neutral rendering: "
            "re-ingest it into a new wiki root, or rewrite the title and its ingest.json "
            "content hash.")


@contextmanager
def query_workspace(query: dict, task: dict, wiki: Path, templates: dict[str, str], runs: Path):
    validate_query(query)
    seal = verify_wiki(wiki)
    if seal["video_id"] != query["video_id"]:
        raise HarnessError("Wiki does not belong to current query")
    require_anonymous_wiki(wiki / "wiki.md")
    runs.mkdir(parents=True, exist_ok=True)
    # The container can read its own bind mount source through /proc, so even
    # the host directory name is named after the alias rather than the query.
    root = Path(tempfile.mkdtemp(prefix=f"{task['query_id']}-", dir=runs)).resolve()
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
        write_json(root / "task.json", task)
        for name in ("AGENTS.md", "task.json"):
            (root / name).chmod(0o444)
        (root / "output").mkdir(mode=0o777)
        (root / "output").chmod(0o777)  # container runs as an unprivileged UID
        root.chmod(0o555)
        before = {name: file_hash(root / name) for name in ("AGENTS.md", "task.json")}
        yield root
        if tree_hashes(target) != expected or any(file_hash(root / k) != v for k, v in before.items()):
            raise RunFailure("Agent changed an immutable input", "tampered")
        verify_wiki(wiki)
    finally:
        remove_tree(root)

