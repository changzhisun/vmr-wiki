"""One-time migration: drop the video id from an existing wiki's title.

``render_wiki`` used to title the timeline ``# Video: <video_id>``, which handed
the agent an exact lookup key into public benchmark data. Only that one line
changes: the sampled frames and the VLM captions are untouched, so this rewrites
the title in place and updates the recorded content hash instead of paying to
re-caption every video. The captions are provably identical because every other
entry in ``content_hashes`` must still match before anything is written.

Frozen wikis are refused. Remove ``frozen.json`` and re-freeze after migrating,
so the seal attests the content that actually exists.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from harness.common import HarnessError, atomic_text, cli, identifier, write_json, read_json
from harness.config import dataset_path, load_config
from harness.freeze import tree_hashes
from harness.workspace import WIKI_TITLE


def migrate_wiki(root: Path) -> bool:
    """Return True when this wiki was rewritten, False when already anonymous."""
    if (root / "frozen.json").exists():
        raise HarnessError(f"{root.name}: remove frozen.json and re-freeze after migrating")
    metadata = read_json(root / "ingest.json")
    recorded = metadata["content_hashes"]
    actual = tree_hashes(root, ("ingest.json",))
    if {k: v for k, v in recorded.items() if k != "ingest.json"} != actual:
        raise HarnessError(f"{root.name}: content already differs from ingest.json; not migrating")
    wiki_md = root / "wiki.md"
    title, _, body = wiki_md.read_text(encoding="utf-8").partition("\n")
    if title.strip() == WIKI_TITLE:
        return False
    if not title.startswith(f"{WIKI_TITLE}:"):
        raise HarnessError(f"{root.name}: unexpected wiki title {title!r}")
    atomic_text(wiki_md, f"{WIKI_TITLE}\n{body}")
    metadata["content_hashes"] = {**tree_hashes(root, ("ingest.json",)),
                                  **{k: v for k, v in recorded.items() if k == "ingest.json"}}
    metadata["migrations"] = [*metadata.get("migrations", []), "anonymous_wiki_title"]
    write_json(root / "ingest.json", metadata)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--apply", action="store_true", help="Without this, only report what would change")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = identifier(args.dataset)
    root = dataset_path(cfg, "wiki") / "videos"
    if not root.is_dir():
        raise HarnessError(f"No wiki directory: {root}")
    pending = sorted(p.parent for p in root.glob("*/wiki.md")
                     if p.read_text(encoding="utf-8").split("\n", 1)[0].strip() != WIKI_TITLE)
    if not args.apply:
        print(f"{len(pending)} wikis would be retitled; rerun with --apply")
        return
    changed = sum(migrate_wiki(directory) for directory in pending)
    print(f"Retitled {changed} wikis; captions unchanged")


if __name__ == "__main__":
    cli(main)
