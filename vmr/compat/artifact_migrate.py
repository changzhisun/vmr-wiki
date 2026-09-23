"""Explicit legacy conversion, isolated from the Artifact reader and Query.

Only sealed, verified legacy input is accepted; nothing is changed in place.
"""

from pathlib import Path
import shutil
from vmr.core.errors import HarnessError
from vmr.core.jsonio import read_json
from vmr.core.hashing import object_hash
from vmr.artifact.store import ArtifactStore
from vmr.artifact.integrity import tree_hashes, remove_tree

# Legacy format policy belongs exclusively to this migration adapter.
LEGACY_TEXT = {
    "wiki.md",
    "nodes.jsonl",
    "observations.jsonl",
    "bottomup_observations.jsonl",
    "coverage.jsonl",
}


def migrate_legacy(root, store):
    root = Path(root)
    seal = read_json(root / "frozen.json")
    actual = tree_hashes(root, ("frozen.json",))
    if seal.get("files") != actual or seal.get("wiki_hash") != object_hash(actual):
        raise HarnessError("Legacy frozen wiki integrity check failed")
    metadata = read_json(root / "ingest.json")
    if metadata["video_id"] != seal["video_id"]:
        raise HarnessError("Legacy video identity mismatch")
    store = ArtifactStore(store)
    stage = store.staging()
    try:
        for name in actual:
            visible = (
                name in LEGACY_TEXT
                or name == "frames.jsonl"
                or name.startswith("frames/")
            )
            target = stage / ("public" if visible else "internal") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        if tree_hashes(root, ("frozen.json",)) != actual:
            raise HarnessError("Legacy input changed during migration")
        names = tree_hashes(stage / "public")
        text = sorted(names.keys() & LEGACY_TEXT)
        return store.publish(
            stage,
            source=dict(
                video_sha256=metadata["source_sha256"], duration_sec=seal["duration"]
            ),
            compiler=dict(
                name=metadata["ingest_config"].get("caption_mode", "simple"),
                version=1,
                content_config_hash="sha256:" + metadata["ingest_config_hash"],
            ),
            text_files=text,
            multimodal_files=sorted(names.keys() - set(text)),
            provenance=dict(
                migration="legacy-v1",
                legacy_wiki_hash=seal["wiki_hash"],
                build=metadata,
            ),
        )
    finally:
        if stage.exists():
            remove_tree(stage)
