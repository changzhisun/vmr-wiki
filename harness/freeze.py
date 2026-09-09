from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
from pathlib import Path

from harness.common import (HarnessError, cli, file_hash, ingest_content_hash,
                            object_hash, read_json, write_json)
from harness.config import dataset_path, load_config
from harness.dataset import dataset_context, load_videos


def tree_hashes(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise HarnessError(f"Expected a real directory: {root}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise HarnessError(f"Symlink is forbidden in frozen input: {relative}")
        if path.is_file() and relative not in exclude:
            hashes[relative] = file_hash(path)
        elif not path.is_dir() and not path.is_file():
            raise HarnessError(f"Non-regular input: {relative}")
    return hashes


def make_readonly(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def remove_tree(root: Path) -> None:
    """Remove our own temporary copies, including their read-only directories."""
    if not root.exists():
        return
    root.chmod(0o700)
    for path in root.rglob("*"):
        if not path.is_symlink() and path.is_dir():
            path.chmod(0o700)
    shutil.rmtree(root)


def _format_ids(video_ids: list[str], limit: int = 8) -> str:
    shown = ", ".join(video_ids[:limit])
    if len(video_ids) > limit:
        shown += f" ... (+{len(video_ids) - limit})"
    return shown


def wiki_readiness(root: Path, video_ids) -> tuple[list[str], list[str]]:
    """Return ``(not_ingested, ingested_but_not_frozen)`` video ids under ``root``."""
    missing, unfrozen = [], []
    for vid in video_ids:
        wiki = root / vid
        if not (wiki / "ingest.json").is_file():
            missing.append(vid)
        elif not (wiki / "frozen.json").is_file():
            unfrozen.append(vid)
    return missing, unfrozen


def split_not_ready_error(dataset: str, split: str, missing: list[str],
                          unfrozen: list[str]) -> HarnessError:
    parts = []
    if missing:
        parts.append(f"{len(missing)} not ingested ({_format_ids(missing)})")
    if unfrozen:
        parts.append(f"{len(unfrozen)} ingested but not frozen")
    return HarnessError(
        f"Split {split!r} is not ready for queries: {'; '.join(parts)}. "
        f"Finish ingest, then run: python harness/freeze.py --dataset {dataset} --split {split}"
    )


def verify_wiki(root: Path) -> dict:
    if not (root / "frozen.json").is_file():
        if (root / "ingest.json").is_file():
            raise HarnessError(
                f"Wiki is ingested but not frozen: {root}. "
                "Run python harness/freeze.py with the same --dataset and --split."
            )
        raise HarnessError(
            f"Frozen wiki not found: {root}. Ingest this video, then freeze the split."
        )
    seal = read_json(root / "frozen.json")
    actual = tree_hashes(root, ("frozen.json",))
    if seal.get("files") != actual or seal.get("wiki_hash") != object_hash(actual):
        raise HarnessError(f"Frozen wiki integrity check failed: {root}")
    return seal


def freeze_wiki(root: Path) -> dict:
    if (root / "frozen.json").exists():
        return verify_wiki(root)
    metadata = read_json(root / "ingest.json")
    files = tree_hashes(root)
    expected = metadata["content_hashes"]
    if {k: v for k, v in files.items() if k != "ingest.json"} != expected:
        raise HarnessError(f"Wiki changed since ingest: {root}")
    seal = {
        "version": 1, "video_id": metadata["video_id"],
        "duration": metadata["duration"], "ingest_config_hash": metadata["ingest_config_hash"],
        "wiki_hash": object_hash(files), "wiki_md_hash": files["wiki.md"],
        "frames_jsonl_hash": files["frames.jsonl"], "files": files,
    }
    write_json(root / "frozen.json", seal)
    make_readonly(root)
    return seal


def freeze_dataset(cfg: dict, *, split: str | None = None) -> dict:
    directory, dataset, split = dataset_context(cfg, split)
    root = dataset_path(cfg, "wiki") / "videos"
    videos = load_videos(directory, dataset, split)
    if not videos:
        raise HarnessError("No videos to freeze")
    missing, _ = wiki_readiness(root, videos)
    if missing:
        raise HarnessError(
            f"{len(missing)} video(s) have no ingest and cannot be frozen: {_format_ids(missing)}"
        )
    # Preflight every video before applying any read-only permissions.
    for vid, video in videos.items():
        metadata = read_json(root / vid / "ingest.json")
        if metadata["video_id"] != vid or ingest_content_hash({"ingest": metadata["ingest_config"]}) != ingest_content_hash(cfg):
            raise HarnessError(f"{vid}: video ID or ingest settings do not match")
        if abs(metadata["duration"] - video["duration"]) > 0.1:
            raise HarnessError(f"{vid}: media duration differs from dataset by more than 0.1s")
    seals = {vid: freeze_wiki(root / vid) for vid in videos}
    manifest = {"version": 2, "dataset": cfg["dataset"]["name"], "split": split,
                "ingest_config_hash": ingest_content_hash(cfg),
                "videos": {vid: seal["wiki_hash"] for vid, seal in seals.items()}}
    # Freeze individual videos only. Other splits may add new videos later;
    # experiments pin their selected video hashes in experiment.json.
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Verify and freeze one split's unique videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--split", help="Exact dataset.json split name (or dataset.split in config)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        from harness.common import identifier
        cfg["dataset"]["name"] = identifier(args.dataset)
    manifest = freeze_dataset(cfg, split=args.split)
    print(f"Frozen {len(manifest['videos'])} videos")


if __name__ == "__main__":
    cli(main)
