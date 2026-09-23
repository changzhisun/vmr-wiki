"""Content identity, exact build retention, and atomic publication races."""

import copy
import errno
from pathlib import Path
import shutil

import pytest

from vmr.artifact.artifact import WikiArtifact
from vmr.artifact.integrity import make_readonly, remove_tree
from vmr.artifact.manifest import identity
from vmr.artifact.store import ArtifactStore
from vmr.artifact.wikiset import write_wikiset
from vmr.core.errors import HarnessError
from vmr.core.hashing import object_hash
from vmr.core.jsonio import write_json


@pytest.fixture
def store(tmp_path):
    result = ArtifactStore(tmp_path / "store")
    yield result
    remove_tree(result.root)


def publish(store, *, audit="first build", wiki="# Video\nA scene", provenance=None):
    stage = store.staging()
    (stage / "public/wiki.md").write_text(wiki)
    (stage / "internal/audit.json").write_text(audit)
    return store.publish(
        stage,
        source={"video_sha256": "a" * 64, "duration_sec": 5},
        compiler={
            "name": "external",
            "version": 1,
            "content_config_hash": "sha256:" + "b" * 64,
        },
        text_files=["wiki.md"],
        multimodal_files=[],
        provenance=provenance if provenance is not None else {"created_at": audit},
    )


def test_stable_content_keeps_distinct_build_records(store, tmp_path):
    first = publish(store)
    second = publish(store, audit="second build")
    third = publish(store, provenance={"created_at": "different provenance only"})
    assert first.artifact_id() == second.artifact_id() == third.artifact_id()
    assert len({a.manifest_hash() for a in (first, second, third)}) == 3
    for i, artifact in enumerate((first, second, third)):
        wiki_set = write_wikiset(
            tmp_path / f"{i}.json",
            dataset="fixture",
            split="train",
            artifacts={"video": artifact},
            store=store.root,
        )
        reopened = wiki_set.artifact("video")
        assert reopened.root == artifact.root
        assert reopened.manifest == artifact.manifest
        assert (reopened.root / "internal/audit.json").read_text() == (
            "second build" if i == 1 else "first build"
        )
    with pytest.raises(HarnessError, match="exact manifest hash"):
        store.open(first.artifact_id())
    assert publish(store).root == first.root  # Exact builds deduplicate.
    assert publish(store, wiki="# Video\nChanged").artifact_id() != first.artifact_id()


def test_stable_content_across_fresh_stores(store, tmp_path):
    other_store = ArtifactStore(tmp_path / "other")
    try:
        first, second = publish(store), publish(other_store, audit="other machine")
        assert first.public_hashes() == second.public_hashes()
        assert first.artifact_id() == second.artifact_id()
        assert first.manifest_hash() != second.manifest_hash()
    finally:
        remove_tree(other_store.root)


def test_private_bytes_still_verified(store):
    artifact = publish(store)
    audit = artifact.root / "internal/audit.json"
    audit.chmod(0o644)
    audit.write_text("tampered")
    audit.chmod(0o444)
    artifact.verify_public()
    with pytest.raises(HarnessError, match="integrity"):
        artifact.verify()


def test_legacy_identity_and_layout_remain_readable(store):
    artifact = publish(store)
    data = copy.deepcopy(artifact.manifest.data)
    data["schema_version"] = 1
    old_payload = {
        k: data[k]
        for k in (
            "schema_version",
            "artifact_type",
            "source",
            "compiler",
            "query_surface",
            "content_hashes",
        )
    }
    data["artifact_id"] = "sha256:" + object_hash(old_payload)
    assert identity(data) == data["artifact_id"]
    legacy = store.path(data["artifact_id"])
    shutil.copytree(artifact.root, legacy)
    legacy.chmod(0o755)
    (legacy / "artifact.json").chmod(0o644)
    write_json(legacy / "artifact.json", data)
    make_readonly(legacy)
    reopened = store.open(data["artifact_id"], manifest_hash=object_hash(data))
    assert reopened.root == legacy
    changed = copy.deepcopy(data)
    changed["content_hashes"]["internal/audit.json"] = "c" * 64
    assert identity(changed) != data["artifact_id"]


@pytest.mark.parametrize("error", [errno.EEXIST, errno.ENOTEMPTY])
def test_concurrent_publish_verifies_winning_record(store, monkeypatch, error):
    def race(source, target):
        shutil.copytree(source, target)
        make_readonly(target)
        raise OSError(error, "concurrent publisher won")

    monkeypatch.setattr(Path, "rename", race)
    artifact = publish(store)
    artifact.verify()
    assert not list(store.root.glob(".publish-*"))


def test_unrelated_rename_error_is_not_swallowed(store, monkeypatch):
    def fail(*args):
        raise OSError(errno.EACCES, "permission denied")

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError) as caught:
        publish(store)
    assert caught.value.errno == errno.EACCES
    assert list(store.root.glob(".publish-*"))


def test_corrupt_concurrent_winner_is_not_accepted(store, monkeypatch):
    def race(source, target):
        shutil.copytree(source, target)
        p = target / "internal/audit.json"
        p.chmod(0o644)
        p.write_text("corrupt winner")
        make_readonly(target)
        raise OSError(errno.ENOTEMPTY, "concurrent publisher won")

    monkeypatch.setattr(Path, "rename", race)
    with pytest.raises(HarnessError, match="integrity"):
        publish(store)
    assert list(store.root.glob(".publish-*"))


def test_public_copy_rechecks_old_handles(store, tmp_path):
    artifact = publish(store)
    artifact.verify()
    wiki = artifact.root / "public/wiki.md"
    wiki.chmod(0o644)
    wiki.write_text("# Video\nChanged after open")
    wiki.chmod(0o444)
    with pytest.raises(HarnessError, match="integrity"):
        artifact.copy_public_to(tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_public_copy_detects_changes_during_copy(store, tmp_path, monkeypatch):
    artifact = publish(store)
    original = shutil.copyfile

    def corrupt_copy(source, target):
        original(source, target)
        Path(target).write_text("corrupt copy")

    monkeypatch.setattr(shutil, "copyfile", corrupt_copy)
    with pytest.raises(HarnessError, match="changed during public copy"):
        artifact.copy_public_to(tmp_path / "copy")


@pytest.mark.parametrize("damage", [None, "bytes", "permissions", "manifest"])
def test_recover_interrupted_root_seal_only(store, monkeypatch, damage):
    original_chmod = Path.chmod
    interrupted = []

    def fail_final_seal(path, mode, *args, **kwargs):
        if path.parent.name.startswith("sha256-") and mode == 0o555:
            interrupted.append(path)
            raise OSError("interrupted after rename")
        return original_chmod(path, mode, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "chmod", fail_final_seal)
        with pytest.raises(OSError, match="interrupted after rename"):
            publish(store)
    (target,) = interrupted
    artifact = WikiArtifact.open(target)
    with pytest.raises(HarnessError):
        artifact.verify()
    if damage:
        payload = target / (
            "artifact.json" if damage == "manifest" else "internal/audit.json"
        )
        payload.chmod(0o644)
        if damage == "bytes":
            payload.write_text("tampered")
            payload.chmod(0o444)
        elif damage == "manifest":
            data = copy.deepcopy(artifact.manifest.data)
            data["provenance"] = {"changed": True}
            write_json(payload, data)
            payload.chmod(0o444)
        with pytest.raises(HarnessError):
            publish(store)
        assert target.stat().st_mode & 0o222
    else:
        recovered = publish(store)
        assert recovered.root == target
        assert recovered.manifest == artifact.manifest
        recovered.verify()
