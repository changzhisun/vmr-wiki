import copy
import pytest
from vmr.artifact.store import ArtifactStore
from vmr.artifact.artifact import WikiArtifact
from vmr.artifact.manifest import ArtifactManifest, identity
from vmr.artifact.integrity import remove_tree
from vmr.core.errors import HarnessError


@pytest.fixture
def artifact(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    stage = store.staging()
    (stage / "public/wiki.md").write_text("# Video\nA scene")
    (stage / "public/new-format.txt").write_text("new compiler text")
    (stage / "public/image.jpg").write_bytes(b"picture")
    (stage / "internal/private.json").write_text('{"dataset":"hidden"}')
    result = store.publish(
        stage,
        source={"video_sha256": "a" * 64, "duration_sec": 5},
        compiler={
            "name": "external",
            "version": 1,
            "content_config_hash": "sha256:" + "b" * 64,
        },
        text_files=["wiki.md", "new-format.txt"],
        multimodal_files=["image.jpg"],
    )
    yield result
    remove_tree(store.root)


def test_modes_are_manifest_owned(artifact, tmp_path):
    artifact.verify()
    artifact.copy_public_to(tmp_path / "text", input_mode="text")
    assert set(p.name for p in (tmp_path / "text").iterdir()) == {
        "wiki.md",
        "new-format.txt",
    }
    artifact.copy_public_to(tmp_path / "multi", input_mode="multimodal")
    assert set(p.name for p in (tmp_path / "multi").iterdir()) == {
        "wiki.md",
        "new-format.txt",
        "image.jpg",
    }


@pytest.mark.parametrize(
    "path", ["../escape", "/absolute", "a/../b", "a//b", "./a", "a\\b", "a/"]
)
def test_manifest_rejects_unsafe_paths(artifact, path):
    data = copy.deepcopy(artifact.manifest.data)
    data["query_surface"]["text_files"].append(path)
    data["content_hashes"]["public/" + path] = "f" * 64
    data["artifact_id"] = identity(data)
    with pytest.raises(HarnessError):
        ArtifactManifest.parse(data)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "surface-version",
        "extra-field",
        "undeclared",
        "internal-path",
        "identity",
        "writable",
        "symlink",
        "extra-file",
        "missing",
        "bytes",
        "root-symlink",
    ],
)
def test_integrity_fail_closed(artifact, tmp_path, change):
    data = copy.deepcopy(artifact.manifest.data)
    if change in (
        "schema",
        "surface-version",
        "extra-field",
        "undeclared",
        "internal-path",
        "identity",
    ):
        if change == "schema":
            data["schema_version"] = 999
        if change == "surface-version":
            data["query_surface"]["version"] = 2
        if change == "extra-field":
            data["routing"] = "unknown semantic field"
        if change == "undeclared":
            data["query_surface"]["text_files"].remove("wiki.md")
        if change == "internal-path":
            data["query_surface"]["text_files"].append("../internal/private.json")
        data["artifact_id"] = identity(data)
        if change == "identity":
            data["artifact_id"] = "sha256:" + "f" * 64
        with pytest.raises(HarnessError):
            ArtifactManifest.parse(data)
        return
    p = artifact.root / "public/wiki.md"
    if change in ("writable", "bytes"):
        p.chmod(0o644)
        if change == "bytes":
            p.write_text("# Video\nChanged")
            p.chmod(0o444)
    elif change in ("symlink", "missing"):
        p.parent.chmod(0o755)
        p.unlink()
        if change == "symlink":
            p.symlink_to(artifact.root / "internal/private.json")
        p.parent.chmod(0o555)
    elif change == "extra-file":
        p.parent.chmod(0o755)
        (p.parent / "surprise").write_text("extra")
        p.parent.chmod(0o555)
    elif change == "root-symlink":
        link = tmp_path / "link"
        link.symlink_to(artifact.root)
        with pytest.raises(HarnessError):
            WikiArtifact.open(link)
        return
    with pytest.raises(HarnessError):
        artifact.verify()


def test_mutable_manifest_cannot_change_after_open(artifact):
    p = artifact.root / "artifact.json"
    p.chmod(0o644)
    data = copy.deepcopy(artifact.manifest.data)
    data["provenance"]["changed"] = True
    p.write_text(__import__("json").dumps(data))
    p.chmod(0o444)
    with pytest.raises(HarnessError, match="manifest changed"):
        artifact.verify()
