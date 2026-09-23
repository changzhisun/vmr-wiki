"""Local publication keyed by Wiki identity and exact build manifest hash."""

import errno
from pathlib import Path
import tempfile

from vmr.core.errors import HarnessError
from vmr.core.locking import exclusive_lock
from vmr.core.jsonio import write_json
from vmr.core.time import now
from .artifact import WikiArtifact
from .manifest import artifact_key, digest, identity, ArtifactManifest
from .integrity import tree_hashes, make_readonly, remove_tree


class ArtifactStore:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, artifact_id, *, manifest_hash=None):
        content = self.root / artifact_key(artifact_id).replace(":", "-")
        return content if manifest_hash is None else content / digest(manifest_hash)

    def open(self, artifact_id, *, manifest_hash=None):
        content = self.path(artifact_id)
        if content.is_symlink():
            raise HarnessError("Artifact store key cannot be a symlink")
        legacy_layout = (content / "artifact.json").exists()
        if legacy_layout:
            # Published v1 stores retain their original layout and identity.
            path = content
        elif manifest_hash is not None:
            path = self.path(artifact_id, manifest_hash=manifest_hash)
        else:
            records = sorted(content.iterdir()) if content.is_dir() else []
            if len(records) != 1:
                raise HarnessError(
                    "Artifact requires an exact manifest hash (missing or ambiguous build)"
                )
            path = records[0]
        result = WikiArtifact.open(path)
        result.verify()
        if result.artifact_id() != artifact_id:
            raise HarnessError("Artifact store key mismatch")
        expected_manifest = manifest_hash if legacy_layout else digest(path.name)
        if (
            expected_manifest is not None
            and result.manifest_hash() != expected_manifest
        ):
            raise HarnessError("Artifact manifest hash mismatch")
        return result

    def staging(self):
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=self.root))
        (staging / "public").mkdir()
        (staging / "internal").mkdir()
        return staging

    def publish(
        self,
        staging,
        *,
        source,
        compiler,
        text_files,
        multimodal_files,
        provenance=None,
    ):
        staging = Path(staging)
        data = dict(
            schema_version=2,
            artifact_type="vmr.wiki",
            sealed=True,
            source=source,
            compiler=compiler,
            query_surface=dict(
                version=1,
                text_files=sorted(text_files),
                multimodal_files=sorted(multimodal_files),
            ),
            content_hashes=tree_hashes(staging),
            provenance=provenance if provenance is not None else {"created_at": now()},
        )
        data["artifact_id"] = identity(data)
        ArtifactManifest.parse(data)
        write_json(staging / "artifact.json", data)
        make_readonly(staging)
        artifact = WikiArtifact.open(staging)
        artifact.verify()
        key, record = artifact.artifact_id(), artifact.manifest_hash()
        target = self.path(key, manifest_hash=record)
        target.parent.mkdir(parents=True, exist_ok=True)

        # All publishers of this exact record share a persistent lock inode.
        # A dead publisher releases the OS lock, allowing validated recovery.
        with exclusive_lock(self.root / ".locks" / record, blocking=True):

            def existing_record():
                existing = WikiArtifact.open(target)
                if existing.manifest != artifact.manifest:
                    raise HarnessError("Existing publication manifest mismatch")
                # Only the top directory may be unsealed after rename. Never
                # repair modified bytes, symlinks, or writable payload files.
                existing._verify(public_only=False, allow_unsealed_root=True)
                if target.stat().st_mode & 0o222:
                    target.chmod(0o555)
                existing.verify()
                remove_tree(staging)
                return existing

            if target.exists():
                return existing_record()
            try:
                # macOS requires the source directory to be writable for rename.
                # Payloads stay read-only; readers reject the unsealed root.
                staging.chmod(0o755)
                staging.rename(target)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
                return existing_record()
            target.chmod(0o555)
            return self.open(key, manifest_hash=record)
