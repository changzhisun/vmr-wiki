from dataclasses import dataclass
from pathlib import Path
import shutil
from vmr.core.errors import HarnessError
from vmr.core.jsonio import read_json
from vmr.core.hashing import object_hash
from .manifest import ArtifactManifest
from .integrity import tree_hashes


@dataclass(frozen=True, slots=True)
class WikiArtifact:
    root: Path
    manifest: ArtifactManifest

    @classmethod
    def open(cls, root: Path):
        root = Path(root)
        if root.is_symlink() or (root / "artifact.json").is_symlink():
            raise HarnessError("Artifact root/manifest cannot be a symlink")
        return cls(root, ArtifactManifest.parse(read_json(root / "artifact.json")))

    def artifact_id(self):
        return self.manifest.data["artifact_id"]

    def duration_sec(self):
        return self.manifest.data["source"]["duration_sec"]

    def manifest_hash(self):
        return object_hash(self.manifest.data)

    def verify(self):
        """Verify the complete build, including private audit bytes."""
        self._verify(public_only=False)

    def verify_public(self):
        """Verify the declared public surface without reading private payloads."""
        self._verify(public_only=True)

    def _verify(self, *, public_only, allow_unsealed_root=False):
        if self.root.is_symlink() or (self.root / "artifact.json").is_symlink():
            raise HarnessError("Artifact root/manifest cannot be a symlink")
        current = ArtifactManifest.parse(read_json(self.root / "artifact.json"))
        if current != self.manifest:
            raise HarnessError("Artifact manifest changed")
        scope = self.root / "public" if public_only else self.root
        actual = tree_hashes(scope, ("artifact.json",) if not public_only else ())
        expected = (
            self.public_hashes() if public_only else current.data["content_hashes"]
        )
        if actual != expected:
            raise HarnessError("Artifact integrity check failed")
        if set(p.name for p in self.root.iterdir()) - {
            "public",
            "internal",
            "artifact.json",
        }:
            raise HarnessError("Unexpected artifact root entries")
        if any(
            (self.root / name).is_symlink() or not (self.root / name).is_dir()
            for name in ("public", "internal")
        ):
            raise HarnessError("Artifact requires public/ and internal/")
        for p in (self.root, self.root / "artifact.json", scope, *scope.rglob("*")):
            if allow_unsealed_root and p == self.root:
                continue
            if p.stat().st_mode & 0o222:
                raise HarnessError(f"Artifact is not read-only sealed: {p.name}")
        if (self.root / "public/wiki.md").read_text(encoding="utf-8").split("\n", 1)[
            0
        ].strip() != "# Video":
            raise HarnessError("Public wiki title must be anonymous: # Video")

    def public_hashes(self, *, input_mode="multimodal"):
        if input_mode not in ("text", "multimodal"):
            raise HarnessError("input_mode must be text or multimodal")
        surface = self.manifest.data["query_surface"]
        names = list(surface["text_files"])
        if input_mode == "multimodal":
            names += surface["multimodal_files"]
        return {p: self.manifest.data["content_hashes"]["public/" + p] for p in names}

    def copy_public_to(self, target: Path, *, input_mode="multimodal"):
        # Always recheck public input: a previously opened handle is not a lease.
        self.verify_public()
        target = Path(target)
        target.mkdir(parents=True, exist_ok=False)
        expected = self.public_hashes(input_mode=input_mode)
        for name in expected:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / "public" / name, destination)
        if tree_hashes(target) != expected:
            raise HarnessError("Artifact changed during public copy")
