from dataclasses import dataclass
from pathlib import Path
import os
from vmr.core.errors import HarnessError
from vmr.core.hashing import object_hash
from vmr.core.jsonio import read_json, write_json
from vmr.core.validation import identifier, nonempty
from .manifest import artifact_key, digest, fields
from .store import ArtifactStore


@dataclass(frozen=True, slots=True)
class WikiSet:
    data: dict
    store: ArtifactStore

    @classmethod
    def open(cls, path, *, store=None):
        path = Path(path)
        data = read_json(path)
        fields(
            data,
            (
                "schema_version",
                "dataset",
                "split",
                "artifact_store",
                "artifacts",
                "manifest_hashes",
            ),
            "WikiSet",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise HarnessError("Unsupported WikiSet schema version")
        identifier(data["dataset"], "dataset")
        nonempty(data["split"], "split")
        nonempty(data["artifact_store"], "artifact_store")
        if not isinstance(data["artifacts"], dict) or not data["artifacts"]:
            raise HarnessError("WikiSet artifacts must be a nonempty mapping")
        if (
            not isinstance(data["manifest_hashes"], dict)
            or data["manifest_hashes"].keys() != data["artifacts"].keys()
        ):
            raise HarnessError("WikiSet must pin each manifest hash")
        for vid, key in data["artifacts"].items():
            identifier(vid, "video_id")
            artifact_key(key)
            digest(data["manifest_hashes"][vid])
        return cls(data, ArtifactStore(store or path.parent / data["artifact_store"]))

    def fingerprint(self):
        return object_hash(
            {k: v for k, v in self.data.items() if k != "artifact_store"}
        )

    def artifact(self, video_id):
        if video_id not in self.data["artifacts"]:
            raise HarnessError(f"Video missing from WikiSet: {video_id}")
        artifact = self.store.open(
            self.data["artifacts"][video_id],
            manifest_hash=self.data["manifest_hashes"][video_id],
        )
        if artifact.manifest_hash() != self.data["manifest_hashes"][video_id]:
            raise HarnessError("WikiSet manifest hash mismatch")
        return artifact

    def verify(self):
        for vid in self.data["artifacts"]:
            self.artifact(vid)


def write_wikiset(path, *, dataset, split, artifacts, store):
    path = Path(path)
    data = dict(
        schema_version=1,
        dataset=dataset,
        split=split,
        artifact_store=os.path.relpath(Path(store).resolve(), path.parent.resolve()),
        artifacts={vid: art.artifact_id() for vid, art in artifacts.items()},
        manifest_hashes={vid: art.manifest_hash() for vid, art in artifacts.items()},
    )
    if path.exists() and read_json(path) != data:
        raise HarnessError(
            "WikiSet already exists with different artifacts; choose a new output set"
        )
    write_json(path, data)
    result = WikiSet.open(path)
    result.verify()
    return result
