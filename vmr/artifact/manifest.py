"""Strict, compiler-independent Artifact v1/v2 wire formats."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from vmr.core.errors import HarnessError
from vmr.core.hashing import object_hash
from vmr.core.validation import number, nonempty, positive_int


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise HarnessError("Expected a SHA256 digest")
    return value


def artifact_key(value):
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise HarnessError("Expected sha256: artifact identity")
    digest(value[7:])
    return value


def safe_path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
    ):
        raise HarnessError(f"Unsafe artifact path: {value!r}")
    return value


def fields(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise HarnessError(f"Invalid {label} fields; expected {sorted(expected)}")


def identity(data):
    payload = {
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
    if data["schema_version"] == 2:
        # Audit bytes remain pinned by the full manifest hash, independently
        # of the stable Wiki content identity. Preserve v1 identity semantics.
        payload["content_hashes"] = {
            path: digest
            for path, digest in data["content_hashes"].items()
            if path.startswith("public/")
        }
    return "sha256:" + object_hash(payload)


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    data: dict

    @classmethod
    def parse(cls, data):
        fields(
            data,
            (
                "schema_version",
                "artifact_type",
                "artifact_id",
                "source",
                "compiler",
                "query_surface",
                "content_hashes",
                "provenance",
                "sealed",
            ),
            "artifact",
        )
        if type(data["schema_version"]) is not int or data["schema_version"] not in (
            1,
            2,
        ):
            raise HarnessError("Unsupported artifact schema version")
        if data["artifact_type"] != "vmr.wiki" or data["sealed"] is not True:
            raise HarnessError("Expected a sealed vmr.wiki artifact")
        fields(data["source"], ("video_sha256", "duration_sec"), "source")
        digest(data["source"]["video_sha256"])
        if number(data["source"]["duration_sec"], "duration_sec") <= 0:
            raise HarnessError("Artifact duration must be positive")
        fields(data["compiler"], ("name", "version", "content_config_hash"), "compiler")
        nonempty(data["compiler"]["name"], "compiler.name")
        positive_int(data["compiler"]["version"], "compiler.version")
        artifact_key(data["compiler"]["content_config_hash"])
        surface = data["query_surface"]
        fields(surface, ("version", "text_files", "multimodal_files"), "query surface")
        if type(surface["version"]) is not int or surface["version"] != 1:
            raise HarnessError("Unsupported query surface version")
        declared = []
        for key in ("text_files", "multimodal_files"):
            if not isinstance(surface[key], list):
                raise HarnessError("Surface paths must be explicit lists of files")
            declared.extend(safe_path(p) for p in surface[key])
        if (
            len(declared) != len(set(declared))
            or "wiki.md" not in surface["text_files"]
        ):
            raise HarnessError("Surface must contain unique paths and a text wiki.md")
        hashes = data["content_hashes"]
        if not isinstance(hashes, dict):
            raise HarnessError("content_hashes must be a mapping")
        for path, value in hashes.items():
            safe_path(path)
            if not path.startswith(("public/", "internal/")):
                raise HarnessError("Content must be inside public/ or internal/")
            digest(value)
        if {p[7:] for p in hashes if p.startswith("public/")} != set(declared):
            raise HarnessError("Public files and query surface disagree")
        if not isinstance(data["provenance"], dict):
            raise HarnessError("provenance must be a mapping")
        artifact_key(data["artifact_id"])
        if identity(data) != data["artifact_id"]:
            raise HarnessError("Artifact identity mismatch")
        return cls(data)
