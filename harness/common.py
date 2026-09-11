from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HarnessError(ValueError):
    """An input or experiment invariant was violated."""


# A run can fail for two very different reasons. Only the kinds below are
# attributable to the agent under test and are therefore a legitimate zero.
# Anything else means the measurement is invalid, not that the agent scored
# zero, so it must never be averaged into a reported metric unattended.
AGENT_FAILURE_KINDS = frozenset({"timeout", "agent_error", "invalid_output", "tampered"})
HARNESS_FAILURE_KINDS = frozenset({"harness_error", "interrupted"})
FAILURE_KINDS = AGENT_FAILURE_KINDS | HARNESS_FAILURE_KINDS


class RunFailure(HarnessError):
    """A run failure caused by the agent under test rather than the harness."""

    def __init__(self, message: str, kind: str):
        if kind not in AGENT_FAILURE_KINDS:
            raise HarnessError(f"Not an agent failure kind: {kind!r}")
        super().__init__(message)
        self.kind = kind


def identifier(value: Any, field: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value):
        raise HarnessError(f"{field} must be a safe, nonempty identifier: {value!r}")
    return value


def number(value: Any, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise HarnessError(f"{field} must be a finite number")
    if minimum is not None and value < minimum:
        raise HarnessError(f"{field} must be >= {minimum}")
    return float(value)


def positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise HarnessError(f"{field} must be a positive integer")
    return value


def nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{field} must be a nonempty string")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    def bad_constant(value):
        raise HarnessError(f"Invalid JSON constant: {value}")
    try:
        return json.loads(text, parse_constant=bad_constant, object_pairs_hook=_object)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"Invalid JSON: {exc}") from exc


def read_json(path: Path) -> Any:
    return parse_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = parse_json(line)
            if not isinstance(row, dict):
                raise HarnessError("Expected an object")
            rows.append(row)
        except HarnessError as exc:
            raise HarnessError(f"{path}:{line_no}: {exc}") from exc
    return rows


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


# Ingest settings that actually determine caption content. Transport, auth, and
# retry settings are deliberately excluded: changing the provider, endpoint,
# API key environment variable, request timeout, or retry count must not
# invalidate an existing ingest.
_INGEST_CONTENT_KEYS = frozenset({
    "sample_interval_sec", "caption_mode", "caption_window_frames", "caption_stride_frames",
    "caption_max_repairs", "image_max_size", "jpeg_quality",
    "dense_timestamp_mode", "caption_processing_version",
})
_INGEST_CONTENT_DEFAULTS = {
    "caption_mode": "simple", "caption_window_frames": 1, "caption_stride_frames": 1,
    # A repair changes which caption is stored, so the budget is content.
    "caption_max_repairs": 2,
    "dense_timestamp_mode": "legacy_auto",
    "caption_processing_version": 1,
}
_VLM_CONTENT_KEYS = frozenset({"model", "prompt", "temperature", "max_tokens"})


def _ingest_content_value(ingest: dict, key: str):
    if key == "dense_timestamp_mode" and ingest.get("caption_mode", "simple") == "simple":
        return None
    if key in _INGEST_CONTENT_DEFAULTS:
        return ingest.get(key, _INGEST_CONTENT_DEFAULTS[key])
    return ingest[key]


def ingest_content_hash(cfg: dict) -> str:
    """SHA-256 of the ingest settings that determine caption content.

    Excludes ``vlm.provider``, ``vlm.base_url``, ``vlm.api_key_env``,
    ``vlm.timeout_sec`` and ``vlm.max_retries`` because they describe transport
    rather than the requested caption content.
    """
    ingest = cfg["ingest"]
    if ingest.get("caption_mode") == "agentic":
        # No VLM settings participate: the coding agent brings its own model,
        # and the instruction templates are the prompt.
        from harness.agentic_config import AGENTIC_VERSION, content_settings
        return object_hash({
            "caption_mode": "agentic",
            "agentic_version": ingest.get("agentic_version", AGENTIC_VERSION),
            "agentic": content_settings(ingest),
            **{key: ingest[key] for key in ("image_max_size", "jpeg_quality")},
        })
    vlm = ingest["vlm"]
    if ingest.get("caption_mode") == "bidirectional":
        from harness.bidirectional_config import PIPELINE_VERSION, content_settings
        return object_hash({"caption_mode": "bidirectional",
            "pipeline_version": ingest.get("pipeline_version", PIPELINE_VERSION),
            "bidirectional": content_settings(ingest),
            **{key: ingest[key] for key in ("image_max_size", "jpeg_quality", "caption_max_repairs")},
            "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)}})
    if ingest.get("caption_mode") == "hierarchical":
        from harness.hierarchy_config import HIERARCHY_VERSION, settings
        return object_hash({
            "caption_mode": "hierarchical",
            "hierarchy_processing_version": ingest.get("hierarchy_processing_version", HIERARCHY_VERSION),
            "hierarchy": settings(ingest),
            **{key: ingest[key] for key in ("image_max_size", "jpeg_quality", "caption_max_repairs")},
            "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
        })
    return object_hash({
        **{key: _ingest_content_value(ingest, key) for key in sorted(_INGEST_CONTENT_KEYS)},
        "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
    })


def ingest_content_diff(stored: dict, cfg: dict) -> list[str]:
    """Human-readable content-setting differences between a wiki and current config."""
    if ingest_content_hash({"ingest": stored}) == ingest_content_hash(cfg):
        return []
    diffs = []
    if "agentic" in (stored.get("caption_mode"), cfg["ingest"].get("caption_mode")):
        if stored.get("caption_mode") != cfg["ingest"].get("caption_mode"):
            return [f"caption_mode {stored.get('caption_mode')!r} vs "
                    f"{cfg['ingest'].get('caption_mode')!r}"]
        from harness.agentic_config import AGENTIC_VERSION, content_settings
        left, right = content_settings(stored), content_settings(cfg["ingest"])
        for key in sorted(left.keys() | right.keys()):
            if left.get(key) != right.get(key):
                diffs.append("agentic instruction templates differ" if key == "templates_hash"
                             else f"agentic.{key} {left.get(key)!r} vs {right.get(key)!r}")
        if stored.get("agentic_version", AGENTIC_VERSION) != cfg["ingest"].get(
                "agentic_version", AGENTIC_VERSION):
            diffs.append("agentic_version differs")
        for key in ("image_max_size", "jpeg_quality"):
            if stored.get(key) != cfg["ingest"].get(key):
                diffs.append(f"{key} {stored.get(key)!r} vs {cfg['ingest'].get(key)!r}")
        return diffs or ["ingest content hash"]
    if stored.get("caption_mode") == cfg["ingest"].get("caption_mode") == "bidirectional":
        from harness.bidirectional_config import PIPELINE_VERSION, content_settings
        if content_settings(stored) != content_settings(cfg["ingest"]):
            diffs.append("bidirectional settings differ")
        if stored.get("pipeline_version", PIPELINE_VERSION) != cfg["ingest"].get("pipeline_version", PIPELINE_VERSION):
            diffs.append("pipeline_version differs")
    if stored.get("caption_mode") == cfg["ingest"].get("caption_mode") == "hierarchical":
        from harness.hierarchy_config import HIERARCHY_VERSION, settings
        if settings(stored) != settings(cfg["ingest"]):
            diffs.append("hierarchy settings differ")
        if stored.get("hierarchy_processing_version", HIERARCHY_VERSION) != cfg["ingest"].get(
                "hierarchy_processing_version", HIERARCHY_VERSION):
            diffs.append("hierarchy_processing_version differs")
    for key in sorted(_INGEST_CONTENT_KEYS):
        left = _ingest_content_value(stored, key)
        right = _ingest_content_value(cfg["ingest"], key)
        if left != right:
            diffs.append(f"{key} {left!r} vs {right!r}")
    stored_vlm = stored.get("vlm") or {}
    current_vlm = cfg["ingest"]["vlm"]
    for key in sorted(_VLM_CONTENT_KEYS):
        left, right = stored_vlm.get(key), current_vlm.get(key)
        if left != right:
            diffs.append(f"vlm.{key} {left!r} vs {right!r}" if key != "prompt"
                         else "vlm.prompt differs")
    return diffs or ["ingest content hash"]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(canonical(row).decode() + "\n" for row in rows))


def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = identifier(row.get(key), key)
        if value in result:
            raise HarnessError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def cli(main) -> None:
    try:
        main()
    except (HarnessError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
