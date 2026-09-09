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
_INGEST_CONTENT_KEYS = frozenset({"sample_interval_sec", "image_max_size", "jpeg_quality"})
_VLM_CONTENT_KEYS = frozenset({"model", "prompt", "temperature", "max_tokens"})


def ingest_content_hash(cfg: dict) -> str:
    """SHA-256 of the ingest settings that determine caption content.

    Excludes ``vlm.provider``, ``vlm.base_url``, ``vlm.api_key_env``,
    ``vlm.timeout_sec`` and ``vlm.max_retries`` because they describe transport
    rather than the requested caption content.
    """
    ingest = cfg["ingest"]
    vlm = ingest["vlm"]
    return object_hash({
        **{key: ingest[key] for key in sorted(_INGEST_CONTENT_KEYS)},
        "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
    })


def ingest_content_diff(stored: dict, cfg: dict) -> list[str]:
    """Human-readable content-setting differences between a wiki and current config."""
    if ingest_content_hash({"ingest": stored}) == ingest_content_hash(cfg):
        return []
    diffs = []
    for key in sorted(_INGEST_CONTENT_KEYS):
        if stored.get(key) != cfg["ingest"].get(key):
            diffs.append(f"{key} {stored.get(key)!r} vs {cfg['ingest'].get(key)!r}")
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
