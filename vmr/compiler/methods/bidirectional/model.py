"""Versioned flat temporal graphs and transactional, evidence-linked edits."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from vmr.core.errors import HarnessError
from vmr.core.hashing import canonical, object_hash
from vmr.core.validation import nonempty, number

TYPES = {
    "chapter",
    "scene",
    "event",
    "action",
    "state_change",
    "transition",
    "dialogue",
    "other",
}
CONFIDENCE = {"high", "medium", "low"}
OPERATIONS = {
    "KEEP",
    "INSERT",
    "DELETE",
    "SPLIT",
    "MERGE",
    "SHIFT",
    "RELABEL",
    "REPARENT",
}
SEMANTICS = {
    "type",
    "title",
    "summary",
    "actors",
    "actions",
    "objects",
    "state_before",
    "state_after",
    "retrieval_text",
    "confidence",
    "needs_refinement",
    "semantic_complexity",
    "multiple_actions",
}
NODE_EXAMPLE = {
    "granularity": 0,
    "type": "event",
    "start": 0.0,
    "end": 1.0,
    "title": "Visible event",
    "summary": "Describe only visible evidence.",
    "actors": [],
    "actions": [],
    "objects": [],
    "state_before": "",
    "state_after": "",
    "confidence": {"semantic": "medium", "boundary": "low", "hierarchy": "medium"},
    "boundary_uncertainty": {"start": [0.0, 0.5], "end": [0.5, 1.0]},
    "needs_refinement": False,
    "semantic_complexity": "low",
    "multiple_actions": False,
    "retrieval_text": ["Visible event"],
}


def strings(value, name):
    if not isinstance(value, list):
        raise HarnessError(f"{name} must be a list")
    return list(dict.fromkeys(nonempty(item, name).strip() for item in value))


def cite_ids(item, key):
    """Read a list of IDs, accepting the singular field name or a bare string."""
    singular = key[:-1] if key.endswith("s") else key
    if key in item:
        value = item[key]
    elif singular in item:
        value = item[singular]
    else:
        value = []
    if isinstance(value, str):
        value = [value] if value.strip() else []
    if value is None:
        value = []
    return strings(value, key)


def _clamp_interval(start, end, duration):
    start = min(max(start, 0.0), duration)
    end = min(max(end, 0.0), duration)
    if not start < end:
        raise HarnessError("Node must satisfy 0 <= start < end <= video duration")
    return start, end


def clip_to_range(node_start, node_end, lo, hi):
    """Clip [node_start, node_end] to [lo, hi]. None if they do not overlap."""
    start = max(lo, min(node_start, hi))
    end = max(lo, min(node_end, hi))
    if not start < end:
        return None
    return start, end


def _uncertainty_range(bounds, point, duration):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return [point, point]
    try:
        lo = number(bounds[0], "uncertainty")
        hi = number(bounds[1], "uncertainty")
    except HarnessError:
        return [point, point]
    lo = min(max(0.0, lo), point)
    hi = max(min(duration, hi), point)
    if lo > hi:
        return [point, point]
    return [lo, hi]


def normalize_node(raw, duration):
    if not isinstance(raw, dict):
        raise HarnessError("Node must be an object")
    row = deepcopy(raw)
    try:
        start, end = _clamp_interval(
            number(row["start"], "start"), number(row["end"], "end"), duration
        )
        row.update(start=start, end=end)
        if (
            type(row["granularity"]) is not int
            or row["granularity"] < 0
            or row["type"] not in TYPES
        ):
            raise HarnessError("Invalid granularity or semantic type")
        for key in ("title", "summary"):
            row[key] = nonempty(row[key], key).strip()
        for key in ("actors", "actions", "objects"):
            row[key] = strings(row.get(key, []), key)
        for key in ("state_before", "state_after"):
            if not isinstance(row.get(key, ""), str):
                raise HarnessError(f"{key} must be a string")
            row.setdefault(key, "")
        conf = row["confidence"]
        if (
            not isinstance(conf, dict)
            or set(conf) != {"semantic", "boundary", "hierarchy"}
            or any(value not in CONFIDENCE for value in conf.values())
        ):
            raise HarnessError(
                "confidence must contain semantic/boundary/hierarchy discrete labels"
            )
        uncertainty = row.get("boundary_uncertainty")
        if not isinstance(uncertainty, dict):
            uncertainty = {}
        row["boundary_uncertainty"] = {
            "start": _uncertainty_range(
                uncertainty.get("start"), row["start"], duration
            ),
            "end": _uncertainty_range(uncertainty.get("end"), row["end"], duration),
        }
        for key in ("needs_refinement", "multiple_actions"):
            row.setdefault(key, False)
            if type(row[key]) is not bool:
                raise HarnessError(f"{key} must be boolean")
        row.setdefault("semantic_complexity", "low")
        if row["semantic_complexity"] not in CONFIDENCE:
            raise HarnessError("Invalid semantic_complexity")
        row["retrieval_text"] = strings(
            row.get("retrieval_text", [row["summary"]]), "retrieval_text"
        )
        row.setdefault("parent_id", None)
        if row["parent_id"] is not None:
            nonempty(row["parent_id"], "parent_id")
        row.setdefault("relations", [])
        if not isinstance(row["relations"], list):
            raise HarnessError("relations must be a list")
        for relation in row["relations"]:
            if (
                not isinstance(relation, dict)
                or set(relation) != {"type", "target_id"}
                or relation["type"]
                not in {"part_of", "related", "continuation", "same_event"}
            ):
                raise HarnessError("Invalid relation")
            nonempty(relation["target_id"], "relation target")
        row.setdefault("history", [])
        if not isinstance(row["history"], list) or any(
            not isinstance(item, dict) for item in row["history"]
        ):
            raise HarnessError("history must be a list of records")
        row.setdefault(
            "evidence",
            {
                "pass": [],
                "observation_ids": [],
                "frame_ids": [],
                "frame_timestamps": [],
            },
        )
        if not isinstance(row["evidence"], dict):
            raise HarnessError("evidence must be an object")
        for key in ("pass", "observation_ids", "frame_ids"):
            row["evidence"][key] = strings(
                row["evidence"].get(key, []), "evidence." + key
            )
        row["evidence"]["frame_timestamps"] = sorted(
            set(
                number(stamp, "evidence timestamp", 0)
                for stamp in row["evidence"].get("frame_timestamps", [])
            )
        )
        if any(stamp >= duration for stamp in row["evidence"]["frame_timestamps"]):
            raise HarnessError("Evidence timestamp outside video")
        row.setdefault(
            "review_status", "unresolved" if "low" in conf.values() else "resolved"
        )
        if row["review_status"] not in {"resolved", "unresolved"}:
            raise HarnessError("Invalid review_status")
        row.setdefault("issues", [])
        row["issues"] = strings(row["issues"], "issues")
        return row
    except (KeyError, TypeError) as exc:
        raise HarnessError(f"Missing or malformed node field: {exc}") from exc


def read_node(raw):
    """Non-mutating legacy adapter; an uncalibrated numeric score stays legacy."""
    row = deepcopy(raw)
    if "granularity" not in row:
        levels = ("chapter", "scene", "event", "action")
        row["granularity"] = levels.index(row["level"])
        row["type"] = row["level"]
        row["legacy_confidence"] = row["confidence"]
        row["confidence"] = dict.fromkeys(("semantic", "boundary", "hierarchy"), "low")
    return row


def combine_evidence(*items):
    result = {
        "pass": [],
        "observation_ids": [],
        "frame_ids": [],
        "frame_timestamps": [],
    }
    for item in items:
        for key in result:
            result[key].extend(item.get(key, []))
    return {
        key: sorted(set(value))
        if key == "frame_timestamps"
        else list(dict.fromkeys(value))
        for key, value in result.items()
    }
