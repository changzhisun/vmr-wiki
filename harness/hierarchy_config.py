"""Versioned settings for hierarchical video semantics."""
from harness.common import HarnessError, number, positive_int


HIERARCHY_VERSION = 1
DEFAULTS = {
    "max_frames": 100,
    "max_depth": 4,
    "min_segment_sec": 2.0,
    "min_sample_interval_sec": 0.25,
    "overlap_ratio": 0.15,
    "uniform_fraction": 0.6,
    "analysis_fps": 2.0,
    "analysis_timeout_sec": 1800.0,
    "scene_threshold": 0.3,
    "max_children": 12,
    "max_nodes": 2000,
    "max_requests": 1000,
    "merge_adjacent": True,
}


def settings(ingest: dict) -> dict:
    raw = ingest.get("hierarchy", {})
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise HarnessError("Unknown or malformed ingest.hierarchy settings")
    result = {**DEFAULTS, **raw}
    for key in ("max_frames", "max_depth", "max_children", "max_nodes", "max_requests"):
        positive_int(result[key], "hierarchy." + key)
    if not 2 <= result["max_frames"] <= 100:
        raise HarnessError("hierarchy.max_frames must be between 2 and 100")
    if result["max_depth"] > 4:
        raise HarnessError("hierarchy.max_depth must be <= 4 (chapter/scene/event/action)")
    for key in ("min_segment_sec", "min_sample_interval_sec", "analysis_fps", "analysis_timeout_sec"):
        if number(result[key], "hierarchy." + key) <= 0:
            raise HarnessError("hierarchy." + key + " must be positive")
    if not 0.1 <= number(result["overlap_ratio"], "overlap_ratio") <= 0.2:
        raise HarnessError("hierarchy.overlap_ratio must be between 0.1 and 0.2")
    if not 0.2 <= number(result["uniform_fraction"], "uniform_fraction") <= 0.8:
        raise HarnessError("hierarchy.uniform_fraction must be between 0.2 and 0.8")
    if not 0 <= number(result["scene_threshold"], "scene_threshold") <= 1:
        raise HarnessError("hierarchy.scene_threshold must be between 0 and 1")
    if type(result["merge_adjacent"]) is not bool:
        raise HarnessError("hierarchy.merge_adjacent must be boolean")
    return result
