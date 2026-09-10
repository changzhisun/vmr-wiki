"""Content settings for the VLM-only bidirectional parser."""
from copy import deepcopy
import math

from harness.common import HarnessError, number, positive_int

PIPELINE_VERSION = "vlm_bidirectional_v1"
SCHEMA_VERSION = 2
DEFAULTS = {
    "max_frames_per_call": 100,
    "topdown": {"enabled": True, "max_depth": 4, "min_segment_duration_sec": 8.0,
                "max_children_per_node": 12},
    "bottomup": {"enabled": True, "window_sec": 45.0, "overlap_ratio": 0.25,
                 "independent_observation": True},
    "reconciliation": {"window_sec": 180.0, "max_records": 128, "max_chars": 48000},
    "boundary_refinement": {"enabled": True, "context_expand_ratio": 0.25,
                            "max_rounds": 2, "uncertainty_sec": 2.0, "endpoint_context_sec": 4.0},
    "confidence": {"self_consistency_for_low_confidence": True, "max_review_rounds": 2},
    "budget": {"requests_per_hour": 1000, "nodes_per_hour": 2000,
               "max_requests": None, "max_nodes": None},
    "shared_cache_dir": None,
}


def settings(ingest: dict) -> dict:
    raw = ingest.get("bidirectional", {})
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys():
        raise HarnessError("Unknown or malformed bidirectional settings")
    result = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - result[key].keys():
                raise HarnessError(f"Unknown bidirectional.{key} setting")
            result[key].update(value)
        else:
            result[key] = value
    for group, keys in (("topdown", ("enabled",)), ("bottomup", ("enabled", "independent_observation")),
                        ("boundary_refinement", ("enabled",)),
                        ("confidence", ("self_consistency_for_low_confidence",))):
        for key in keys:
            if type(result[group][key]) is not bool:
                raise HarnessError(f"{group}.{key} must be boolean")
    if not result["topdown"]["enabled"] and not result["bottomup"]["enabled"]:
        raise HarnessError("At least one evidence pass must be enabled")
    if not result["topdown"]["enabled"] and not result["bottomup"]["independent_observation"]:
        raise HarnessError("Contextual bottom-up requires topdown")
    if not 1 <= positive_int(result["max_frames_per_call"], "max_frames_per_call") <= 100:
        raise HarnessError("max_frames_per_call must be in [1,100]")
    for group, keys in (("topdown", ("max_depth", "max_children_per_node")),
                        ("reconciliation", ("max_records", "max_chars")),
                        ("boundary_refinement", ("max_rounds",)),
                        ("confidence", ("max_review_rounds",)),
                        ("budget", ("requests_per_hour", "nodes_per_hour"))):
        for key in keys:
            positive_int(result[group][key], f"{group}.{key}")
    if result["topdown"]["max_depth"] > 32:
        raise HarnessError("max_depth must be <= 32")
    if any(result[g][k] > 2 for g, k in (("boundary_refinement", "max_rounds"),
                                        ("confidence", "max_review_rounds"))):
        raise HarnessError("Boundary/conflict reviews support at most two rounds")
    for group, key in (("topdown", "min_segment_duration_sec"), ("bottomup", "window_sec"),
                       ("reconciliation", "window_sec"), ("boundary_refinement", "uncertainty_sec"),
                       ("boundary_refinement", "endpoint_context_sec")):
        if number(result[group][key], f"{group}.{key}") <= 0:
            raise HarnessError(f"{group}.{key} must be positive")
    if not 20 <= result["bottomup"]["window_sec"] <= 120:
        raise HarnessError("bottomup.window_sec must be in [20,120]")
    if not 0 <= number(result["bottomup"]["overlap_ratio"], "overlap_ratio") <= 0.5:
        raise HarnessError("bottomup.overlap_ratio must be in [0,0.5]")
    if number(result["boundary_refinement"]["context_expand_ratio"], "context_expand_ratio") < 0:
        raise HarnessError("context_expand_ratio must be nonnegative")
    for key in ("max_requests", "max_nodes"):
        if result["budget"][key] is not None:
            positive_int(result["budget"][key], key)
    if result["shared_cache_dir"] is not None and not isinstance(result["shared_cache_dir"], str):
        raise HarnessError("shared_cache_dir must be a path string or null")
    return result


def content_settings(ingest):
    result = settings(ingest)
    result.pop("shared_cache_dir")
    return result


def budgets(options, duration):
    hours = max(1, math.ceil(duration / 3600))
    raw = options["budget"]
    return {"max_requests": raw["max_requests"] or hours * raw["requests_per_hour"],
            "max_nodes": raw["max_nodes"] or hours * raw["nodes_per_hour"]}
