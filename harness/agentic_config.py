"""Versioned settings for the agent-compiled (coding agent) ingest backend.

The method itself lives in the instruction templates rather than in this file:
a coding agent decides how to sample and how to describe. Their hash is
therefore part of the content identity, exactly like a VLM prompt is.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from harness.common import HarnessError, file_hash, identifier, nonempty, number, object_hash, positive_int

AGENTIC_VERSION = "agentic_v1"

# The instruction files that define the method. Both are copied into the job
# workspace and both belong to the content hash.
TEMPLATES = ("wiki_agents.md", "wiki_prompt.md")

AGENTS = ("codex", "claude_code")

DEFAULTS = {
    "agent": "codex",
    # No usable default: an unpinned model would silently change the method.
    "model": None,
    # The same image as the query agent, so both always run identical CLI
    # versions. It carries ffmpeg for the agent's own frame extraction.
    "container_image": "vmr-wiki-agents:local",
    "timeout_sec": 3600,
    # The agent extracts candidate frames before selecting evidence, so it
    # needs writable scratch far larger than the query agent's /tmp.
    "scratch_size_gb": 8,
    "memory_gb": 8,
    "cpus": 4,
    "pids_limit": 512,
    # Bounds the published evidence set; passed to the agent in task.json.
    "max_frames": 500,
    # Host-side publication guards. They do not change what the agent is asked
    # to do, so they stay out of the content hash.
    "max_wiki_bytes": 4 * 1024 * 1024,
    "max_frame_bytes": 4 * 1024 * 1024,
    "api_key_env": {"codex": "CODEX_API_KEY", "claude_code": "ANTHROPIC_API_KEY"},
    "base_url": {"codex": None, "claude_code": None},
    # No default: the allowlist must name the endpoints of this deployment.
    "egress_allowed_hosts": {},
    "frame_extraction": {
        "strategy": "adaptive",
        "initial_interval_sec": 5.0,
        "min_interval_sec": 0.25,
    },
    "wiki": {
        "levels": ["chapter", "event", "moment"],
        "include_entities": True,
        "include_objects": True,
        "include_locations": True,
        "include_temporal_relations": True,
        "include_retrieval_aliases": True,
    },
}

STRATEGIES = ("adaptive", "uniform")
LEVELS = ("chapter", "event", "moment")

# Settings that describe the container, its budget or its transport. Changing
# any of them must not invalidate an existing wiki.
_TRANSPORT_KEYS = ("container_image", "timeout_sec", "scratch_size_gb", "memory_gb", "cpus",
                   "pids_limit", "max_wiki_bytes", "max_frame_bytes",
                   "api_key_env", "base_url", "egress_allowed_hosts")


def template_hash(templates: Path) -> str:
    """Hash of the instruction files that define the agentic method."""
    hashes = {}
    for name in TEMPLATES:
        path = Path(templates) / name
        if not path.is_file():
            raise HarnessError(f"Missing agentic instruction template: {path}")
        hashes[name] = file_hash(path)
    return object_hash(hashes)


def settings(ingest: dict, templates: Path | None = None) -> dict:
    """Normalize ``ingest.agentic``.

    ``templates`` is passed when loading a configuration, so the current
    instruction files determine ``templates_hash``. A stored ingest config
    carries its own recorded hash instead and is read back without the files.
    """
    raw = ingest.get("agentic", {})
    if not isinstance(raw, dict) or set(raw) - DEFAULTS.keys() - {"templates_hash"}:
        raise HarnessError("Unknown or malformed ingest.agentic settings")
    result = deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key == "templates_hash":
            continue
        if isinstance(result[key], dict) and result[key]:
            if not isinstance(value, dict) or set(value) - result[key].keys():
                raise HarnessError(f"Unknown ingest.agentic.{key} setting")
            result[key] = {**result[key], **value}
        else:
            result[key] = value

    if result["agent"] not in AGENTS:
        raise HarnessError("ingest.agentic.agent must be codex or claude_code")
    model = nonempty(result["model"], "ingest.agentic.model")
    if model.startswith("REPLACE_"):
        raise HarnessError("Configure an explicit ingest.agentic.model")
    nonempty(result["container_image"], "ingest.agentic.container_image")
    if number(result["timeout_sec"], "ingest.agentic.timeout_sec") <= 0:
        raise HarnessError("ingest.agentic.timeout_sec must be positive")
    for key in ("scratch_size_gb", "memory_gb", "pids_limit", "max_frames",
                "max_wiki_bytes", "max_frame_bytes"):
        positive_int(result[key], "ingest.agentic." + key)
    if number(result["cpus"], "ingest.agentic.cpus") <= 0:
        raise HarnessError("ingest.agentic.cpus must be positive")
    if result["max_frames"] > 5000:
        raise HarnessError("ingest.agentic.max_frames must be <= 5000")

    hosts = result["egress_allowed_hosts"]
    if not isinstance(hosts, dict):
        raise HarnessError("ingest.agentic.egress_allowed_hosts must be a mapping of agent to hosts")
    if not isinstance(result["api_key_env"], dict) or not isinstance(result["base_url"], dict):
        raise HarnessError("ingest.agentic.api_key_env and base_url must be mappings")
    for name in (*result["api_key_env"], *result["base_url"], *hosts):
        if name not in AGENTS:
            raise HarnessError(f"Unknown agent in ingest.agentic settings: {name!r}")
    identifier(result["api_key_env"].get(result["agent"]), "ingest.agentic.api_key_env")

    extraction = result["frame_extraction"]
    if extraction["strategy"] not in STRATEGIES:
        raise HarnessError("ingest.agentic.frame_extraction.strategy must be adaptive or uniform")
    for key in ("initial_interval_sec", "min_interval_sec"):
        if number(extraction[key], "frame_extraction." + key) <= 0:
            raise HarnessError(f"frame_extraction.{key} must be positive")
    if extraction["min_interval_sec"] > extraction["initial_interval_sec"]:
        raise HarnessError("frame_extraction.min_interval_sec must not exceed initial_interval_sec")

    wiki = result["wiki"]
    levels = wiki["levels"]
    if not isinstance(levels, list) or not levels or len(set(levels)) != len(levels):
        raise HarnessError("ingest.agentic.wiki.levels must be a list of distinct level names")
    if any(level not in LEVELS for level in levels):
        raise HarnessError(f"ingest.agentic.wiki.levels must be chosen from {', '.join(LEVELS)}")
    # Levels are a hierarchy, not a set: keep the coarse-to-fine order fixed so
    # the same selection always yields the same instructions.
    if levels != [level for level in LEVELS if level in levels]:
        raise HarnessError("ingest.agentic.wiki.levels must stay in chapter, event, moment order")
    for key, value in wiki.items():
        if key != "levels" and type(value) is not bool:
            raise HarnessError(f"ingest.agentic.wiki.{key} must be boolean")

    recorded = raw.get("templates_hash")
    if templates is not None:
        result["templates_hash"] = template_hash(templates)
    elif recorded is None:
        raise HarnessError("Stored agentic settings have no templates_hash; use a new wiki root")
    else:
        result["templates_hash"] = nonempty(recorded, "ingest.agentic.templates_hash")
    return result


def content_settings(ingest: dict) -> dict:
    """The agentic settings that determine what the agent is asked to produce."""
    result = settings(ingest)
    for key in _TRANSPORT_KEYS:
        result.pop(key, None)
    return result
