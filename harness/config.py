from __future__ import annotations

from copy import deepcopy
import ipaddress
import re
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from harness.common import HarnessError, identifier, nonempty, number, positive_int


def hostname(value, field: str) -> str:
    value = nonempty(value, field).lower()
    if len(value) > 253 or "." not in value or not re.fullmatch(r"[a-z0-9.-]+", value):
        raise HarnessError(f"{field} must be an exact DNS hostname")
    if any(not label or len(label) > 63 or label[0] == "-" or label[-1] == "-"
           for label in value.split(".")):
        raise HarnessError(f"{field} must be an exact DNS hostname")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return value
    raise HarnessError(f"{field} must not be an IP address")


def endpoint_host(value, field: str) -> str:
    """Return the host of an agent API endpoint the egress proxy can reach.

    The proxy only tunnels CONNECT to port 443, so any other scheme or port
    is rejected here rather than surfacing as a proxy 403 inside the agent.
    """
    url = urlsplit(nonempty(value, field))
    if url.scheme != "https":
        raise HarnessError(f"{field} must be an https URL")
    try:
        port = url.port
    except ValueError as exc:
        raise HarnessError(f"{field} has an invalid port") from exc
    if port not in (None, 443):
        raise HarnessError(f"{field} must use port 443; the egress proxy tunnels no other port")
    if url.username or url.password or url.query or url.fragment:
        raise HarnessError(f"{field} must not carry credentials, a query string, or a fragment")
    return hostname(url.hostname, field)


def load_config(path: str | Path = "config.yaml") -> dict:
    path = Path(path).resolve()
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HarnessError(f"Invalid YAML: {exc}") from exc
    if not isinstance(cfg, dict):
        raise HarnessError("Configuration must be a mapping")
    cfg = deepcopy(cfg)
    try:
        identifier(cfg["dataset"]["name"], "dataset.name")
        if cfg["dataset"].get("split") is not None:
            nonempty(cfg["dataset"]["split"], "dataset.split")
        for name in ("datasets", "wiki", "runs", "results", "templates"):
            cfg["paths"][name] = str((path.parent / nonempty(cfg["paths"][name], name)).resolve())
        ingest = cfg["ingest"]
        if number(ingest["sample_interval_sec"], "sample_interval_sec") <= 0:
            raise HarnessError("sample_interval_sec must be positive")
        ingest.setdefault("caption_mode", "simple")
        if ingest["caption_mode"] not in ("simple", "dense", "hierarchical", "bidirectional"):
            raise HarnessError("caption_mode must be simple, dense, hierarchical, or bidirectional")
        if ingest["caption_mode"] == "bidirectional":
            from harness.bidirectional_config import PIPELINE_VERSION, settings
            if ingest.get("pipeline_version", PIPELINE_VERSION) != PIPELINE_VERSION:
                raise HarnessError("Unsupported bidirectional pipeline_version")
            ingest["pipeline_version"] = PIPELINE_VERSION
            ingest["bidirectional"] = settings(ingest)
            cache_dir = ingest["bidirectional"]["shared_cache_dir"]
            if cache_dir is not None:
                ingest["bidirectional"]["shared_cache_dir"] = str((path.parent / cache_dir).resolve())
        if ingest["caption_mode"] == "hierarchical":
            from harness.hierarchy_config import HIERARCHY_VERSION, settings
            if ingest.get("hierarchy_processing_version", HIERARCHY_VERSION) != HIERARCHY_VERSION:
                raise HarnessError("Unsupported hierarchy_processing_version; use a new wiki root")
            ingest["hierarchy_processing_version"] = HIERARCHY_VERSION
            ingest["hierarchy"] = settings(ingest)
        ingest.setdefault("dense_timestamp_mode", "absolute_seconds")
        if ingest["dense_timestamp_mode"] not in ("absolute_seconds", "frame_index"):
            raise HarnessError("dense_timestamp_mode must be absolute_seconds or frame_index")
        # Code-owned version: cannot silently run new rules under an old identity.
        if ingest.get("caption_processing_version", 4) != 4:
            raise HarnessError("caption_processing_version is unsupported; use a new wiki root")
        ingest["caption_processing_version"] = 4
        ingest.setdefault("caption_window_frames", 1)
        ingest.setdefault("caption_stride_frames", 1)
        positive_int(ingest["caption_window_frames"], "caption_window_frames")
        positive_int(ingest["caption_stride_frames"], "caption_stride_frames")
        if (ingest["caption_mode"] == "dense" and ingest["caption_window_frames"] > 1
                and ingest["caption_stride_frames"] > ingest["caption_window_frames"] // 2):
            raise HarnessError(
                "dense caption_stride_frames must not exceed half caption_window_frames "
                "so each target stays inside its context window"
            )
        ingest.setdefault("caption_max_repairs", 2)
        if type(ingest["caption_max_repairs"]) is not int or ingest["caption_max_repairs"] < 0:
            raise HarnessError("caption_max_repairs must be a nonnegative integer")
        positive_int(ingest["image_max_size"], "image_max_size")
        if not 2 <= positive_int(ingest["jpeg_quality"], "jpeg_quality") <= 31:
            raise HarnessError("jpeg_quality must be in [2, 31]")
        vlm = ingest["vlm"]
        if vlm["provider"] != "openai-compatible":
            raise HarnessError("Supported VLM provider: openai-compatible")
        for key in ("model", "prompt", "base_url", "api_key_env"):
            nonempty(vlm[key], f"vlm.{key}")
        timeline_fields = vlm["prompt"].count("{{FRAME_TIMESTAMPS}}")
        if ingest["caption_mode"] == "dense" and timeline_fields != 1:
            raise HarnessError(
                "dense caption prompt must contain exactly one {{FRAME_TIMESTAMPS}} placeholder"
            )
        if ingest["caption_mode"] == "simple" and timeline_fields:
            raise HarnessError(
                "simple caption prompt must not contain {{FRAME_TIMESTAMPS}}"
            )
        if ingest["caption_mode"] in ("hierarchical", "bidirectional") and timeline_fields:
            raise HarnessError(
                "hierarchical builds its own frame timeline; prompt must not contain {{FRAME_TIMESTAMPS}}"
            )
        number(vlm["temperature"], "temperature", 0)
        positive_int(vlm["max_tokens"], "max_tokens")
        if number(vlm["timeout_sec"], "vlm.timeout_sec") <= 0:
            raise HarnessError("vlm.timeout_sec must be positive")
        if type(vlm["max_retries"]) is not int or vlm["max_retries"] < 0:
            raise HarnessError("max_retries must be a nonnegative integer")
        query = cfg["query"]
        if query["agent"] not in ("codex", "claude_code"):
            raise HarnessError("query.agent must be codex or claude_code")
        nonempty(query["model"], "query.model")
        positive_int(query["max_predictions"], "max_predictions")
        if number(query["timeout_sec"], "query.timeout_sec") <= 0:
            raise HarnessError("query.timeout_sec must be positive")
        nonempty(query["container_image"], "container_image")
        query.setdefault("base_url", {})
        if not isinstance(query["base_url"], dict):
            raise HarnessError("query.base_url must be a mapping of agent to URL")
        for agent in ("codex", "claude_code"):
            identifier(query["api_key_env"][agent], "api_key_env")
            hosts = query["egress_allowed_hosts"][agent]
            if not isinstance(hosts, list) or not hosts:
                raise HarnessError(f"query.egress_allowed_hosts.{agent} must be a nonempty list")
            normalized = [hostname(host, f"query.egress_allowed_hosts.{agent}") for host in hosts]
            if len(set(normalized)) != len(normalized):
                raise HarnessError(f"query.egress_allowed_hosts.{agent} contains duplicates")
            query["egress_allowed_hosts"][agent] = normalized
            query["base_url"].setdefault(agent, None)
            if query["base_url"][agent] is None:
                continue
            # A gateway the proxy would refuse is a misconfiguration, not an
            # agent failure: the agent only sees an opaque 403 from the proxy.
            host = endpoint_host(query["base_url"][agent], f"query.base_url.{agent}")
            if host not in normalized:
                raise HarnessError(
                    f"query.base_url.{agent} host {host!r} is not in "
                    f"query.egress_allowed_hosts.{agent}; the agent could not reach it")
        ev = cfg["evaluation"]
        identifier(ev["evaluator"], "evaluator")
        if not ev["top_k"] or not ev["iou_thresholds"]:
            raise HarnessError("Evaluation thresholds and top_k cannot be empty")
        for k in ev["top_k"]:
            positive_int(k, "top_k")
        for t in ev["iou_thresholds"]:
            if not 0 < number(t, "iou_threshold") <= 1:
                raise HarnessError("IoU thresholds must be in (0, 1]")
    except (KeyError, TypeError) as exc:
        raise HarnessError(f"Missing or malformed configuration field: {exc}") from exc
    return cfg


def dataset_path(cfg: dict, kind: str) -> Path:
    return Path(cfg["paths"][kind]) / identifier(cfg["dataset"]["name"])
