from __future__ import annotations

from copy import deepcopy
import ipaddress
import re
from pathlib import Path

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
        ingest.setdefault("caption_window_frames", 1)
        ingest.setdefault("caption_stride_frames", 1)
        positive_int(ingest["caption_window_frames"], "caption_window_frames")
        positive_int(ingest["caption_stride_frames"], "caption_stride_frames")
        positive_int(ingest["image_max_size"], "image_max_size")
        if not 2 <= positive_int(ingest["jpeg_quality"], "jpeg_quality") <= 31:
            raise HarnessError("jpeg_quality must be in [2, 31]")
        vlm = ingest["vlm"]
        if vlm["provider"] != "openai-compatible":
            raise HarnessError("Supported VLM provider: openai-compatible")
        for key in ("model", "prompt", "base_url", "api_key_env"):
            nonempty(vlm[key], f"vlm.{key}")
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
        for agent in ("codex", "claude_code"):
            identifier(query["api_key_env"][agent], "api_key_env")
            hosts = query["egress_allowed_hosts"][agent]
            if not isinstance(hosts, list) or not hosts:
                raise HarnessError(f"query.egress_allowed_hosts.{agent} must be a nonempty list")
            normalized = [hostname(host, f"query.egress_allowed_hosts.{agent}") for host in hosts]
            if len(set(normalized)) != len(normalized):
                raise HarnessError(f"query.egress_allowed_hosts.{agent} contains duplicates")
            query["egress_allowed_hosts"][agent] = normalized
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
