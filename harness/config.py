from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
import ipaddress
import re
from pathlib import Path
import warnings
from urllib.parse import urlsplit

import yaml

from harness.common import HarnessError, identifier, nonempty, number, positive_int


AGENT_KINDS = ("codex", "claude_code")
WIKI_METHODS = ("simple", "dense", "hierarchical", "bidirectional", "agentic")


def _mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise HarnessError(f"{field} must be a mapping")
    return value


def _known(mapping: dict, allowed: set[str], field: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise HarnessError(f"Unknown {field} setting(s): {', '.join(sorted(unknown))}")


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; lists and scalar values are replaced."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_extended(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise HarnessError(f"Configuration extends cycle: {chain}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HarnessError(f"Invalid YAML in {path}: {exc}") from exc
    except OSError as exc:
        raise HarnessError(f"Could not read configuration {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HarnessError(f"Configuration {path} must be a mapping")
    raw = deepcopy(raw)
    parents = raw.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(not isinstance(item, str) or not item.strip()
                                            for item in parents):
        raise HarnessError(f"extends in {path} must be a path or list of paths")
    merged = {}
    for item in parents:
        parent = (path.parent / item).resolve()
        merged = _merge(merged, _load_extended(parent, (*stack, path)))
    return _merge(merged, raw)


def _agent_profiles(raw: dict) -> dict[str, dict]:
    profiles = _mapping(raw, "profiles.agents")
    result = {}
    allowed = {"kind", "model", "api_key_env", "base_url", "egress_allowed_hosts",
               "container_image"}
    for name, value in profiles.items():
        identifier(name, "agent profile name")
        profile = _mapping(value, f"profiles.agents.{name}")
        _known(profile, allowed, f"profiles.agents.{name}")
        try:
            kind = profile["kind"]
            if kind not in AGENT_KINDS:
                raise HarnessError(
                    f"profiles.agents.{name}.kind must be codex or claude_code")
            hosts = profile["egress_allowed_hosts"]
            if not isinstance(hosts, list) or not hosts:
                raise HarnessError(
                    f"profiles.agents.{name}.egress_allowed_hosts must be a nonempty list")
            normalized_hosts = [
                hostname(host, f"profiles.agents.{name}.egress_allowed_hosts") for host in hosts]
            if len(set(normalized_hosts)) != len(normalized_hosts):
                raise HarnessError(
                    f"profiles.agents.{name}.egress_allowed_hosts contains duplicates")
            base_url = profile.get("base_url")
            if base_url is not None:
                host = endpoint_host(base_url, f"profiles.agents.{name}.base_url")
                if host not in normalized_hosts:
                    raise HarnessError(
                        f"profiles.agents.{name}.base_url host {host!r} is not in "
                        f"profiles.agents.{name}.egress_allowed_hosts")
            result[name] = {
                "kind": kind,
                "model": nonempty(profile["model"], f"profiles.agents.{name}.model"),
                "api_key_env": identifier(
                    profile["api_key_env"], f"profiles.agents.{name}.api_key_env"),
                "base_url": base_url,
                "egress_allowed_hosts": normalized_hosts,
                "container_image": nonempty(
                    profile["container_image"], f"profiles.agents.{name}.container_image"),
            }
        except KeyError as exc:
            raise HarnessError(
                f"Missing configuration field: profiles.agents.{name}.{exc.args[0]}") from exc
    return result


def _vlm_profiles(raw: dict) -> dict[str, dict]:
    profiles = _mapping(raw, "profiles.vlms")
    result = {}
    allowed = {"provider", "model", "prompt", "base_url", "api_key_env",
               "generation", "transport"}
    generation_allowed = {"temperature", "max_tokens"}
    transport_allowed = {"timeout_sec", "max_retries", "max_concurrent_requests",
                         "max_retry_delay_sec", "queue_timeout_sec"}
    transport_defaults = {"timeout_sec": 120, "max_retries": 3,
                          "max_concurrent_requests": 4, "max_retry_delay_sec": 60,
                          "queue_timeout_sec": 300}
    for name, value in profiles.items():
        identifier(name, "VLM profile name")
        profile = _mapping(value, f"profiles.vlms.{name}")
        _known(profile, allowed, f"profiles.vlms.{name}")
        generation = _mapping(profile.get("generation", {}),
                              f"profiles.vlms.{name}.generation")
        transport = _mapping(profile.get("transport", {}),
                             f"profiles.vlms.{name}.transport")
        _known(generation, generation_allowed, f"profiles.vlms.{name}.generation")
        _known(transport, transport_allowed, f"profiles.vlms.{name}.transport")
        try:
            result[name] = {
                "provider": profile["provider"],
                "model": profile["model"],
                "prompt": profile["prompt"],
                "base_url": profile["base_url"],
                "api_key_env": profile["api_key_env"],
                "temperature": generation.get("temperature", 0),
                "max_tokens": generation["max_tokens"],
                **transport_defaults,
                **transport,
            }
        except KeyError as exc:
            raise HarnessError(
                f"Missing configuration field: profiles.vlms.{name}.{exc.args[0]}") from exc
        vlm = result[name]
        if vlm["provider"] != "openai-compatible":
            raise HarnessError(
                f"profiles.vlms.{name}.provider must be openai-compatible")
        for key in ("model", "prompt", "base_url", "api_key_env"):
            nonempty(vlm[key], f"profiles.vlms.{name}.{key}")
        number(vlm["temperature"], f"profiles.vlms.{name}.generation.temperature", 0)
        positive_int(vlm["max_tokens"], f"profiles.vlms.{name}.generation.max_tokens")
        if number(vlm["timeout_sec"], f"profiles.vlms.{name}.transport.timeout_sec") <= 0:
            raise HarnessError(f"profiles.vlms.{name}.transport.timeout_sec must be positive")
        if type(vlm["max_retries"]) is not int or vlm["max_retries"] < 0:
            raise HarnessError(
                f"profiles.vlms.{name}.transport.max_retries must be a nonnegative integer")
        positive_int(vlm["max_concurrent_requests"],
                     f"profiles.vlms.{name}.transport.max_concurrent_requests")
        for key in ("max_retry_delay_sec", "queue_timeout_sec"):
            number(vlm[key], f"profiles.vlms.{name}.transport.{key}", 0.001)
    return result


def _select_profile(profiles: dict[str, dict], name, field: str) -> dict:
    name = nonempty(name, field)
    try:
        return deepcopy(profiles[name])
    except KeyError as exc:
        raise HarnessError(f"Unknown {field}: {name!r}") from exc


def _normalize_v2(raw: dict, path: Path, *, query_agent: str | None = None,
                  query_agent_profile: str | None = None) -> dict:
    """Translate the user-facing v2 schema into the stable internal schema."""
    _known(raw, {"version", "dataset", "storage", "profiles", "wiki", "query",
                 "batch", "evaluation"}, "top-level")
    if raw.get("version") != 2:
        raise HarnessError("Configuration version must be 2")

    storage = _mapping(raw.get("storage"), "storage")
    _known(storage, {"root", "datasets", "wikis", "runs", "results", "templates"},
           "storage")
    root = Path(nonempty(storage.get("root", "."), "storage.root"))
    if not root.is_absolute():
        root = path.parent / root
    paths = {}
    for public, internal in (("datasets", "datasets"), ("wikis", "wiki"),
                             ("runs", "runs"), ("results", "results"),
                             ("templates", "templates")):
        value = Path(nonempty(storage.get(public), f"storage.{public}"))
        paths[internal] = str(value if value.is_absolute() else root / value)

    profiles = _mapping(raw.get("profiles"), "profiles")
    _known(profiles, {"agents", "vlms"}, "profiles")
    agents = _agent_profiles(profiles.get("agents", {}))
    vlms = _vlm_profiles(profiles.get("vlms", {}))

    wiki = _mapping(raw.get("wiki"), "wiki")
    _known(wiki, {"method", "media", "captioner_profile", "agent_profile",
                  "repair_attempts", "method_config"}, "wiki")
    method = wiki.get("method")
    if method not in WIKI_METHODS:
        raise HarnessError(
            "wiki.method must be simple, dense, hierarchical, bidirectional, or agentic")
    media = _mapping(wiki.get("media"), "wiki.media")
    _known(media, {"sample_interval_sec", "image_max_size", "jpeg_quality"}, "wiki.media")
    method_config = _mapping(wiki.get("method_config", {}), "wiki.method_config")
    batch = _mapping(raw.get("batch", {}), "batch")
    _known(batch, {"consecutive_failure_limit"}, "batch")
    try:
        ingest = {
            "consecutive_failure_limit": batch.get("consecutive_failure_limit", 3),
            "sample_interval_sec": media["sample_interval_sec"],
            "caption_mode": method,
            "caption_max_repairs": wiki.get("repair_attempts", 2),
            "image_max_size": media["image_max_size"],
            "jpeg_quality": media["jpeg_quality"],
            "caption_window_frames": 1,
            "caption_stride_frames": 1,
            "dense_timestamp_mode": "absolute_seconds",
        }
    except KeyError as exc:
        raise HarnessError(f"Missing configuration field: wiki.media.{exc.args[0]}") from exc

    if method == "agentic":
        if "captioner_profile" in wiki:
            raise HarnessError("wiki.captioner_profile is not used by the agentic method")
        profile = _select_profile(agents, wiki.get("agent_profile"), "wiki.agent_profile")
        injected = {"agent", "model", "container_image", "api_key_env", "base_url",
                    "egress_allowed_hosts"}
        if set(method_config) & injected:
            raise HarnessError(
                "wiki.method_config must not repeat settings owned by wiki.agent_profile")
        ingest["agentic"] = {
            **deepcopy(method_config),
            "agent": profile["kind"],
            "model": profile["model"],
            "container_image": profile["container_image"],
            "api_key_env": {profile["kind"]: profile["api_key_env"]},
            "base_url": {profile["kind"]: profile["base_url"]},
            "egress_allowed_hosts": {
                profile["kind"]: profile["egress_allowed_hosts"]},
        }
    else:
        if "agent_profile" in wiki:
            raise HarnessError("wiki.agent_profile is only valid for the agentic method")
        ingest["vlm"] = _select_profile(
            vlms, wiki.get("captioner_profile"), "wiki.captioner_profile")
        if method in ("simple", "dense"):
            _known(method_config, {"window_frames", "stride_frames", "timestamp_mode"},
                   "wiki.method_config")
            ingest["caption_window_frames"] = method_config.get("window_frames", 1)
            ingest["caption_stride_frames"] = method_config.get("stride_frames", 1)
            ingest["dense_timestamp_mode"] = method_config.get(
                "timestamp_mode", "absolute_seconds")
        elif method == "hierarchical":
            ingest["hierarchy"] = deepcopy(method_config)
        elif method == "bidirectional":
            ingest["bidirectional"] = deepcopy(method_config)

    query = _mapping(raw.get("query"), "query")
    _known(query, {"agent_profile", "input_mode", "max_predictions", "timeout_sec"}, "query")
    selected_name = query_agent_profile or query.get("agent_profile")
    if query_agent is not None:
        if query_agent not in AGENT_KINDS:
            raise HarnessError("query agent must be codex or claude_code")
        if query_agent_profile is not None:
            selected = _select_profile(agents, selected_name, "query.agent_profile")
            if selected["kind"] != query_agent:
                raise HarnessError("--agent and --agent-profile select different agent kinds")
        else:
            matches = [name for name, value in agents.items() if value["kind"] == query_agent]
            if len(matches) != 1:
                raise HarnessError(
                    f"--agent {query_agent} matches {len(matches)} profiles; use --agent-profile")
            selected_name = matches[0]
    selected = _select_profile(agents, selected_name, "query.agent_profile")
    input_mode = query.get("input_mode", "multimodal")
    if input_mode not in ("text", "multimodal"):
        raise HarnessError("query.input_mode must be text or multimodal")
    try:
        internal_query = {
            "agent": selected["kind"],
            "text_only": input_mode == "text",
            "model": selected["model"],
            "max_predictions": query["max_predictions"],
            "timeout_sec": query["timeout_sec"],
            "container_image": selected["container_image"],
            "api_key_env": {selected["kind"]: selected["api_key_env"]},
            "base_url": {selected["kind"]: selected["base_url"]},
            "egress_allowed_hosts": {
                selected["kind"]: selected["egress_allowed_hosts"]},
        }
    except KeyError as exc:
        raise HarnessError(f"Missing configuration field: query.{exc.args[0]}") from exc
    return {
        "dataset": deepcopy(raw.get("dataset")),
        "paths": paths,
        "ingest": ingest,
        "query": internal_query,
        "evaluation": deepcopy(raw.get("evaluation")),
    }


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


def load_config(path: str | Path = "config.yaml", *, query_agent: str | None = None,
                query_model: str | None = None,
                query_agent_profile: str | None = None,
                warn_legacy: bool = True) -> dict:
    path = Path(path).resolve()
    raw = _load_extended(path)
    version = raw.get("version", 1)
    if type(version) is not int or version not in (1, 2):
        raise HarnessError(f"Unsupported configuration version: {version!r}")
    if version == 2:
        cfg = _normalize_v2(raw, path, query_agent=query_agent,
                            query_agent_profile=query_agent_profile)
    else:
        if query_agent_profile is not None:
            raise HarnessError("--agent-profile requires a version 2 configuration")
        cfg = deepcopy(raw)
        cfg.pop("version", None)
        if query_agent is not None:
            try:
                cfg["query"]["agent"] = query_agent
            except (KeyError, TypeError) as exc:
                raise HarnessError(f"Missing or malformed configuration field: {exc}") from exc
        if warn_legacy:
            warnings.warn(
                "Version 1 configuration is deprecated; migrate to version: 2",
                DeprecationWarning,
                stacklevel=2,
            )
    if query_model is not None:
        cfg["query"]["model"] = query_model
    try:
        identifier(cfg["dataset"]["name"], "dataset.name")
        if cfg["dataset"].get("split") is not None:
            nonempty(cfg["dataset"]["split"], "dataset.split")
        for name in ("datasets", "wiki", "runs", "results", "templates"):
            cfg["paths"][name] = str((path.parent / nonempty(cfg["paths"][name], name)).resolve())
        ingest = cfg["ingest"]
        ingest.setdefault("consecutive_failure_limit", 3)
        positive_int(ingest["consecutive_failure_limit"], "ingest.consecutive_failure_limit")
        if number(ingest["sample_interval_sec"], "sample_interval_sec") <= 0:
            raise HarnessError("sample_interval_sec must be positive")
        ingest.setdefault("caption_mode", "simple")
        if ingest["caption_mode"] not in WIKI_METHODS:
            raise HarnessError(
                "caption_mode must be simple, dense, hierarchical, bidirectional, or agentic")
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
        if ingest["caption_mode"] == "agentic":
            from harness.agentic_config import AGENTIC_VERSION, settings
            if ingest.get("agentic_version", AGENTIC_VERSION) != AGENTIC_VERSION:
                raise HarnessError("Unsupported agentic_version; use a new wiki root")
            ingest["agentic_version"] = AGENTIC_VERSION
            ingest["agentic"] = settings(ingest, Path(cfg["paths"]["templates"]))
            agentic = ingest["agentic"]
            agent = agentic["agent"]
            hosts = agentic["egress_allowed_hosts"].get(agent)
            if not isinstance(hosts, list) or not hosts:
                raise HarnessError(
                    f"ingest.agentic.egress_allowed_hosts.{agent} must be a nonempty list")
            normalized = [hostname(host, f"ingest.agentic.egress_allowed_hosts.{agent}")
                          for host in hosts]
            if len(set(normalized)) != len(normalized):
                raise HarnessError(f"ingest.agentic.egress_allowed_hosts.{agent} contains duplicates")
            agentic["egress_allowed_hosts"] = {agent: normalized}
            agentic["base_url"] = {agent: agentic["base_url"].get(agent)}
            agentic["api_key_env"] = {agent: agentic["api_key_env"][agent]}
            if agentic["base_url"][agent] is not None:
                host = endpoint_host(agentic["base_url"][agent], f"ingest.agentic.base_url.{agent}")
                if host not in normalized:
                    raise HarnessError(
                        f"ingest.agentic.base_url.{agent} host {host!r} is not in "
                        f"ingest.agentic.egress_allowed_hosts.{agent}; the agent could not reach it")
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
        # A coding agent brings its own model and endpoint, so an agentic wiki
        # root must not require a VLM deployment it never calls.
        if ingest["caption_mode"] != "agentic":
            vlm = ingest["vlm"]
            vlm.setdefault("max_concurrent_requests", 4)
            vlm.setdefault("max_retry_delay_sec", 60)
            vlm.setdefault("queue_timeout_sec", 300)
            positive_int(vlm["max_concurrent_requests"], "vlm.max_concurrent_requests")
            for key in ("max_retry_delay_sec", "queue_timeout_sec"):
                number(vlm[key], "vlm." + key, 0.001)
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
        query.setdefault("text_only", False)
        if type(query["text_only"]) is not bool:
            raise HarnessError("query.text_only must be a boolean")
        if query["agent"] not in AGENT_KINDS:
            raise HarnessError("query.agent must be codex or claude_code")
        agent = query["agent"]
        nonempty(query["model"], "query.model")
        positive_int(query["max_predictions"], "max_predictions")
        if number(query["timeout_sec"], "query.timeout_sec") <= 0:
            raise HarnessError("query.timeout_sec must be positive")
        nonempty(query["container_image"], "container_image")
        if not isinstance(query.get("api_key_env"), dict):
            raise HarnessError("query.api_key_env must be a mapping of agent to environment variable")
        if not isinstance(query.get("egress_allowed_hosts"), dict):
            raise HarnessError("query.egress_allowed_hosts must be a mapping of agent to hosts")
        query.setdefault("base_url", {})
        if not isinstance(query["base_url"], dict):
            raise HarnessError("query.base_url must be a mapping of agent to URL")
        identifier(query["api_key_env"][agent], f"query.api_key_env.{agent}")
        hosts = query["egress_allowed_hosts"][agent]
        if not isinstance(hosts, list) or not hosts:
            raise HarnessError(f"query.egress_allowed_hosts.{agent} must be a nonempty list")
        normalized = [hostname(host, f"query.egress_allowed_hosts.{agent}") for host in hosts]
        if len(set(normalized)) != len(normalized):
            raise HarnessError(f"query.egress_allowed_hosts.{agent} contains duplicates")
        base_url = query["base_url"].get(agent)
        if base_url is not None:
            # A gateway the proxy would refuse is a misconfiguration, not an
            # agent failure: the agent only sees an opaque 403 from the proxy.
            host = endpoint_host(base_url, f"query.base_url.{agent}")
            if host not in normalized:
                raise HarnessError(
                    f"query.base_url.{agent} host {host!r} is not in "
                    f"query.egress_allowed_hosts.{agent}; the agent could not reach it")
        # Downstream runners only consume the selected deployment. Keeping
        # inactive agent mappings would reintroduce the v1 ambiguity.
        query["api_key_env"] = {agent: query["api_key_env"][agent]}
        query["egress_allowed_hosts"] = {agent: normalized}
        query["base_url"] = {agent: base_url}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a VMR Wiki configuration")
    subparsers = parser.add_subparsers(dest="command", required=True)
    explain = subparsers.add_parser("explain", help="Print the normalized runtime configuration")
    explain.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    from harness.common import ingest_content_hash
    print(f"wiki_content_hash: {ingest_content_hash(cfg)}")
    print("normalized_config:")
    print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).rstrip())


if __name__ == "__main__":
    from harness.common import cli
    cli(main)
