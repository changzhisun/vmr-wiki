from __future__ import annotations
from copy import deepcopy
import os
import re
import ipaddress
from urllib.parse import urlsplit
from vmr.core.errors import HarnessError
from vmr.core.validation import identifier, nonempty, number, positive_int

AGENT_KINDS = ("codex", "claude_code")


def _mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise HarnessError(f"{field} must be a mapping")
    return value


def _known(mapping: dict, allowed: set[str], field: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise HarnessError(f"Unknown {field} setting(s): {', '.join(sorted(unknown))}")


def _model_name(value, field: str) -> str:
    model = nonempty(value, field)
    if model in ("AGENT_MODEL", "VLM_MODEL"):
        configured = os.environ.get(model, "").strip()
        if configured:
            return configured
    return model


def _vlm_base_url(value, field: str) -> str:
    url = nonempty(value, field)
    if url == "VLM_BASE_URL":
        configured = os.environ.get("VLM_BASE_URL", "").strip()
        if configured:
            return configured
    return url


def _base_url(value, field: str, hosts_field: str, hosts: list[str]):
    if value == "AGENT_BASE_URL":
        value = os.environ.get("AGENT_BASE_URL", "").strip() or None
    if value is None:
        return None
    host = endpoint_host(value, field)
    if host not in hosts:
        raise HarnessError(f"{field} host {host!r} is not in {hosts_field}")
    return value


def agent_kind(value, field: str) -> str:
    if value not in AGENT_KINDS:
        raise HarnessError(f"{field} must be codex or claude_code")
    return value


def agent_settings(raw, field: str = "profiles.agent") -> dict:
    """Shared coding-agent deployment. Kind is selected separately."""
    profile = _mapping(raw, field)
    _known(
        profile,
        {
            "model",
            "api_key_env",
            "base_url",
            "egress_allowed_hosts",
            "container_image",
        },
        field,
    )
    try:
        hosts = profile["egress_allowed_hosts"]
        if not isinstance(hosts, list) or not hosts:
            raise HarnessError(f"{field}.egress_allowed_hosts must be a nonempty list")
        normalized_hosts = [
            hostname(host, f"{field}.egress_allowed_hosts") for host in hosts
        ]
        if len(set(normalized_hosts)) != len(normalized_hosts):
            raise HarnessError(f"{field}.egress_allowed_hosts contains duplicates")
        hosts_field = f"{field}.egress_allowed_hosts"
        return {
            "model": _model_name(profile["model"], f"{field}.model"),
            "api_key_env": identifier(
                profile.get("api_key_env", "AGENT_API_KEY"), f"{field}.api_key_env"
            ),
            "base_url": _base_url(
                profile.get("base_url"),
                f"{field}.base_url",
                hosts_field,
                normalized_hosts,
            ),
            "egress_allowed_hosts": normalized_hosts,
            "container_image": nonempty(
                profile["container_image"], f"{field}.container_image"
            ),
        }
    except KeyError as exc:
        raise HarnessError(
            f"Missing configuration field: {field}.{exc.args[0]}"
        ) from exc


def _vlm_profiles(raw: dict) -> dict[str, dict]:
    profiles = _mapping(raw, "profiles.vlms")
    result = {}
    allowed = {
        "provider",
        "model",
        "prompt",
        "base_url",
        "api_key_env",
        "generation",
        "transport",
    }
    generation_allowed = {"temperature", "max_tokens"}
    transport_allowed = {
        "timeout_sec",
        "max_retries",
        "max_concurrent_requests",
        "max_retry_delay_sec",
        "queue_timeout_sec",
    }
    transport_defaults = {
        "timeout_sec": 120,
        "max_retries": 3,
        "max_concurrent_requests": 4,
        "max_retry_delay_sec": 60,
        "queue_timeout_sec": 300,
    }
    for name, value in profiles.items():
        identifier(name, "VLM profile name")
        profile = _mapping(value, f"profiles.vlms.{name}")
        _known(profile, allowed, f"profiles.vlms.{name}")
        generation = _mapping(
            profile.get("generation", {}), f"profiles.vlms.{name}.generation"
        )
        transport = _mapping(
            profile.get("transport", {}), f"profiles.vlms.{name}.transport"
        )
        _known(generation, generation_allowed, f"profiles.vlms.{name}.generation")
        _known(transport, transport_allowed, f"profiles.vlms.{name}.transport")
        try:
            result[name] = {
                "provider": profile["provider"],
                "model": _model_name(profile["model"], f"profiles.vlms.{name}.model"),
                "prompt": profile["prompt"],
                "base_url": _vlm_base_url(
                    profile["base_url"], f"profiles.vlms.{name}.base_url"
                ),
                "api_key_env": identifier(
                    profile.get("api_key_env", "VLM_API_KEY"),
                    f"profiles.vlms.{name}.api_key_env",
                ),
                "temperature": generation.get("temperature", 0),
                "max_tokens": generation["max_tokens"],
                **transport_defaults,
                **transport,
            }
        except KeyError as exc:
            raise HarnessError(
                f"Missing configuration field: profiles.vlms.{name}.{exc.args[0]}"
            ) from exc
        vlm = result[name]
        if vlm["provider"] != "openai-compatible":
            raise HarnessError(
                f"profiles.vlms.{name}.provider must be openai-compatible"
            )
        for key in ("model", "prompt", "base_url", "api_key_env"):
            nonempty(vlm[key], f"profiles.vlms.{name}.{key}")
        number(vlm["temperature"], f"profiles.vlms.{name}.generation.temperature", 0)
        positive_int(vlm["max_tokens"], f"profiles.vlms.{name}.generation.max_tokens")
        if (
            number(vlm["timeout_sec"], f"profiles.vlms.{name}.transport.timeout_sec")
            <= 0
        ):
            raise HarnessError(
                f"profiles.vlms.{name}.transport.timeout_sec must be positive"
            )
        if type(vlm["max_retries"]) is not int or vlm["max_retries"] < 0:
            raise HarnessError(
                f"profiles.vlms.{name}.transport.max_retries must be a nonnegative integer"
            )
        positive_int(
            vlm["max_concurrent_requests"],
            f"profiles.vlms.{name}.transport.max_concurrent_requests",
        )
        for key in ("max_retry_delay_sec", "queue_timeout_sec"):
            number(vlm[key], f"profiles.vlms.{name}.transport.{key}", 0.001)
    return result


def _select_profile(profiles: dict[str, dict], name, field: str) -> dict:
    name = nonempty(name, field)
    try:
        return deepcopy(profiles[name])
    except KeyError as exc:
        raise HarnessError(f"Unknown {field}: {name!r}") from exc


def hostname(value, field: str) -> str:
    value = nonempty(value, field).lower()
    if len(value) > 253 or "." not in value or not re.fullmatch(r"[a-z0-9.-]+", value):
        raise HarnessError(f"{field} must be an exact DNS hostname")
    if any(
        not label or len(label) > 63 or label[0] == "-" or label[-1] == "-"
        for label in value.split(".")
    ):
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
        raise HarnessError(
            f"{field} must use port 443; the egress proxy tunnels no other port"
        )
    if url.username or url.password or url.query or url.fragment:
        raise HarnessError(
            f"{field} must not carry credentials, a query string, or a fragment"
        )
    return hostname(url.hostname, field)


def _agent_profiles(raw: dict) -> dict[str, dict]:
    profiles = _mapping(raw, "profiles.agents")
    result = {}
    allowed = {
        "kind",
        "model",
        "api_key_env",
        "base_url",
        "egress_allowed_hosts",
        "container_image",
    }
    for name, value in profiles.items():
        identifier(name, "agent profile name")
        profile = _mapping(value, f"profiles.agents.{name}")
        _known(profile, allowed, f"profiles.agents.{name}")
        try:
            kind = profile["kind"]
            if kind not in AGENT_KINDS:
                raise HarnessError(
                    f"profiles.agents.{name}.kind must be codex or claude_code"
                )
            hosts = profile["egress_allowed_hosts"]
            if not isinstance(hosts, list) or not hosts:
                raise HarnessError(
                    f"profiles.agents.{name}.egress_allowed_hosts must be a nonempty list"
                )
            normalized_hosts = [
                hostname(host, f"profiles.agents.{name}.egress_allowed_hosts")
                for host in hosts
            ]
            if len(set(normalized_hosts)) != len(normalized_hosts):
                raise HarnessError(
                    f"profiles.agents.{name}.egress_allowed_hosts contains duplicates"
                )
            base_url = profile.get("base_url")
            if base_url is not None:
                host = endpoint_host(base_url, f"profiles.agents.{name}.base_url")
                if host not in normalized_hosts:
                    raise HarnessError(
                        f"profiles.agents.{name}.base_url host {host!r} is not in "
                        f"profiles.agents.{name}.egress_allowed_hosts"
                    )
            result[name] = {
                "kind": kind,
                "model": nonempty(profile["model"], f"profiles.agents.{name}.model"),
                "api_key_env": identifier(
                    profile["api_key_env"], f"profiles.agents.{name}.api_key_env"
                ),
                "base_url": base_url,
                "egress_allowed_hosts": normalized_hosts,
                "container_image": nonempty(
                    profile["container_image"],
                    f"profiles.agents.{name}.container_image",
                ),
            }
        except KeyError as exc:
            raise HarnessError(
                f"Missing configuration field: profiles.agents.{name}.{exc.args[0]}"
            ) from exc
    return result


def select_named_agent(profiles, query, *, query_agent=None, query_agent_profile=None):
    _known(profiles, {"agents", "vlms"}, "profiles")
    agents = _agent_profiles(profiles.get("agents", {}))
    selected_name = query_agent_profile or query.get("agent_profile")
    if query_agent is not None:
        if query_agent not in AGENT_KINDS:
            raise HarnessError("query agent must be codex or claude_code")
        if query_agent_profile is not None:
            selected = _select_profile(agents, selected_name, "query.agent_profile")
            if selected["kind"] != query_agent:
                raise HarnessError(
                    "--agent and --agent-profile select different agent kinds"
                )
        else:
            matches = [
                name for name, value in agents.items() if value["kind"] == query_agent
            ]
            if len(matches) != 1:
                raise HarnessError(
                    f"--agent {query_agent} matches {len(matches)} profiles; use --agent-profile"
                )
            selected_name = matches[0]
    selected = _select_profile(agents, selected_name, "query.agent_profile")
    return selected
