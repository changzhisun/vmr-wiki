"""Query-only config loading. No compiler settings are resolved or validated."""

from pathlib import Path
from .loader import _load_extended
from .resolve import agent_kind, agent_settings, select_named_agent, _known
from .models import QueryConfig, StorageConfig, AppConfig, DatasetConfig
from vmr.core.errors import HarnessError
from vmr.query.templates import query_templates


def from_legacy_query(q, templates=None):
    agent = q["agent"]
    spec = templates if templates is not None else q.get("templates")
    names = query_templates(spec, text_only=False)
    text_names = query_templates(spec, text_only=True)
    return QueryConfig(
        agent=agent,
        model=q["model"],
        container_image=q["container_image"],
        api_key_env=q["api_key_env"][agent],
        base_url=q.get("base_url", {}).get(agent),
        egress_allowed_hosts=tuple(q["egress_allowed_hosts"][agent]),
        input_mode="text" if q.get("text_only", False) else "multimodal",
        max_predictions=q["max_predictions"],
        timeout_sec=q["timeout_sec"],
        agents_template=names["agents"],
        prompt_template=names["prompt"],
        text_agents_template=text_names["agents"],
        text_prompt_template=text_names["prompt"],
    )


def load_query_config(path):
    path = Path(path).resolve()
    raw = _load_extended(path)
    version = raw.get("version", 1)
    if type(version) is not int or version not in (1, 2, 3):
        raise HarnessError("Unsupported configuration version")
    if version == 1:
        query = from_legacy_query(raw["query"])
        paths = raw["paths"]
        storage = dict(
            root=".",
            artifacts=paths.get("artifacts", "artifacts"),
            runs=paths["runs"],
            results=paths["results"],
            templates=paths["templates"],
            datasets=paths["datasets"],
        )
    else:
        q = raw["query"]
        profiles = raw["profiles"]
        named = version == 2 and "agents" in profiles
        _known(profiles, {"agents" if named else "agent", "vlms"}, "profiles")
        _known(
            q,
            {
                "agent_profile" if named else "kind",
                "input_mode",
                "max_predictions",
                "timeout_sec",
                "templates",
            },
            "query",
        )
        if named:
            profile = select_named_agent(profiles, q)
            kind = profile["kind"]
        else:
            kind = agent_kind(q["kind"], "query.kind")
            profile = agent_settings(profiles["agent"])
        names = query_templates(q.get("templates"), text_only=False)
        text_names = query_templates(q.get("templates"), text_only=True)
        query = QueryConfig(
            agent=kind,
            model=profile["model"],
            container_image=profile["container_image"],
            api_key_env=profile["api_key_env"],
            base_url=profile["base_url"],
            egress_allowed_hosts=tuple(profile["egress_allowed_hosts"]),
            input_mode=q.get("input_mode", "multimodal"),
            max_predictions=q.get("max_predictions", 5),
            timeout_sec=q.get("timeout_sec", 600),
            agents_template=names["agents"],
            prompt_template=names["prompt"],
            text_agents_template=text_names["agents"],
            text_prompt_template=text_names["prompt"],
        )
        storage = raw.get("storage", {})
    root = (path.parent / storage.get("root", ".")).resolve()
    resolved = StorageConfig(
        root=root,
        **{
            key: (root / storage.get(key, default)).resolve()
            for key, default in dict(
                artifacts="artifacts",
                runs="runs",
                results="results",
                templates="templates",
                datasets="datasets",
            ).items()
        },
    )
    return AppConfig(
        storage=resolved, dataset=DatasetConfig(**raw.get("dataset", {})), query=query
    )
