from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.validation import relative_template

# Filenames are relative to storage.templates. input_mode selects the pair.
DEFAULT_QUERY_TEMPLATES = {
    "agents": "query_agents.video_wiki.md",
    "prompt": "query_prompt.video_wiki.md",
    "text": {
        "agents": "query_agents.video_wiki.text_only.md",
        "prompt": "query_prompt.video_wiki.text_only.md",
    },
}
VIDEO_ONLY_TEMPLATES = {
    "agents": "query_agents.video_only.md",
    "prompt": "query_prompt.video_only.md",
}


def query_templates(
    raw=None, *, text_only: bool = False, query_type: str = "video-wiki"
) -> dict[str, str]:
    """Resolve the agents/prompt files for one query input mode."""
    spec = raw if isinstance(raw, dict) else {}
    unknown = set(spec) - {"agents", "prompt", "text"}
    if unknown:
        raise HarnessError(
            f"Unknown query.templates setting(s): {', '.join(sorted(unknown))}"
        )
    value = spec.get("text", {})
    if value is not None and (
        not isinstance(value, dict) or set(value) - {"agents", "prompt"}
    ):
        raise HarnessError("query.templates.text must set agents and prompt")
    selected = (spec.get("text") or {}) if text_only else spec
    defaults = (
        DEFAULT_QUERY_TEMPLATES["text"]
        if text_only
        else VIDEO_ONLY_TEMPLATES
        if query_type == "video-only"
        else DEFAULT_QUERY_TEMPLATES
    )
    field = "query.templates.text" if text_only else "query.templates"
    return {
        "agents": relative_template(
            selected.get("agents", defaults["agents"]), f"{field}.agents"
        ),
        "prompt": relative_template(
            selected.get("prompt", defaults["prompt"]), f"{field}.prompt"
        ),
    }


def load_query_templates(
    directory: Path,
    *,
    text_only: bool = False,
    templates=None,
    query_type: str = "video-wiki",
) -> dict[str, str]:
    names = query_templates(templates, text_only=text_only, query_type=query_type)
    paths = {
        "AGENTS.md": Path(directory) / names["agents"],
        "query_prompt.md": Path(directory) / names["prompt"],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise HarnessError("Missing query template: " + ", ".join(missing))
    loaded = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    if query_type == "video-only":
        marker = "<!-- vmr-query-type: video-only -->"
        for name, content in loaded.items():
            if not content.lstrip().startswith(marker) or "video.mp4" not in content:
                raise HarnessError(
                    f"Video-only template must declare its type and refer to video.mp4: {paths[name]}"
                )
    return loaded
