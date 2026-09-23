from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.validation import relative_template

# Filenames are relative to storage.templates. input_mode selects the pair.
DEFAULT_QUERY_TEMPLATES = {
    "agents": "query_agents.md",
    "prompt": "query_prompt.md",
    "text": {
        "agents": "query_agents_text_only.md",
        "prompt": "query_prompt_text_only.md",
    },
}


def query_templates(raw=None, *, text_only: bool = False) -> dict[str, str]:
    """Resolve the agents/prompt files for one query input mode."""
    spec = raw if isinstance(raw, dict) else {}
    unknown = set(spec) - {"agents", "prompt", "text"}
    if unknown:
        raise HarnessError(
            f"Unknown query.templates setting(s): {', '.join(sorted(unknown))}"
        )
    text = spec.get("text", {})
    if text is None:
        text = {}
    if not isinstance(text, dict) or set(text) - {"agents", "prompt"}:
        raise HarnessError("query.templates.text must set agents and prompt")
    selected = text if text_only else spec
    defaults = DEFAULT_QUERY_TEMPLATES["text"] if text_only else DEFAULT_QUERY_TEMPLATES
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
    directory: Path, *, text_only: bool = False, templates=None
) -> dict[str, str]:
    names = query_templates(templates, text_only=text_only)
    paths = {
        "AGENTS.md": Path(directory) / names["agents"],
        "query_prompt.md": Path(directory) / names["prompt"],
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise HarnessError("Missing query template: " + ", ".join(missing))
    return {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
