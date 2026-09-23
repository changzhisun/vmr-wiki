from __future__ import annotations
from pathlib import Path
from copy import deepcopy
import yaml
from vmr.core.errors import HarnessError

AGENT_KINDS = ("codex", "claude_code")


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
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or not item.strip() for item in parents
    ):
        raise HarnessError(f"extends in {path} must be a path or list of paths")
    merged = {}
    for item in parents:
        parent = (path.parent / item).resolve()
        merged = _merge(merged, _load_extended(parent, (*stack, path)))
    return _merge(merged, raw)
