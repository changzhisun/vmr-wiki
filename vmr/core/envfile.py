"""Load a local .env into the process environment. Existing variables win."""

from pathlib import Path
import os


def load_env_file(path: Path | None = None) -> None:
    """Read KEY=VALUE lines from ``path`` (default: ``./.env``)."""
    path = Path.cwd() / ".env" if path is None else Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = _assignment(line)
        if key is not None and key not in os.environ:
            os.environ[key] = value


def _assignment(line: str) -> tuple[str | None, str | None]:
    text = line.strip()
    if not text or text.startswith("#"):
        return None, None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None, None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key.isidentifier():
        return None, None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value
