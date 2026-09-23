from __future__ import annotations
import math
import re
from typing import Any
from .errors import HarnessError


def identifier(value: Any, field: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", value
    ):
        raise HarnessError(f"{field} must be a safe, nonempty identifier: {value!r}")
    return value


def number(value: Any, field: str, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise HarnessError(f"{field} must be a finite number")
    if minimum is not None and value < minimum:
        raise HarnessError(f"{field} must be >= {minimum}")
    return float(value)


def positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise HarnessError(f"{field} must be a positive integer")
    return value


def nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{field} must be a nonempty string")
    return value


def relative_template(value: Any, field: str) -> str:
    """A template path relative to the configured templates directory."""
    from pathlib import Path

    name = nonempty(value, field).strip()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise HarnessError(f"{field} must stay inside the templates directory")
    return path.as_posix()


def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        value = identifier(row.get(key), key)
        if value in result:
            raise HarnessError(f"Duplicate {key}: {value}")
        result[value] = row
    return result


def cli(main) -> None:
    try:
        main()
    except (HarnessError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc
