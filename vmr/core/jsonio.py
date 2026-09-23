from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any
from .errors import HarnessError
from .hashing import canonical


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise HarnessError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    def bad_constant(value):
        raise HarnessError(f"Invalid JSON constant: {value}")

    try:
        return json.loads(text, parse_constant=bad_constant, object_pairs_hook=_object)
    except json.JSONDecodeError as exc:
        raise HarnessError(f"Invalid JSON: {exc}") from exc


def read_json(path: Path) -> Any:
    return parse_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = parse_json(line)
            if not isinstance(row, dict):
                raise HarnessError("Expected an object")
            rows.append(row)
        except HarnessError as exc:
            raise HarnessError(f"{path}:{line_no}: {exc}") from exc
    return rows


def atomic_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    atomic_text(
        path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def write_jsonl(path: Path, rows) -> None:
    atomic_text(path, "".join(canonical(row).decode() + "\n" for row in rows))
