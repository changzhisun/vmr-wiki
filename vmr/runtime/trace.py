import json
import re
from pathlib import Path
from collections import deque
from typing import Iterable

_DATA_URL = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+",
)
_IMAGE_PAYLOAD_CHARS = 256


def trace_path(stdout: Path) -> Path:
    """Return the raw event-stream path paired with a readable stdout log."""
    suffix = ".stdout.log"
    if stdout.name.endswith(suffix):
        return stdout.with_name(stdout.name[: -len(suffix)] + ".trace.jsonl")
    return stdout.with_name(stdout.name + ".trace.jsonl")


def _omitted_image(media_type: str, payload: str) -> str:
    return f"<omitted {media_type}, {len(payload)} base64 chars>"


def _remember_tool_paths(value, paths: dict) -> None:
    if isinstance(value, dict):
        if value.get("type") == "tool_use" and isinstance(value.get("input"), dict):
            tool_id = value.get("id")
            for key in ("file_path", "path", "image_path"):
                item = value["input"].get(key)
                if isinstance(tool_id, str) and isinstance(item, str) and item:
                    paths[tool_id] = item
                    break
        for item in value.values():
            _remember_tool_paths(item, paths)
    elif isinstance(value, list):
        for item in value:
            _remember_tool_paths(item, paths)


def _redact_images(value, paths: dict, path: str | None):
    """Return ``(value, changed)`` with embedded image bytes removed."""
    if isinstance(value, dict):
        if value.get("type") == "tool_result":
            path = paths.get(value.get("tool_use_id"), path)
        changed = False
        redacted = {}
        source_image = (
            value.get("type") == "base64"
            and isinstance(value.get("data"), str)
            and len(value["data"]) >= _IMAGE_PAYLOAD_CHARS
        )
        file_image = (
            isinstance(value.get("base64"), str)
            and len(value["base64"]) >= _IMAGE_PAYLOAD_CHARS
            and (
                str(value.get("type", "")).startswith("image/")
                or "originalSize" in value
            )
        )
        for key, item in value.items():
            if source_image and key == "data":
                redacted[key] = path or _omitted_image(
                    str(value.get("media_type") or "image"), item
                )
                changed = True
            elif file_image and key == "base64":
                redacted[key] = path or _omitted_image(
                    str(value.get("type") or "image"), item
                )
                changed = True
            else:
                redacted[key], nested = _redact_images(item, paths, path)
                changed = changed or nested
        return redacted, changed
    if isinstance(value, list):
        changed = False
        items = []
        for item in value:
            redacted, nested = _redact_images(item, paths, path)
            items.append(redacted)
            changed = changed or nested
        return items, changed
    if isinstance(value, str) and "data:image" in value and ";base64," in value:
        replaced, count = _DATA_URL.subn("<omitted data-url>", value)
        if count:
            return replaced, True
    return value, False


def _event_image_path(event, paths: dict) -> str | None:
    message = event.get("message") if isinstance(event, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    found = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                item = paths.get(block.get("tool_use_id"))
                if item and item not in found:
                    found.append(item)
    return found[0] if len(found) == 1 else None


def redact_trace_line(line: bytes, paths: dict | None = None) -> bytes:
    """Drop image payloads from one CLI event. Other lines stay unchanged.

    ``paths`` remembers ``tool_use`` file paths so a later image result can
    record that path instead of the bytes. Pass the same dict for a stream.
    """
    paths = {} if paths is None else paths
    raw = line.strip()
    if not raw:
        return line
    if (
        b"base64" not in raw
        and b"data:image" not in raw
        and b"file_path" not in raw
        and b"tool_use" not in raw
    ):
        return line
    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        text = line.decode("utf-8", errors="replace")
        replaced, count = _DATA_URL.subn("<omitted data-url>", text)
        return replaced.encode("utf-8") if count else line
    if isinstance(event, (dict, list)):
        _remember_tool_paths(event, paths)
    if b"base64" not in raw and b"data:image" not in raw:
        return line
    redacted, changed = _redact_images(event, paths, _event_image_path(event, paths))
    if not changed:
        return line
    return (
        json.dumps(redacted, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _trace_lines(raw: bytes | Path) -> Iterable[bytes]:
    if isinstance(raw, Path):
        with raw.open("rb") as stream:
            yield from stream
    else:
        yield from raw.splitlines()


def _event_label(event: dict) -> str:
    label = str(event.get("type") or "unknown")
    subtype = event.get("subtype")
    return f"{label}/{subtype}" if subtype else label


def readable_trace(agent: str, raw: bytes | Path) -> bytes:
    """Stream a trace and extract its final answer or a compact diagnostic.

    Codex and Claude emit different JSONL event schemas. The complete stream is
    the audit artifact; stdout remains a small human-readable diagnostic.
    """
    final_answer = None
    assistant_text = None
    fallback: deque[str] = deque(maxlen=32)
    last_event = None
    for raw_line in _trace_lines(raw):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            fallback.append(line[-8192:])
            continue
        if not isinstance(event, dict) or event.get("type") == "harness.input":
            continue
        last_event = _event_label(event)
        if agent == "claude_code" and event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str) and result:
                final_answer = result
        elif agent == "claude_code" and event.get("type") == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                texts = [
                    block.get("text")
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block.get("text")
                ]
                if texts:
                    assistant_text = "\n".join(texts)
        elif agent == "codex" and event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    final_answer = text
    if final_answer:
        text = final_answer
    elif assistant_text:
        text = assistant_text
    elif fallback:
        text = "\n".join(fallback)
    elif last_event:
        text = f"[no final answer event; last agent event: {last_event}]"
    else:
        text = "[no agent events were emitted]"
    return (text + ("\n" if text else "")).encode("utf-8")
