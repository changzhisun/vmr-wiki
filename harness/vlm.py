"""Fixed, query-independent image captioning via a Chat Completions-compatible API."""
from __future__ import annotations

import base64
import json
import os
import re
import time
import threading
import urllib.error
import urllib.request
from collections.abc import Sequence
from collections import OrderedDict
from pathlib import Path

from harness.common import HarnessError, nonempty

# OpenAI-compatible servers use "stop". Some Qwen/vLLM builds use eos/end_turn
# or omit the field when the message is already complete.
_COMPLETE_REASONS = frozenset({None, "stop", "eos", "end_turn"})
_REFUSAL_REASONS = frozenset({"content_filter", "safety", "refuse"})
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


def _visible_caption(text: str) -> str:
    """Drop Qwen3 reasoning from the returned caption.

    An unclosed ``<think>`` block contains no usable visible answer.
    """
    stripped = text.strip()
    if re.search(r"<think>", stripped, re.IGNORECASE):
        if not re.search(r"</think>", stripped, re.IGNORECASE):
            return ""
        stripped = _THINK_BLOCK.sub("", stripped).strip()
    return stripped


def _choice_text(choice: dict) -> str:
    message = choice.get("message")
    if not isinstance(message, dict):
        raise TypeError("VLM message")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    if not isinstance(content, str):
        raise TypeError("VLM caption")
    return _visible_caption(content)


def _request_extras(model: str) -> tuple[str, dict]:
    """Qwen3 thinks by default and will fill max_tokens with reasoning."""
    extras: dict = {}
    suffix = ""
    if "qwen3" in model.lower():
        suffix = "\n/no_think"
        extras = {
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False, "thinking": False},
        }
    return suffix, extras


class VLMClient:
    def __init__(self, config: dict, *, timestamp_mode: str = "absolute_seconds"):
        self.config = config
        self.timestamp_mode = timestamp_mode
        self._local = threading.local()
        self._images = OrderedDict()
        self._image_bytes = 0
        self._cache_lock = threading.Lock()
        self.key = os.environ.get(config["api_key_env"])
        if not self.key:
            raise HarnessError(f"Set {config['api_key_env']} before ingest")
        if config["model"].startswith("REPLACE_"):
            raise HarnessError("Configure an explicit VLM model before ingest")

    @property
    def last_requests(self) -> list[dict]:
        return getattr(self._local, "requests", [])

    def _image_url(self, image: Path) -> str:
        stat = image.stat()
        key = (str(image.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        with self._cache_lock:
            value = self._images.pop(key, None)
            if value is not None:
                self._images[key] = value
                return value
            value = "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
            limit = 32 * 1024 * 1024
            while self._images and self._image_bytes + len(value) > limit:
                _, old = self._images.popitem(last=False)
                self._image_bytes -= len(old)
            if len(value) <= limit:
                self._images[key] = value
                self._image_bytes += len(value)
            return value

    def caption(self, images: Path | Sequence[Path], *,
                timestamps: Sequence[float] | None = None,
                target_timestamps: Sequence[float] | None = None,
                correction: str | None = None,
                prompt_override: str | None = None) -> str:
        cfg = self.config
        self._local.requests = []
        paths = [images] if isinstance(images, Path) else list(images)
        if not paths:
            raise HarnessError("VLM caption requires at least one image")
        if prompt_override is not None and len(paths) > 100:
            raise HarnessError("Hierarchical VLM requests accept at most 100 frames")
        suffix, extras = _request_extras(cfg["model"])
        prompt = cfg["prompt"] if prompt_override is None else nonempty(prompt_override, "prompt_override")
        placeholder = "{{FRAME_TIMESTAMPS}}"
        if timestamps is not None:
            timeline = list(timestamps)
            if len(timeline) != len(paths):
                raise HarnessError("VLM image and timestamp counts differ")
            if prompt.count(placeholder) != 1:
                raise HarnessError("VLM timestamp prompt must contain exactly one placeholder")
            # A closed set of values, without frame numbers: a filename is
            # 1-based while its timestamp starts at zero, and pairing the two
            # invites off-by-one boundaries the window cannot express.
            prompt = prompt.replace(placeholder, ", ".join(
                f"{timestamp}" for timestamp in timeline
            ))
            if target_timestamps is not None:
                targets = list(target_timestamps)
                if not targets or any(target not in timeline for target in targets):
                    raise HarnessError("VLM target timeline must be a nonempty subset of timestamps")
                prompt += (
                    "\n\nCentered target region: " + ", ".join(str(value) for value in targets) + ". "
                    "Focus the answer on visible states, actions, and transitions in this target region. "
                    "The other frames are temporal context; do not separately caption unrelated "
                    "context-only content. An event may still use any listed window timestamp when "
                    "it genuinely crosses a target boundary. Every event object must include kind, "
                    "set to state, action, or transition. Use state for a stable visible condition, "
                    "action for ongoing motion or interaction, and transition for an entry, exit, "
                    "start, stop, or cut. Use person instead of guessing gender or identity when "
                    "unclear. Ignore timestamps, channel labels, watermarks, and other overlay text "
                    "unless a change in the video feed itself is meaningful. A segment with equal "
                    "start and end is observed at one sampled instant; it is not known to have zero "
                    "real-world duration."
                )
            if self.timestamp_mode == "frame_index":
                prompt += (
                    "\nBoundary coordinate contract: the listed values are zero-based "
                    "frame indices, NOT seconds. Both start and end are inclusive indices."
                )
        elif placeholder in prompt:
            raise HarnessError("VLM timestamp prompt requires frame timestamps")
        elif target_timestamps is not None:
            raise HarnessError("VLM target timestamps require frame timestamps")
        elif len(paths) > 1:
            prompt += (
                "\nThe images are sampled video frames in chronological order. "
                "Describe the visible temporal progression across them."
            )
        if correction is not None:
            # Temperature is 0, so an identical request would return the
            # identical rejected answer; the rejection has to go back in.
            prompt += (
                "\n\nYour previous answer was rejected: "
                f"{nonempty(correction, 'correction')}\n"
                "Answer again and satisfy every requirement above."
            )
        prompt += suffix
        content = [{"type": "text", "text": prompt}]
        for image in paths:
            image_url = self._image_url(image)
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url, "detail": "high"},
            })
        for attempt in range(cfg["max_retries"] + 1):
            payload = {
                "model": cfg["model"], "temperature": cfg["temperature"],
                "max_tokens": cfg["max_tokens"],
                "messages": [{"role": "user", "content": content}],
                **extras,
            }
            request = urllib.request.Request(
                cfg["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(payload, allow_nan=False).encode(),
                headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            )
            started = time.monotonic()
            trace = {"attempt": attempt + 1, "image_count": len(paths)}
            try:
                with urllib.request.urlopen(request, timeout=cfg["timeout_sec"]) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                finish_reason = choice.get("finish_reason")
                text = _choice_text(choice)
                trace.update(finish_reason=finish_reason, raw_response=text,
                             status="success")
                usage = result.get("usage")
                trace["usage"] = {
                    key: value for key, value in (usage.items() if isinstance(usage, dict) else [])
                    if key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    and type(value) is int and value >= 0
                }
                if finish_reason in _REFUSAL_REASONS or finish_reason == "length":
                    raise HarnessError(
                        f"VLM caption was truncated or refused (finish_reason={finish_reason!r})"
                    )
                if finish_reason not in _COMPLETE_REASONS:
                    raise HarnessError(
                        f"VLM caption did not complete (finish_reason={finish_reason!r})"
                    )
                return nonempty(text, "VLM caption")
            except urllib.error.HTTPError as exc:
                trace.update(status="http_error", http_status=exc.code)
                # Do not log response bodies, request headers, or credentials.
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == cfg["max_retries"]:
                    raise HarnessError(f"VLM request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError):
                trace["status"] = "connection_error"
                if attempt == cfg["max_retries"]:
                    raise HarnessError("VLM request failed: connection error or timeout") from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                trace["status"] = "invalid_response"
                raise HarnessError("VLM returned an invalid caption response") from exc
            except HarnessError:
                trace["status"] = "rejected_response"
                raise
            finally:
                trace["elapsed_sec"] = time.monotonic() - started
                self._local.requests.append(trace)
            if attempt == cfg["max_retries"]:
                break
            time.sleep(min(2 ** attempt, 10))
        raise AssertionError("Unreachable")
