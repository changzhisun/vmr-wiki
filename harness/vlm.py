"""Fixed, query-independent image captioning via a Chat Completions-compatible API."""
from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
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
    def __init__(self, config: dict):
        self.config = config
        self.key = os.environ.get(config["api_key_env"])
        if not self.key:
            raise HarnessError(f"Set {config['api_key_env']} before ingest")
        if config["model"].startswith("REPLACE_"):
            raise HarnessError("Configure an explicit VLM model before ingest")

    def caption(self, images: Path | Sequence[Path]) -> str:
        cfg = self.config
        paths = [images] if isinstance(images, Path) else list(images)
        if not paths:
            raise HarnessError("VLM caption requires at least one image")
        suffix, extras = _request_extras(cfg["model"])
        prompt = cfg["prompt"]
        if len(paths) > 1:
            prompt += (
                "\nThe images are sampled video frames in chronological order. "
                "Describe the visible temporal progression across them."
            )
        prompt += suffix
        content = [{"type": "text", "text": prompt}]
        for image in paths:
            image_url = (
                "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()
            )
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
            try:
                with urllib.request.urlopen(request, timeout=cfg["timeout_sec"]) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                finish_reason = choice.get("finish_reason")
                text = _choice_text(choice)
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
                # Do not log response bodies, request headers, or credentials.
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == cfg["max_retries"]:
                    raise HarnessError(f"VLM request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == cfg["max_retries"]:
                    raise HarnessError("VLM request failed: connection error or timeout") from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise HarnessError("VLM returned an invalid caption response") from exc
            if attempt == cfg["max_retries"]:
                break
            time.sleep(min(2 ** attempt, 10))
        raise AssertionError("Unreachable")
