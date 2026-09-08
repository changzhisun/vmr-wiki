"""Fixed, query-independent image captioning via a Chat Completions-compatible API."""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness.common import HarnessError, nonempty


class VLMClient:
    def __init__(self, config: dict):
        self.config = config
        self.key = os.environ.get(config["api_key_env"])
        if not self.key:
            raise HarnessError(f"Set {config['api_key_env']} before ingest")
        if config["model"].startswith("REPLACE_"):
            raise HarnessError("Configure an explicit VLM model before ingest")

    def caption(self, image: Path) -> str:
        cfg = self.config
        payload = {
            "model": cfg["model"], "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": cfg["prompt"]},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode(),
                    "detail": "high",
                }},
            ]}],
        }
        request = urllib.request.Request(
            cfg["base_url"].rstrip("/") + "/chat/completions",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
        )
        for attempt in range(cfg["max_retries"] + 1):
            try:
                with urllib.request.urlopen(request, timeout=cfg["timeout_sec"]) as response:
                    result = json.load(response)
                choice = result["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise HarnessError("VLM caption was truncated or refused")
                return nonempty(choice["message"]["content"], "VLM caption").strip()
            except urllib.error.HTTPError as exc:
                # Do not log response bodies, request headers, or credentials.
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == cfg["max_retries"]:
                    raise HarnessError(f"VLM request failed with HTTP {exc.code}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt == cfg["max_retries"]:
                    raise HarnessError("VLM request failed: connection error or timeout") from None
            except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
                raise HarnessError("VLM returned an invalid caption response") from exc
            time.sleep(min(2 ** attempt, 10))
        raise AssertionError("Unreachable")

