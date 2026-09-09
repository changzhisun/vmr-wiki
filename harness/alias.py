"""Per-experiment opaque identifiers for the agent's workspace.

Public VMR benchmarks are in the training data of the models under test, and the
official query id, video id, and split name are exact lookup keys into them. A
read-only mount and an internal network stop the agent from *reading* ground
truth; neither stops it from recalling a label it recognizes, or from asking the
model API about it. So the workspace never carries the real identifiers.

Aliases are keyed by a secret minted once per experiment, so they are neither
guessable from public inputs nor correlatable across experiments. The mapping
stays on the host: predictions are validated against the aliases the agent was
given and rewritten to real identifiers before they are saved.

This narrows the channel but does not close it. The query text itself is the
task input and cannot be obscured, and for a public benchmark that text is also
searchable. Treat aliasing as removing the precise lookup key, not as a
guarantee of decontamination.
"""
from __future__ import annotations

import hmac
import secrets
from hashlib import sha256

from harness.common import HarnessError, nonempty

_PREFIX = {"query": "q", "video": "v", "split": "s"}


def new_secret() -> str:
    return secrets.token_hex(32)


def alias(secret: str, kind: str, value: str) -> str:
    try:
        key = bytes.fromhex(nonempty(secret, "alias_secret"))
    except ValueError as exc:
        raise HarnessError("alias_secret must be a hex string") from exc
    if len(key) < 16:
        raise HarnessError("alias_secret must be at least 16 bytes")
    digest = hmac.new(key, f"{kind}:{value}".encode(), sha256).hexdigest()
    return f"{_PREFIX[kind]}{digest[:16]}"


class Aliases:
    """Two-way map between real identifiers and one experiment's opaque tokens."""

    def __init__(self, secret: str, split: str, queries: list[dict]):
        self.split = alias(secret, "split", nonempty(split, "split"))
        self.query = {q["query_id"]: alias(secret, "query", q["query_id"]) for q in queries}
        self.video = {q["video_id"]: alias(secret, "video", q["video_id"]) for q in queries}
        for kind, mapping in (("query", self.query), ("video", self.video)):
            if len(set(mapping.values())) != len(mapping):
                raise HarnessError(f"Alias collision for {kind} identifiers")

    def task(self, query: dict, max_predictions: int) -> dict:
        """The only description of the task the agent ever sees."""
        return {"query_id": self.query[query["query_id"]],
                "video_id": self.video[query["video_id"]],
                "split": self.split,
                "query": query["query"],
                "max_predictions": max_predictions}
