from __future__ import annotations

AGENT_FAILURE_KINDS = frozenset(
    {"timeout", "agent_error", "invalid_output", "tampered"}
)
HARNESS_FAILURE_KINDS = frozenset({"harness_error", "interrupted"})
FAILURE_KINDS = AGENT_FAILURE_KINDS | HARNESS_FAILURE_KINDS


class HarnessError(ValueError):
    """An input or experiment invariant was violated."""


class RunFailure(HarnessError):
    """A run failure caused by the agent under test rather than the harness."""

    def __init__(self, message: str, kind: str):
        if kind not in AGENT_FAILURE_KINDS:
            raise HarnessError(f"Not an agent failure kind: {kind!r}")
        super().__init__(message)
        self.kind = kind
