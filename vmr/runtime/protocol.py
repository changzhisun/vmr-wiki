from pathlib import Path
from typing import Protocol
from .types import AgentResult, AgentCancelled


class AgentRuntime(Protocol):
    provenance: dict

    def run(
        self,
        workspace: Path,
        prompt: str,
        stdout: Path,
        stderr: Path,
        *,
        cancel_event=None,
    ) -> AgentResult: ...
