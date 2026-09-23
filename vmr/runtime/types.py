from dataclasses import dataclass
from vmr.core.errors import HarnessError


@dataclass
class AgentResult:
    exit_code: int | None
    timed_out: bool = False


class FatalAgentError(HarnessError):
    """Shared credential, image or endpoint failure, not one video's fault."""


class AgentCancelled(HarnessError):
    """A running agent was stopped cooperatively after batch cancellation."""
