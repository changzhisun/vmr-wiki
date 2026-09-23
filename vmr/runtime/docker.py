from agents.runner import DockerRunner
from .types import AgentResult, AgentCancelled
from .trace import trace_path


def query_runtime(config):
    return DockerRunner({"query": config.runtime_config()})
