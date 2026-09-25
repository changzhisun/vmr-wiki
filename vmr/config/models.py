from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any
from vmr.core.errors import HarnessError
from vmr.core.validation import positive_int, number, relative_template
from .resolve import agent_kind, agent_settings


@dataclass(frozen=True, slots=True)
class MediaConfig:
    sample_interval_sec: float = 1.0
    image_max_size: int = 768
    jpeg_quality: int = 2


@dataclass(frozen=True, slots=True)
class CompileConfig:
    compiler: str
    media: MediaConfig
    method: Any
    consecutive_failure_limit: int = 3

    def __post_init__(self):
        positive_int(self.consecutive_failure_limit, "consecutive_failure_limit")


@dataclass(frozen=True, slots=True)
class QueryConfig:
    agent: str
    model: str
    container_image: str
    api_key_env: str
    egress_allowed_hosts: tuple[str, ...]
    base_url: str | None = None
    input_mode: str = "multimodal"
    type: str = "video-wiki"
    max_predictions: int = 5
    timeout_sec: float = 600
    agents_template: str | None = None
    prompt_template: str | None = None
    text_agents_template: str = "query_agents.video_wiki.text_only.md"
    text_prompt_template: str = "query_prompt.video_wiki.text_only.md"

    def __post_init__(self):
        if self.input_mode not in ("text", "multimodal"):
            raise HarnessError("query.input_mode must be text or multimodal")
        if self.type not in ("video-wiki", "video-only"):
            raise HarnessError("query.type must be video-wiki or video-only")
        if self.type == "video-only" and self.input_mode != "multimodal":
            raise HarnessError("query.type video-only requires input_mode multimodal")
        if self.agents_template is None:
            object.__setattr__(
                self,
                "agents_template",
                "query_agents.video_only.md"
                if self.type == "video-only"
                else "query_agents.video_wiki.md",
            )
        if self.prompt_template is None:
            object.__setattr__(
                self,
                "prompt_template",
                "query_prompt.video_only.md"
                if self.type == "video-only"
                else "query_prompt.video_wiki.md",
            )
        positive_int(self.max_predictions, "max_predictions")
        if number(self.timeout_sec, "timeout_sec") <= 0:
            raise HarnessError("timeout_sec must be positive")
        agent_kind(self.agent, "agent")
        relative_template(self.agents_template, "query.templates.agents")
        relative_template(self.prompt_template, "query.templates.prompt")
        relative_template(self.text_agents_template, "query.templates.text.agents")
        relative_template(self.text_prompt_template, "query.templates.text.prompt")
        agent_settings(
            {
                "model": self.model,
                "container_image": self.container_image,
                "api_key_env": self.api_key_env,
                "egress_allowed_hosts": list(self.egress_allowed_hosts),
                "base_url": self.base_url,
            },
            field="query",
        )

    def runtime_config(self):
        return dict(
            agent=self.agent,
            model=self.model,
            container_image=self.container_image,
            api_key_env={self.agent: self.api_key_env},
            base_url={self.agent: self.base_url},
            egress_allowed_hosts={self.agent: list(self.egress_allowed_hosts)},
            timeout_sec=self.timeout_sec,
            max_predictions=self.max_predictions,
            text_only=self.input_mode == "text",
        )

    def snapshot(self):
        result = asdict(self)
        result["egress_allowed_hosts"] = list(self.egress_allowed_hosts)
        return result


@dataclass(frozen=True, slots=True)
class StorageConfig:
    root: Path
    artifacts: Path
    runs: Path
    results: Path
    templates: Path
    datasets: Path


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    name: str | None = None
    split: str | None = None


@dataclass(frozen=True, slots=True)
class AppConfig:
    storage: StorageConfig
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    compile: CompileConfig | None = None
    query: QueryConfig | None = None
