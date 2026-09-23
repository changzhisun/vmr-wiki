from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class CompilerConfig:
    """Resolved backend options; owned and validated by the selected compiler."""

    settings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class VideoSource:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class CompileContext:
    source: VideoSource
    config: CompilerConfig
    output: Path
    templates: Path
    captioner: Any = None
    runtime: Any = None
    cancel_event: Any = None
    build: Any = None


@dataclass(frozen=True, slots=True)
class CompileResult:
    root: Path
    duration_sec: float
    text_files: tuple[str, ...]
    multimodal_files: tuple[str, ...]
    provenance: Mapping[str, Any]


class WikiCompiler(Protocol):
    name: str
    version: int

    def parse_config(self, raw: Mapping[str, Any]) -> CompilerConfig: ...
    def content_identity(
        self, config: CompilerConfig, source: VideoSource | None
    ) -> Mapping[str, Any]: ...
    def compile(self, context: CompileContext) -> CompileResult: ...
