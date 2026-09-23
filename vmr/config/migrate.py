"""Explicit compiler-side migration of v1/v2 config into typed domains."""

from .models import AppConfig, CompileConfig, MediaConfig, DatasetConfig, StorageConfig
from .query import from_legacy_query
from pathlib import Path


def from_legacy(cfg):
    from vmr.compiler.registry import get_compiler

    options = cfg["ingest"]
    compiler = get_compiler(options["caption_mode"])
    media = MediaConfig(
        **{key: options[key] for key in MediaConfig.__dataclass_fields__}
    )
    paths = cfg["paths"]
    root = Path(paths["wiki"]).parent
    return AppConfig(
        dataset=DatasetConfig(**cfg["dataset"]),
        storage=StorageConfig(
            root=root,
            artifacts=Path(paths.get("artifacts", root / "artifacts")),
            runs=Path(paths["runs"]),
            results=Path(paths["results"]),
            templates=Path(paths["templates"]),
            datasets=Path(paths["datasets"]),
        ),
        compile=CompileConfig(
            compiler.name,
            media,
            compiler.parse_config(options),
            options.get("consecutive_failure_limit", 3),
        ),
        query=from_legacy_query(cfg["query"], cfg.get("query_templates"))
        if "query" in cfg
        else None,
    )


def load_compile_config(path):
    from .loader import _load_extended
    from .compile import load_modern_compile

    raw = _load_extended(Path(path).resolve())
    version = raw.get("version", 1)
    if type(version) is not int or version not in (1, 2, 3):
        from vmr.core.errors import HarnessError

        raise HarnessError("Unsupported configuration version")
    if version == 3:
        return load_modern_compile(raw, Path(path).resolve())
    from vmr.compat.config import load_config

    return from_legacy(load_config(path))
