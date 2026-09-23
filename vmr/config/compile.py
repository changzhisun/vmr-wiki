"""Compile-only v3 schema. Backend config resolution is registry-owned."""

from .models import AppConfig, CompileConfig, MediaConfig, StorageConfig, DatasetConfig
from .resolve import _known


def load_modern_compile(raw, path):
    from vmr.compiler.registry import get_compiler

    _known(
        raw,
        {
            "version",
            "dataset",
            "storage",
            "profiles",
            "compile",
            "query",
            "batch",
            "evaluation",
        },
        "top-level",
    )
    section = raw["compile"]
    _known(
        section,
        {
            "compiler",
            "media",
            "method_config",
            "captioner_profile",
            "kind",
            "repair_attempts",
        },
        "compile",
    )
    compiler = get_compiler(section["compiler"])
    media = MediaConfig(**section.get("media", {}))
    storage = raw.get("storage", {})
    root = (path.parent / storage.get("root", ".")).resolve()
    resolved = StorageConfig(
        root=root,
        **{
            key: (root / storage.get(key, key)).resolve()
            for key in ("artifacts", "runs", "results", "templates", "datasets")
        },
    )
    method = compiler.resolve_config(
        section, raw.get("profiles", {}), media, resolved.templates
    )
    return AppConfig(
        storage=resolved,
        dataset=DatasetConfig(**raw.get("dataset", {})),
        compile=CompileConfig(
            compiler.name,
            media,
            method,
            raw.get("batch", {}).get("consecutive_failure_limit", 3),
        ),
    )
