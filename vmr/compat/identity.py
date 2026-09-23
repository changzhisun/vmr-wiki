from vmr.core.hashing import object_hash

# Ingest settings that actually determine caption content. Transport, auth, and
# retry settings are deliberately excluded: changing the provider, endpoint,
# API key environment variable, request timeout, or retry count must not
# invalidate an existing ingest.
_INGEST_CONTENT_KEYS = frozenset(
    {
        "sample_interval_sec",
        "caption_mode",
        "caption_window_frames",
        "caption_stride_frames",
        "caption_max_repairs",
        "image_max_size",
        "jpeg_quality",
        "dense_timestamp_mode",
        "caption_processing_version",
    }
)
_INGEST_CONTENT_DEFAULTS = {
    "caption_mode": "simple",
    "caption_window_frames": 1,
    "caption_stride_frames": 1,
    # A repair changes which caption is stored, so the budget is content.
    "caption_max_repairs": 2,
    "dense_timestamp_mode": "legacy_auto",
    "caption_processing_version": 1,
}
_VLM_CONTENT_KEYS = frozenset({"model", "prompt", "temperature", "max_tokens"})


def _ingest_content_value(ingest: dict, key: str):
    if (
        key == "dense_timestamp_mode"
        and ingest.get("caption_mode", "simple") == "simple"
    ):
        return None
    if key in _INGEST_CONTENT_DEFAULTS:
        return ingest.get(key, _INGEST_CONTENT_DEFAULTS[key])
    return ingest[key]


def ingest_content_diff(stored: dict, cfg: dict) -> list[str]:
    """Human-readable content-setting differences between a wiki and current config."""
    if compile_content_hash({"ingest": stored}) == compile_content_hash(cfg):
        return []
    diffs = []
    if "agentic" in (stored.get("caption_mode"), cfg["ingest"].get("caption_mode")):
        if stored.get("caption_mode") != cfg["ingest"].get("caption_mode"):
            return [
                f"caption_mode {stored.get('caption_mode')!r} vs "
                f"{cfg['ingest'].get('caption_mode')!r}"
            ]
        from vmr.compiler.methods.agentic.config import (
            AGENTIC_VERSION,
            content_settings,
        )

        left, right = content_settings(stored), content_settings(cfg["ingest"])
        for key in sorted(left.keys() | right.keys()):
            if left.get(key) != right.get(key):
                diffs.append(
                    "agentic instruction templates differ"
                    if key == "templates_hash"
                    else f"agentic.{key} {left.get(key)!r} vs {right.get(key)!r}"
                )
        if stored.get("agentic_version", AGENTIC_VERSION) != cfg["ingest"].get(
            "agentic_version", AGENTIC_VERSION
        ):
            diffs.append("agentic_version differs")
        for key in ("image_max_size", "jpeg_quality"):
            if stored.get(key) != cfg["ingest"].get(key):
                diffs.append(f"{key} {stored.get(key)!r} vs {cfg['ingest'].get(key)!r}")
        return diffs or ["ingest content hash"]
    if (
        stored.get("caption_mode")
        == cfg["ingest"].get("caption_mode")
        == "bidirectional"
    ):
        from vmr.compiler.methods.bidirectional.config import (
            PIPELINE_VERSION,
            content_settings,
        )

        if content_settings(stored) != content_settings(cfg["ingest"]):
            diffs.append("bidirectional settings differ")
        if stored.get("pipeline_version", PIPELINE_VERSION) != cfg["ingest"].get(
            "pipeline_version", PIPELINE_VERSION
        ):
            diffs.append("pipeline_version differs")
    if (
        stored.get("caption_mode")
        == cfg["ingest"].get("caption_mode")
        == "hierarchical"
    ):
        from vmr.compiler.methods.hierarchical.config import HIERARCHY_VERSION, settings

        if settings(stored) != settings(cfg["ingest"]):
            diffs.append("hierarchy settings differ")
        if stored.get("hierarchy_processing_version", HIERARCHY_VERSION) != cfg[
            "ingest"
        ].get("hierarchy_processing_version", HIERARCHY_VERSION):
            diffs.append("hierarchy_processing_version differs")
    for key in sorted(_INGEST_CONTENT_KEYS):
        left = _ingest_content_value(stored, key)
        right = _ingest_content_value(cfg["ingest"], key)
        if left != right:
            diffs.append(f"{key} {left!r} vs {right!r}")
    stored_vlm = stored.get("vlm") or {}
    current_vlm = cfg["ingest"]["vlm"]
    for key in sorted(_VLM_CONTENT_KEYS):
        left, right = stored_vlm.get(key), current_vlm.get(key)
        if left != right:
            diffs.append(
                f"vlm.{key} {left!r} vs {right!r}"
                if key != "prompt"
                else "vlm.prompt differs"
            )
    return diffs or ["ingest content hash"]


def compile_content_hash(cfg):
    from vmr.compiler.registry import get_compiler

    compiler = get_compiler(cfg["ingest"].get("caption_mode", "simple"))
    return object_hash(
        compiler.content_identity(compiler.parse_config(cfg["ingest"]), None)
    )
