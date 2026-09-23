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
