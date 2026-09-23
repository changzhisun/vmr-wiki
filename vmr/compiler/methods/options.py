from dataclasses import asdict
from copy import deepcopy
from vmr.config.resolve import _known, _select_profile, _vlm_profiles
from vmr.core.errors import HarnessError
from vmr.core.validation import positive_int


def vlm_options(compiler, section, profiles, media):
    if "agent_profile" in section or "kind" in section:
        raise HarnessError("kind is not valid for this compiler")
    options = dict(
        asdict(media),
        caption_mode=compiler.name,
        caption_max_repairs=section.get("repair_attempts", 2),
        caption_processing_version=4,
        caption_window_frames=1,
        caption_stride_frames=1,
        dense_timestamp_mode="absolute_seconds",
    )
    if (
        type(options["caption_max_repairs"]) is not int
        or options["caption_max_repairs"] < 0
    ):
        raise HarnessError("repair_attempts must be a nonnegative integer")
    options["vlm"] = _select_profile(
        _vlm_profiles(profiles.get("vlms", {})),
        section.get("captioner_profile") or "qwen_default",
        "compile.captioner_profile",
    )
    return options


def window_options(compiler, section, profiles, media, templates):
    options = vlm_options(compiler, section, profiles, media)
    raw = section.get("method_config", {})
    _known(
        raw,
        {"window_frames", "stride_frames", "timestamp_mode"},
        "compile.method_config",
    )
    options.update(
        caption_window_frames=raw.get("window_frames", 1),
        caption_stride_frames=raw.get("stride_frames", 1),
        dense_timestamp_mode=raw.get("timestamp_mode", "absolute_seconds"),
    )
    for key in ("caption_window_frames", "caption_stride_frames"):
        positive_int(options[key], key)
    if options["dense_timestamp_mode"] not in ("absolute_seconds", "frame_index"):
        raise HarnessError("Unsupported timestamp_mode")
    return options
