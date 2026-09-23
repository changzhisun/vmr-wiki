from vmr.compiler.protocol import CompilerConfig
from vmr.compiler.methods.base import LegacyBackend
from vmr.compiler.methods.window_identity import (
    _INGEST_CONTENT_KEYS,
    _ingest_content_value,
    _VLM_CONTENT_KEYS,
)


class DenseCompiler(LegacyBackend):
    name = "dense"
    version = 1
    text_files = ("wiki.md",)

    def content_identity(self, config, source):
        ingest = config.settings
        vlm = ingest["vlm"]
        return {
            **{
                key: _ingest_content_value(ingest, key)
                for key in sorted(_INGEST_CONTENT_KEYS)
            },
            "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
        }

    def build(self, *args, runtime=None, cancel_event=None):
        from vmr.compiler.methods.window import publish_window

        return publish_window(*args, cancel_event=cancel_event)

    def resolve_config(self, section, profiles, media, templates):
        from vmr.compiler.methods.options import window_options

        options = window_options(self, section, profiles, media, templates)
        from vmr.core.errors import HarnessError

        if (
            options["caption_window_frames"] > 1
            and options["caption_stride_frames"] > options["caption_window_frames"] // 2
        ):
            raise HarnessError("Dense stride must not exceed half the window")
        if options["vlm"]["prompt"].count("{{FRAME_TIMESTAMPS}}") != 1:
            raise HarnessError("Dense prompt needs exactly one {{FRAME_TIMESTAMPS}}")
        return self.parse_config(options)
