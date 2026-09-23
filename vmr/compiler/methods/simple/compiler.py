from vmr.compiler.protocol import CompilerConfig
from vmr.compiler.methods.base import LegacyBackend
from vmr.compiler.methods.window_identity import (
    _INGEST_CONTENT_KEYS,
    _ingest_content_value,
    _VLM_CONTENT_KEYS,
)


class SimpleCompiler(LegacyBackend):
    name = "simple"
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

        if "{{FRAME_TIMESTAMPS}}" in options["vlm"]["prompt"]:
            raise HarnessError(
                "Simple prompt must not contain a frame timeline placeholder"
            )
        return self.parse_config(options)
