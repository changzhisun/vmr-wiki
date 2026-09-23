from vmr.compiler.protocol import CompilerConfig
from vmr.compiler.methods.base import LegacyBackend

_VLM_CONTENT_KEYS = ("model", "prompt", "temperature", "max_tokens")


class BidirectionalCompiler(LegacyBackend):
    name = "bidirectional"
    version = 1
    text_files = (
        "wiki.md",
        "nodes.jsonl",
        "observations.jsonl",
        "bottomup_observations.jsonl",
        "coverage.jsonl",
    )

    def content_identity(self, config, source):
        ingest = config.settings
        vlm = ingest["vlm"]
        from vmr.compiler.methods.bidirectional.config import (
            PIPELINE_VERSION,
            content_settings,
        )

        return {
            "caption_mode": "bidirectional",
            "pipeline_version": ingest.get("pipeline_version", PIPELINE_VERSION),
            "bidirectional": content_settings(ingest),
            **{
                key: ingest[key]
                for key in ("image_max_size", "jpeg_quality", "caption_max_repairs")
            },
            "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
        }

    def build(self, *args, runtime=None, cancel_event=None):
        from vmr.compiler.methods.bidirectional.build import publish_bidirectional

        return publish_bidirectional(*args)

    def resolve_config(self, section, profiles, media, templates):
        from vmr.compiler.methods.options import vlm_options
        from vmr.compiler.methods.bidirectional.config import settings, PIPELINE_VERSION
        from vmr.core.errors import HarnessError

        options = vlm_options(self, section, profiles, media)
        if "{{FRAME_TIMESTAMPS}}" in options["vlm"]["prompt"]:
            raise HarnessError("This compiler constructs its own timeline")
        options["bidirectional"] = section.get("method_config", {})
        options["bidirectional"] = settings(options)
        options["pipeline_version"] = PIPELINE_VERSION
        return self.parse_config(options)
