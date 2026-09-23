from vmr.compiler.protocol import CompilerConfig
from vmr.compiler.methods.base import LegacyBackend

_VLM_CONTENT_KEYS = ("model", "prompt", "temperature", "max_tokens")


class HierarchicalCompiler(LegacyBackend):
    name = "hierarchical"
    version = 1
    text_files = ("wiki.md", "nodes.jsonl", "observations.jsonl")

    def content_identity(self, config, source):
        ingest = config.settings
        vlm = ingest["vlm"]
        from vmr.compiler.methods.hierarchical.config import HIERARCHY_VERSION, settings

        return {
            "caption_mode": "hierarchical",
            "hierarchy_processing_version": ingest.get(
                "hierarchy_processing_version", HIERARCHY_VERSION
            ),
            "hierarchy": settings(ingest),
            **{
                key: ingest[key]
                for key in ("image_max_size", "jpeg_quality", "caption_max_repairs")
            },
            "vlm": {key: vlm[key] for key in sorted(_VLM_CONTENT_KEYS)},
        }

    def build(self, *args, runtime=None, cancel_event=None):
        from vmr.compiler.methods.hierarchical.build import publish_hierarchy

        return publish_hierarchy(*args)

    def resolve_config(self, section, profiles, media, templates):
        from vmr.compiler.methods.options import vlm_options
        from vmr.compiler.methods.hierarchical.config import settings, HIERARCHY_VERSION
        from vmr.core.errors import HarnessError

        options = vlm_options(self, section, profiles, media)
        if "{{FRAME_TIMESTAMPS}}" in options["vlm"]["prompt"]:
            raise HarnessError("This compiler constructs its own timeline")
        options["hierarchy"] = section.get("method_config", {})
        options["hierarchy"] = settings(options)
        options["hierarchy_processing_version"] = HIERARCHY_VERSION
        return self.parse_config(options)
