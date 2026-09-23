from vmr.compiler.protocol import CompilerConfig
from vmr.compiler.methods.base import LegacyBackend


class AgenticCompiler(LegacyBackend):
    name = "agentic"
    version = 1
    text_files = ("wiki.md",)

    def content_identity(self, config, source):
        ingest = config.settings
        from vmr.compiler.methods.agentic.config import (
            AGENTIC_VERSION,
            content_settings,
        )

        return {
            "caption_mode": "agentic",
            "agentic_version": ingest.get("agentic_version", AGENTIC_VERSION),
            "agentic": content_settings(ingest),
            **{key: ingest[key] for key in ("image_max_size", "jpeg_quality")},
        }

    def build(self, *args, runtime=None, cancel_event=None):
        from vmr.compiler.methods.agentic.build import publish_agentic

        return publish_agentic(*args, runner=runtime, cancel_event=cancel_event)

    requires_vlm = False

    def resolve_config(self, section, profiles, media, templates):
        from dataclasses import asdict
        from vmr.config.resolve import agent_kind, agent_settings
        from vmr.compiler.methods.agentic.config import settings, AGENTIC_VERSION
        from vmr.core.errors import HarnessError

        if "captioner_profile" in section:
            raise HarnessError("Agentic compilation does not use captioner_profile")
        kind = agent_kind(section.get("kind"), "compile.kind")
        profile = agent_settings(profiles.get("agent"))
        raw = dict(section.get("method_config", {}))
        if set(raw) & {
            "agent",
            "model",
            "container_image",
            "api_key_env",
            "base_url",
            "egress_allowed_hosts",
        }:
            raise HarnessError("method_config must not repeat agent profile settings")
        raw.update(
            agent=kind,
            model=profile["model"],
            container_image=profile["container_image"],
            api_key_env={kind: profile["api_key_env"]},
            base_url={kind: profile["base_url"]},
            egress_allowed_hosts={kind: profile["egress_allowed_hosts"]},
        )
        options = dict(
            asdict(media),
            caption_mode=self.name,
            agentic=raw,
            agentic_version=AGENTIC_VERSION,
        )
        options["agentic"] = settings(options, templates)
        return self.parse_config(options)
