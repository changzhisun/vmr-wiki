"""Adapter for the established algorithms during the compatibility window.

Only this compiler-side adapter knows the historical on-disk build format.
It is never imported by Artifact or Query.
"""

from copy import deepcopy
from vmr.core.errors import HarnessError
from vmr.core.validation import positive_int, number
from vmr.compiler.protocol import CompilerConfig, CompileResult


class LegacyBackend:
    def parse_config(self, raw):
        settings = deepcopy(dict(raw))
        settings["caption_mode"] = self.name
        for name in ("image_max_size", "jpeg_quality"):
            positive_int(settings[name], name)
        if not 2 <= settings["jpeg_quality"] <= 31:
            raise HarnessError("jpeg_quality must be in [2, 31]")
        if number(settings["sample_interval_sec"], "sample_interval_sec") <= 0:
            raise HarnessError("sample_interval_sec must be positive")
        return CompilerConfig(settings)

    requires_vlm = True

    def compile(self, context):
        from vmr.compiler.context import prepare_backend_context
        from vmr.artifact.integrity import remove_tree

        prepared = context.build is None
        if prepared:
            context = prepare_backend_context(context, self)
        try:
            return self._compile_prepared(context)
        finally:
            if prepared and context.build["staging"].exists():
                remove_tree(context.build["staging"])

    def _compile_prepared(self, context):
        b = context.build
        metadata = self.build(
            context.source.path,
            b["video_id"],
            context.output,
            b["staging"],
            b["checkpoint"],
            context.captioner,
            b["cfg"],
            b["duration"],
            b["video_stream_duration"],
            context.source.sha256,
            b["content_hash"],
            b["ffmpeg_version"],
            b["check"],
            runtime=context.runtime,
            cancel_event=context.cancel_event,
        )
        if b["cfg"].get("build_record") == "build.json":
            from vmr.core.jsonio import write_json

            metadata = dict(metadata)
            metadata["compile_config"] = metadata.pop("ingest_config")
            metadata["compile_config_hash"] = metadata.pop("ingest_config_hash")
            write_json(context.output / "build.json", metadata)
        text = tuple(p for p in self.text_files if (context.output / p).is_file())
        media = tuple(
            p.relative_to(context.output).as_posix()
            for p in context.output.rglob("*")
            if p.is_file()
            and (
                p.name == "frames.jsonl"
                or p.relative_to(context.output).parts[0] == "frames"
            )
        )
        return CompileResult(
            context.output, metadata["duration"], text, media, metadata
        )
