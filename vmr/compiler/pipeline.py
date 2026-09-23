"""Compile transaction: reusable private build -> validated sealed publication."""

from pathlib import Path
import shutil
import subprocess
import logging
from vmr.core.errors import HarnessError
from vmr.core.locking import exclusive_lock
from vmr.core.hashing import file_hash, object_hash
from vmr.core.jsonio import read_json, write_json
from vmr.artifact.store import ArtifactStore
from vmr.artifact.integrity import remove_tree, tree_hashes
from .protocol import VideoSource, CompileContext, CompileResult
from .registry import get_compiler


def compile_video(
    video,
    config,
    store,
    *,
    templates=Path("templates"),
    captioner=None,
    runtime=None,
    cancel_event=None,
):
    compiler = get_compiler(config.compiler)
    settings = compiler.parse_config(config.method.settings)
    source = VideoSource(Path(video).resolve(), file_hash(video))
    content_hash = object_hash(compiler.content_identity(settings, source))
    build_key = object_hash(
        dict(
            source=source.sha256,
            compiler=compiler.name,
            version=compiler.version,
            config=content_hash,
        )
    )
    store = ArtifactStore(store)
    index = store.root / ".builds" / (build_key + ".json")

    def cached_artifact():
        record = read_json(index)
        artifact = store.open(
            record["artifact_id"], manifest_hash=record.get("manifest_hash")
        )
        if artifact.manifest.data["source"][
            "video_sha256"
        ] != source.sha256 or artifact.manifest.data["compiler"] != dict(
            name=compiler.name,
            version=compiler.version,
            content_config_hash="sha256:" + content_hash,
        ):
            raise HarnessError("Build index identity mismatch")
        return artifact

    if index.exists():
        return cached_artifact()
    build = store.root / ".work" / build_key / "output"
    build.parent.mkdir(parents=True, exist_ok=True)
    lock = build.parent / ".compile.lock"
    with exclusive_lock(lock, message="Compile in progress"):
        if index.exists():
            return cached_artifact()
        completed = build.parent / "result.json"
        if completed.exists():
            saved = read_json(completed)
            if saved.get("schema_version") != 1 or tree_hashes(build) != saved.get(
                "content_hashes"
            ):
                raise HarnessError("Completed private build changed before publication")
            result = CompileResult(
                build,
                saved["duration_sec"],
                tuple(saved["text_files"]),
                tuple(saved["multimodal_files"]),
                saved["provenance"],
            )
        else:
            if build.exists():
                remove_tree(build)
            result = compiler.compile(
                CompileContext(
                    source,
                    settings,
                    build,
                    Path(templates),
                    captioner,
                    runtime,
                    cancel_event,
                )
            )
        if result.root.resolve() != build.resolve() or result.root.is_symlink():
            raise HarnessError("Compiler returned a different build root")
        if file_hash(source.path) != source.sha256:
            raise HarnessError("Source video changed during compile")
        if not completed.exists():
            write_json(
                completed,
                dict(
                    schema_version=1,
                    duration_sec=result.duration_sec,
                    text_files=list(result.text_files),
                    multimodal_files=list(result.multimodal_files),
                    provenance=dict(result.provenance),
                    content_hashes=tree_hashes(build),
                ),
            )
        artifact = _publish_result(store, compiler, source, content_hash, result)
        write_json(
            index,
            {
                "artifact_id": artifact.artifact_id(),
                "manifest_hash": artifact.manifest_hash(),
            },
        )
        try:
            remove_tree(build)
            completed.unlink()
        except OSError:
            logging.getLogger(__name__).warning(
                "Artifact published; private build cleanup can be retried: %s", build
            )
        return artifact


def _publish_result(store, compiler, source, content_hash, result):
    build = result.root
    stage = store.staging()
    try:
        public = set(result.text_files) | set(result.multimodal_files)
        actual = tree_hashes(build)
        if not public <= actual.keys():
            raise HarnessError("Compiler declared missing public files")
        for path in build.rglob("*"):
            if not path.is_file():
                continue
            name = path.relative_to(build).as_posix()
            target = "public/" + name if name in public else "internal/" + name
            destination = stage / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        if tree_hashes(build) != actual:
            raise HarnessError("Build changed while publishing artifact")
        expected = {
            ("public/" if name in public else "internal/") + name: digest
            for name, digest in actual.items()
        }
        if tree_hashes(stage) != expected:
            raise HarnessError("Staged artifact does not match completed build")
        text = list(result.text_files)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
        artifact = store.publish(
            stage,
            source=dict(video_sha256=source.sha256, duration_sec=result.duration_sec),
            compiler=dict(
                name=compiler.name,
                version=compiler.version,
                content_config_hash="sha256:" + content_hash,
            ),
            text_files=text,
            multimodal_files=list(result.multimodal_files),
            provenance=dict(
                created_at=result.provenance.get("created_at"),
                source_commit=commit.stdout.strip() or None,
                source_hash=object_hash(
                    {
                        str(
                            p.relative_to(Path(__file__).resolve().parents[2])
                        ): file_hash(p)
                        for folder in ("compiler", "core", "media", "vlm")
                        for p in (Path(__file__).resolve().parents[1] / folder).rglob(
                            "*.py"
                        )
                    }
                ),
                build=dict(result.provenance),
            ),
        )
        return artifact
    finally:
        if stage.exists():
            remove_tree(stage)
