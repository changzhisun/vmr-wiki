"""Build isolated query workspaces from pinned Wiki artifacts or videos."""

from contextlib import contextmanager
from pathlib import Path
import tempfile
import shutil
from vmr.core.errors import RunFailure, HarnessError
from vmr.core.hashing import file_hash
from vmr.core.jsonio import write_json
from vmr.artifact.integrity import tree_hashes, make_readonly, remove_tree


@contextmanager
def query_workspace(query, task, artifact, templates, runs, *, text_only=False):
    Path(runs).mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=task["query_id"] + "-", dir=runs)).resolve()
    try:
        target = root / "wiki"
        mode = "text" if text_only else "multimodal"
        artifact.copy_public_to(target, input_mode=mode)
        expected = artifact.public_hashes(input_mode=mode)
        make_readonly(target)
        (root / "AGENTS.md").write_text(templates["AGENTS.md"], encoding="utf-8")
        write_json(root / "task.json", task)
        for name in ("AGENTS.md", "task.json"):
            (root / name).chmod(0o444)
        (root / "output").mkdir(mode=0o777)
        (root / "output").chmod(0o777)
        root.chmod(0o555)
        before = {n: file_hash(root / n) for n in ("AGENTS.md", "task.json")}
        yield root
        try:
            valid = (
                tree_hashes(target) == expected
                and all(file_hash(root / n) == h for n, h in before.items())
                and set(p.name for p in root.iterdir())
                == {"wiki", "AGENTS.md", "task.json", "output"}
            )
        except (HarnessError, OSError) as exc:
            raise RunFailure("Agent changed an immutable input", "tampered") from exc
        if not valid:
            raise RunFailure("Agent changed an immutable input", "tampered")
        artifact.verify()
    finally:
        remove_tree(root)


@contextmanager
def video_query_workspace(task, templates, source):
    """Stage the pinned video under an anonymous path for one read-only bind."""
    # Docker Desktop shares /tmp (resolved to /private/tmp on macOS), while
    # Python's default TMPDIR may be under /private/var/folders.
    private = Path(
        tempfile.mkdtemp(prefix="vmr-video-private-", dir=Path("/tmp").resolve())
    ).resolve()
    try:
        root = Path(tempfile.mkdtemp(prefix="vmr-video-", dir=private)).resolve()
        try:
            (root / "AGENTS.md").write_text(templates["AGENTS.md"], encoding="utf-8")
            write_json(root / "task.json", task)
            source.verify()
            shutil.copyfile(source.path, root / "video.mp4")
            if file_hash(root / "video.mp4") != source.digest:
                raise HarnessError("Staged video does not match the pinned source")
            immutable = ("AGENTS.md", "task.json", "video.mp4")
            for name in immutable:
                (root / name).chmod(0o444)
            (root / "output").mkdir(mode=0o777)
            (root / "output").chmod(0o777)
            root.chmod(0o555)
            before = {n: file_hash(root / n) for n in immutable}
            yield root
            try:
                valid = all(
                    not (root / n).is_symlink()
                    and (root / n).is_file()
                    and file_hash(root / n) == h
                    for n, h in before.items()
                ) and {p.name for p in root.iterdir()} == {*immutable, "output"}
            except (HarnessError, OSError) as exc:
                raise RunFailure(
                    "Agent changed an immutable input", "tampered"
                ) from exc
            if not valid:
                raise RunFailure("Agent changed an immutable input", "tampered")
            source.verify()
        finally:
            remove_tree(root)
    finally:
        remove_tree(private)
