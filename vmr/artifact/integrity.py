from pathlib import Path
import shutil
from vmr.core.errors import HarnessError
from vmr.core.hashing import file_hash


def tree_hashes(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise HarnessError(f"Expected a real directory: {root}")
    hashes = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise HarnessError(f"Symlink is forbidden in frozen input: {relative}")
        if path.is_file() and relative not in exclude:
            hashes[relative] = file_hash(path)
        elif not path.is_dir() and not path.is_file():
            raise HarnessError(f"Non-regular input: {relative}")
    return hashes


def make_readonly(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def remove_tree(root: Path) -> None:
    """Remove our own temporary copies, including their read-only directories."""
    if not root.exists():
        return
    root.chmod(0o700)
    for path in root.rglob("*"):
        if not path.is_symlink() and path.is_dir():
            path.chmod(0o700)
    shutil.rmtree(root)
