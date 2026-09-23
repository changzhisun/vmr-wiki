"""Process-lifetime locks for the supported POSIX hosts (Linux and macOS)."""

from contextlib import contextmanager
import fcntl
from pathlib import Path

from .errors import HarnessError


@contextmanager
def exclusive_lock(path: Path, *, blocking=False, message="Operation in progress"):
    """Hold a kernel lock; keep its inode so contenders cannot lock replacements.

    Empty files and stale PID text are harmless. Closing the descriptor, including
    on process exit, releases ownership without a check/delete recovery race.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        try:
            fcntl.flock(stream.fileno(), flags)
        except BlockingIOError as exc:
            raise HarnessError(message) from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
