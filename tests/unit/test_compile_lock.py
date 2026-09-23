import os
from pathlib import Path
import select
import subprocess
import sys

import pytest

from vmr.core.locking import exclusive_lock
from vmr.core.errors import HarnessError


@pytest.mark.parametrize("contents", ["", "99999999", str(os.getpid())])
def test_unowned_lock_file_can_be_reused(tmp_path, contents):
    lock = tmp_path / ".compile.lock"
    lock.write_text(contents)
    inode = lock.stat().st_ino
    with exclusive_lock(lock):
        with pytest.raises(HarnessError, match="Compile in progress"):
            with exclusive_lock(lock, message="Compile in progress"):
                pytest.fail("two owners acquired one lock")
    with exclusive_lock(lock):
        assert lock.stat().st_ino == inode


def test_process_death_releases_lock_without_replacing_inode(tmp_path):
    lock = tmp_path / ".compile.lock"
    script = """
import sys
from pathlib import Path
from vmr.core.locking import exclusive_lock
with exclusive_lock(Path(sys.argv[1])):
    print('acquired', flush=True)
    sys.stdin.read()
"""
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(lock)],
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert select.select([child.stdout], [], [], 10)[0], (
            "child did not acquire lock"
        )
        assert child.stdout.readline().strip() == "acquired"
        inode = lock.stat().st_ino
        # The file remains empty for its entire lifetime: absence of a PID must
        # never allow a second writer to delete it and claim a replacement.
        assert lock.read_text() == ""
        with pytest.raises(HarnessError):
            with exclusive_lock(lock):
                pytest.fail("child lock was stolen")
        child.kill()
        child.wait(timeout=10)
        with exclusive_lock(lock):
            assert lock.stat().st_ino == inode
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_exception_releases_lock(tmp_path):
    lock = tmp_path / ".compile.lock"
    with pytest.raises(RuntimeError):
        with exclusive_lock(lock):
            raise RuntimeError("failed build")
    with exclusive_lock(lock):
        pass
