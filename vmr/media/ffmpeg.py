from __future__ import annotations
import subprocess
from vmr.core.errors import HarnessError


def media_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=True
        )
    except subprocess.CalledProcessError as exc:
        raise HarnessError(f"{command[0]} failed: {exc.stderr[-2000:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise HarnessError(f"{command[0]} timed out") from exc
    return result.stdout
