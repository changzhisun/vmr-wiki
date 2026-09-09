"""Thread-safe terminal progress with throughput and ETA."""
from __future__ import annotations

import math
import sys
import threading
import time
from collections.abc import Callable


def _duration(seconds: float) -> str:
    total = int(math.ceil(max(seconds, 0.0)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _rate(speed: float | None, unit: str) -> str:
    if speed is None:
        return f"-- {unit}/s"
    if speed >= 0.1:
        return f"{speed:.2f} {unit}/s"
    per_minute = speed * 60
    if per_minute >= 0.1:
        return f"{per_minute:.2f} {unit}/min"
    return f"{speed * 3600:.2f} {unit}/h"


class ProgressBar:
    """Render progress, average throughput, and ETA without dependencies.

    TTY output updates in place. Non-TTY output stays quiet until ``finish``
    so redirected logs are not flooded with one line per update.

    ``speed`` and ETA are averages over the whole run so far, so work whose
    per-item cost varies widely (cached versus freshly captioned videos)
    tracks the average rather than the current pace.
    """

    def __init__(
        self,
        total: int,
        desc: str = "Processing",
        *,
        unit: str = "items",
        log_stream: str = "stderr",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if total < 0:
            raise ValueError("progress total must be nonnegative")
        if log_stream not in ("stdout", "stderr"):
            raise ValueError("log_stream must be 'stdout' or 'stderr'")
        self.total = total
        self.desc = desc
        self.unit = unit
        self.current = 0
        self._log_stream = log_stream
        self._clock = clock
        self._started_at = clock()
        # Reentrant: an interrupt handler runs on the thread it interrupts and
        # may call finish() while that thread already holds the lock.
        self._lock = threading.RLock()
        self._width = 30
        self._finished = False
        self._is_tty = sys.stdout.isatty()
        self._rendered_width = 0

    def start(self) -> None:
        """Show the initial state immediately when attached to a terminal."""
        with self._lock:
            if self._is_tty and not self._finished:
                self._render_unlocked()

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.current += n
            if self._is_tty and not self._finished:
                self._render_unlocked()

    def log(self, message: str) -> None:
        """Write a log line above the bar without corrupting either output."""
        with self._lock:
            # Resolved per call so a redirected stream is picked up.
            stream = getattr(sys, self._log_stream)
            if self._is_tty and not self._finished:
                self._clear_unlocked()
                stream.write(message + "\n")
                stream.flush()
                self._render_unlocked()
            else:
                stream.write(message + "\n")
                stream.flush()

    def _line_unlocked(self, *, finished: bool = False) -> str:
        elapsed = max(self._clock() - self._started_at, 0.0)
        fraction = min(self.current / self.total, 1.0) if self.total else 1.0
        filled = int(self._width * fraction)
        bar = "=" * filled + "-" * (self._width - filled)
        percent = int(100 * fraction)
        speed = self.current / elapsed if self.current > 0 and elapsed > 0 else None
        rate = _rate(speed, self.unit)
        if finished:
            timing = f"elapsed {_duration(elapsed)}"
        elif speed is None:
            timing = "ETA --:--:--"
        else:
            remaining = max(self.total - self.current, 0) / speed
            timing = f"ETA {_duration(remaining)}"
        return (
            f"{self.desc}: [{bar}] {self.current}/{self.total} ({percent}%)"
            f" | {rate} | {timing}"
        )

    def _render_unlocked(self) -> None:
        line = self._line_unlocked()
        self._rendered_width = max(self._rendered_width, len(line))
        sys.stdout.write("\r" + line.ljust(self._rendered_width) + "\r")
        sys.stdout.flush()

    def _clear_unlocked(self) -> None:
        sys.stdout.write("\r" + " " * self._rendered_width + "\r")
        sys.stdout.flush()

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            line = self._line_unlocked(finished=True)
            self._finished = True
            if self._is_tty:
                self._clear_unlocked()
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
