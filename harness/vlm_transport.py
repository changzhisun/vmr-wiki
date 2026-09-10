"""Process-wide endpoint admission and bounded, cancellation-aware backoff."""
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import random
import threading
import time
from weakref import WeakValueDictionary

from harness.common import HarnessError


class ResponseRejected(HarnessError):
    """A completed HTTP request whose answer needs a changed generation prompt."""


class FatalVLMError(HarnessError):
    """Shared authentication, endpoint or model configuration failure."""


def check_cancelled(event):
    if event is not None and event.is_set():
        raise HarnessError("VLM request cancelled")


def pause(seconds, event):
    if event is None:
        time.sleep(seconds)
    elif event.wait(seconds):
        raise HarnessError("VLM request cancelled")


def retry_delay(attempt, maximum, retry_after=None):
    delay = min(maximum, (2 ** min(attempt, 20)) + random.uniform(0, 1))
    if retry_after:
        try:
            requested = float(retry_after)
        except (TypeError, ValueError):
            try:
                requested = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                requested = 0
        if math.isfinite(requested):
            # Preserve the requested delay for diagnostics and fail/resume;
            # EndpointGate separately bounds its shared cooldown.
            return max(delay, requested)
    return delay


class EndpointGate:
    def __init__(self, limit):
        self.limit, self.active, self.cooldown = limit, 0, 0.0
        self.condition = threading.Condition()
        self.cooldown_reason = ""

    def defer(self, seconds, maximum=60):
        applied = min(seconds, maximum)
        with self.condition:
            until = time.monotonic() + applied
            if until >= self.cooldown:
                self.cooldown = until
                self.cooldown_reason = f"server requested {seconds:g}s cooldown; shared cooldown capped at {applied:g}s"
            self.condition.notify_all()
        return applied

    @contextmanager
    def slot(self, event, timeout):
        started = time.monotonic()
        with self.condition:
            while True:
                check_cancelled(event)
                now = time.monotonic()
                if now - started >= timeout:
                    reason = f" ({self.cooldown_reason})" if now < self.cooldown else ""
                    raise HarnessError(f"VLM admission wait timed out{reason}; retry from checkpoint")
                if self.active < self.limit and now >= self.cooldown:
                    self.active += 1
                    break
                self.condition.wait(min(0.2, timeout - (now - started)))
        try:
            yield time.monotonic() - started
        finally:
            with self.condition:
                self.active -= 1
                self.condition.notify_all()


_gates = WeakValueDictionary()
_lock = threading.Lock()


def endpoint_gate(url, limit):
    # Share across model instances, videos and API keys targeting one endpoint.
    # Multiple OS processes must divide the server quota themselves.
    key = url.rstrip("/").lower()
    with _lock:
        gate = _gates.get(key)
        if gate is None:
            gate = EndpointGate(limit)
            _gates[key] = gate
        else:
            with gate.condition:
                # Deliberately monotone while this gate has live clients: a
                # larger new quota cannot override a peer's smaller quota.
                # It resets only after all owners release this shared gate.
                gate.limit = min(gate.limit, limit)
        return gate
