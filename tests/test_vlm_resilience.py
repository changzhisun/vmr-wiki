"""Offline adversarial response and actual threaded admission tests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import io
import json
import threading
import urllib.error

import pytest

from harness.common import HarnessError
from harness.vlm import VLMClient
from harness.vlm_transport import EndpointGate, retry_delay


def response(content='{"nodes":[]}'):
    return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": content}}],
                                "usage": {"total_tokens": 7}}).encode())


@pytest.mark.parametrize("bad", [b'not json', b'null', b'{"choices":[]}', b'{"choices":[null]}',
                                     b'{"choices":[{"message":{"content":null}}]}',
                                     b'{"choices":[{"message":{"content":" "}}]}',
                                     b'{"choices":[{"message":{"content":[123]}}]}'])
def test_invalid_envelope_retries_then_records_usage(cfg, monkeypatch, bad):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr("harness.vlm.pause", lambda *args: None)
    calls = []
    def request(req, timeout):
        calls.append(req)
        return io.BytesIO(bad) if len(calls) == 1 else response()
    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    assert client.complete("Read text") == '{"nodes":[]}'
    assert len(calls) == 2
    assert client.last_requests[0]["status"] == "invalid_response"
    assert client.last_requests[1]["usage"]["total_tokens"] == 7


def test_persistent_bad_envelope_has_finite_retry_budget(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr("harness.vlm.pause", lambda *args: None)
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b'null'))
    client = VLMClient(cfg["ingest"]["vlm"])
    with pytest.raises(HarnessError, match="bounded retries"):
        client.complete("Read text")
    assert len(client.last_requests) == cfg["ingest"]["vlm"]["max_retries"] + 1
    assert client.gate.active == 0


def test_concurrent_clients_share_endpoint_slots_for_vision_and_text(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = deepcopy(cfg["ingest"]["vlm"])
    config.update(max_concurrent_requests=2, base_url="https://concurrency.example/v1")
    clients = [VLMClient(config) for _ in range(6)]
    image = tmp_path / "frame.jpg"
    image.write_bytes(b'image')
    release, full = threading.Event(), threading.Event()
    lock = threading.Lock()
    active = peak = 0
    def request(req, timeout):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                full.set()
        try:
            assert release.wait(3)
            return response()
        finally:
            with lock:
                active -= 1
    monkeypatch.setattr("urllib.request.urlopen", request)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(client.complete, "Read", [image] if i % 2 else ()) for i, client in enumerate(clients)]
        try:
            assert full.wait(2)
            assert clients[0].gate.active == 2
        finally:
            release.set()
        assert all(f.result() == '{"nodes":[]}' for f in futures)
    assert peak == 2
    assert len({id(c.gate) for c in clients}) == 1
    assert clients[0].gate.active == 0


def test_cancelled_queue_and_backoff_do_not_send_requests(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    cancelled = threading.Event()
    client = VLMClient(cfg["ingest"]["vlm"], cancel_event=cancelled)
    client.gate = EndpointGate(1)
    with client.gate.slot(None, 1):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.complete, "Read text")
            cancelled.set()
            with pytest.raises(HarnessError, match="cancelled"):
                future.result(timeout=2)
    assert client.gate.active == 0
    cancelled.clear()
    calls = []
    def request(req, timeout):
        calls.append(req)
        cancelled.set()
        raise urllib.error.URLError("temporary")
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError, match="cancelled"):
        client.complete("Read text")
    assert len(calls) == 1


def test_admission_timeout_and_long_retry_after(cfg, monkeypatch):
    gate = EndpointGate(1)
    with gate.slot(None, 1):
        with pytest.raises(HarnessError, match="admission wait timed out"):
            with gate.slot(None, .01):
                pytest.fail("Acquired occupied slot")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = VLMClient(cfg["ingest"]["vlm"])
    client.gate = EndpointGate(1)
    calls = []
    def request(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 429, "rate limit", {"Retry-After": "3600"}, None)
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError, match="Retry-After exceeds"):
        client.complete("Read text")
    assert len(calls) == 1 and client.gate.active == 0
    assert client.gate.cooldown > 0
    assert retry_delay(0, 60, "12") == 12
    assert retry_delay(0, 60, "invalid") <= 2


def test_retry_after_backoff_is_shared_and_releases_slot(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    client = VLMClient(cfg["ingest"]["vlm"])
    client.gate = EndpointGate(1)
    delayed, waits, calls = [], [], []
    def defer(seconds, maximum):
        delayed.append(seconds)
        return min(seconds, maximum)
    def pause(seconds, event):
        assert client.gate.active == 0
        waits.append(seconds)
    def request(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "unavailable", {"Retry-After": "12"}, None)
        return response()
    monkeypatch.setattr(client.gate, "defer", defer)
    monkeypatch.setattr("harness.vlm.pause", pause)
    monkeypatch.setattr("urllib.request.urlopen", request)
    client.complete("Read text")
    assert delayed == waits == [12]
    assert client.last_requests[0]["retry_delay_sec"] == 12


def test_transport_limits_do_not_change_frozen_content_identity(cfg):
    from harness.common import ingest_content_hash
    before = ingest_content_hash(cfg)
    cfg["ingest"]["vlm"].update(max_concurrent_requests=1, max_retry_delay_sec=15, queue_timeout_sec=30)
    assert ingest_content_hash(cfg) == before


def test_null_truncation_is_a_reask_signal_and_preserves_usage(cfg, monkeypatch):
    from harness.vlm_transport import ResponseRejected
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    def request(req, timeout):
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "length", "message": {"content": None}}],
                                    "usage": {"total_tokens": 7}}).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    with pytest.raises(ResponseRejected):
        client.complete("Read")
    assert len(client.last_requests) == 1
    assert client.last_requests[0]["usage"]["total_tokens"] == 7


def test_long_retry_after_does_not_poison_second_client(cfg, monkeypatch):
    import time
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    config = {**cfg["ingest"]["vlm"], "base_url": "https://cooldown.example/v1",
              "max_retry_delay_sec": .02, "queue_timeout_sec": 1, "max_retries": 0}
    first, second = VLMClient(config), VLMClient(config)
    assert first.gate is second.gate
    calls = []
    def request(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "rate limit", {"Retry-After": "3600"}, None)
        return response()
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError, match="requested 3600s, limit 0.02s"):
        first.complete("Read")
    assert first.last_requests[0]["server_retry_delay_sec"] == 3600
    assert first.last_requests[0]["shared_cooldown_sec"] == .02
    assert first.gate.cooldown - time.monotonic() <= .02
    assert second.complete("Read") == '{"nodes":[]}'
    assert len(calls) == 2


def test_admission_error_explains_bounded_server_cooldown():
    gate = EndpointGate(1)
    gate.defer(3600, .1)
    with pytest.raises(HarnessError, match="server requested 3600s cooldown; shared cooldown capped at 0.1s"):
        with gate.slot(None, .01):
            pytest.fail("Cooldown not applied")


def test_multipart_answer_filters_non_text_without_retry(cfg, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    content = [{"type": "reasoning_content", "reasoning": "private reasoning"},
               {"type": "reasoning", "text": "not the answer"},
               {"type": "text", "text": '{"nodes":'},
               {"type": "image_url", "image_url": {"url": "unused"}},
               123, {"type": "text", "text": '[]}'}]
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response(content))
    client = VLMClient(cfg["ingest"]["vlm"])
    assert client.complete("Read") == '{"nodes":[]}'
    assert len(client.last_requests) == 1


@pytest.mark.parametrize("status", [401, 403, 404])
def test_shared_http_configuration_errors_are_typed(cfg, monkeypatch, status):
    from harness.vlm_transport import FatalVLMError
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    def request(req, timeout):
        raise urllib.error.HTTPError(req.full_url, status, "secret error body", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    with pytest.raises(FatalVLMError, match=f"HTTP {status}"):
        client.complete("Read")
    assert len(client.last_requests) == 1
