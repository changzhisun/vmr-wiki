import base64
import io
import json
import urllib.error

import pytest

from harness.common import HarnessError
from harness.vlm import VLMClient


def test_fixed_image_payload_and_caption(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image bytes")
    seen = []
    def request(req, timeout):
        seen.append(json.loads(req.data))
        assert req.get_header("Authorization") == "Bearer fake-key"
        return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "A person."}}]}).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    assert VLMClient(cfg["ingest"]["vlm"]).caption(image) == "A person."
    assert seen[0]["temperature"] == 0
    assert len(seen[0]["messages"]) == 1
    assert seen[0]["messages"][0]["content"][0]["text"] == cfg["ingest"]["vlm"]["prompt"]
    assert seen[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_multi_image_payload_preserves_chronological_order(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    images = [tmp_path / "first.jpg", tmp_path / "second.jpg"]
    images[0].write_bytes(b"first")
    images[1].write_bytes(b"second")
    seen = []

    def request(req, timeout):
        seen.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": "A sequence."}}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    assert VLMClient(cfg["ingest"]["vlm"]).caption(images) == "A sequence."
    content = seen[0]["messages"][0]["content"]
    assert "chronological order" in content[0]["text"]
    encoded = [part["image_url"]["url"].split(",", 1)[1] for part in content[1:]]
    assert [base64.b64decode(value) for value in encoded] == [b"first", b"second"]


def test_dense_prompt_injects_frame_timestamps(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    cfg["ingest"]["vlm"]["prompt"] = "Timeline:\n{{FRAME_TIMESTAMPS}}\nReturn JSON."
    images = [tmp_path / "000001.jpg", tmp_path / "000002.jpg"]
    for image in images:
        image.write_bytes(b"image")
    seen = []

    def request(req, timeout):
        seen.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": '{"events":[]}'}}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    assert client.caption(images, timestamps=[0.0, 1.0]) == '{"events":[]}'
    prompt = seen[0]["messages"][0]["content"][0]["text"]
    # Frame numbers are 1-based while their timestamps start at zero, so the
    # timeline lists only the values the window may use.
    assert "Timeline:\n0.0, 1.0\n" in prompt
    assert "000001.jpg" not in prompt
    assert "{{FRAME_TIMESTAMPS}}" not in prompt
    with pytest.raises(HarnessError, match="counts differ"):
        client.caption(images, timestamps=[0.0])


def test_dense_prompt_marks_soft_center_target_and_schema(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    default_prompt = cfg["ingest"]["vlm"]["prompt"]
    assert "guessing gender" not in default_prompt
    assert "watermarks" not in default_prompt
    assert "zero real-world duration" not in default_prompt
    cfg["ingest"]["vlm"]["prompt"] = "Timeline: {{FRAME_TIMESTAMPS}}"
    images = [tmp_path / f"{index}.jpg" for index in range(5)]
    for image in images:
        image.write_bytes(b"image")
    seen = []

    def request(req, timeout):
        seen.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": '{"events":[]}'}}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    client.caption(images, timestamps=[0, 1, 2, 3, 4], target_timestamps=[2, 3])
    prompt = seen[0]["messages"][0]["content"][0]["text"]
    assert "Centered target region: 2, 3" in prompt
    assert "may still use any listed window timestamp" in prompt
    assert "state, action, or transition" in prompt
    assert "Use state for a stable visible condition" in prompt
    assert "instead of guessing gender" in prompt
    assert "overlay text" in prompt
    with pytest.raises(HarnessError, match="nonempty subset"):
        client.caption(images, timestamps=[0, 1, 2, 3, 4], target_timestamps=[])
    with pytest.raises(HarnessError, match="requires frame timestamps"):
        client.caption(images, target_timestamps=[2])


def test_rejection_is_fed_back_to_the_captioner(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    cfg["ingest"]["vlm"]["prompt"] = "Timeline:\n{{FRAME_TIMESTAMPS}}\nReturn JSON."
    image = tmp_path / "000001.jpg"
    image.write_bytes(b"image")
    seen = []

    def request(req, timeout):
        seen.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {"content": '{"events":[]}'}}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    client.caption([image], timestamps=[0.0], correction="end 14.0 is outside its window")
    prompt = seen[0]["messages"][0]["content"][0]["text"]
    assert "previous answer was rejected: end 14.0 is outside its window" in prompt


def test_retry_transient_only_and_no_credential_in_errors(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setattr("time.sleep", lambda _: None)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    calls = []
    def request(req, timeout):
        calls.append(req)
        raise urllib.error.HTTPError(req.full_url, 429, "secret", {}, None)
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError) as exc:
        VLMClient(cfg["ingest"]["vlm"]).caption(image)
    assert len(calls) == cfg["ingest"]["vlm"]["max_retries"] + 1
    assert "secret" not in str(exc.value)


def test_length_with_visible_caption_is_rejected(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    calls = []
    def request(req, timeout):
        calls.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "length",
                         "message": {"content": "A person stands indoors near a doorway."}}]
        }).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError, match="finish_reason='length'"):
        VLMClient(cfg["ingest"]["vlm"]).caption(image)
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == cfg["ingest"]["vlm"]["max_tokens"]


def test_qwen3_disables_thinking(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    cfg["ingest"]["vlm"]["model"] = "Qwen3-VL-8B-Instruct"
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    payloads = []
    def request(req, timeout):
        payloads.append(json.loads(req.data))
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop",
                         "message": {"content": "A person stands indoors."}}]
        }).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    assert VLMClient(cfg["ingest"]["vlm"]).caption(image) == "A person stands indoors."
    assert payloads[0]["messages"][0]["content"][0]["text"].endswith("/no_think")
    assert payloads[0]["enable_thinking"] is False
    assert payloads[0]["max_tokens"] == cfg["ingest"]["vlm"]["max_tokens"]


def test_think_block_is_stripped_from_caption(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    def request(req, timeout):
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "stop", "message": {
                "content": "<think>plan</think>\nA red chair is in the room."}}]
        }).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    assert VLMClient(cfg["ingest"]["vlm"]).caption(image) == "A red chair is in the room."


def test_content_filter_fails_immediately(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    calls = []
    def request(req, timeout):
        calls.append(req)
        return io.BytesIO(json.dumps({
            "choices": [{"finish_reason": "content_filter", "message": {"content": ""}}]
        }).encode())
    monkeypatch.setattr("urllib.request.urlopen", request)
    with pytest.raises(HarnessError, match="finish_reason='content_filter'"):
        VLMClient(cfg["ingest"]["vlm"]).caption(image)
    assert len(calls) == 1


def test_encoded_images_are_reused_and_invalidated(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"first")
    client = VLMClient(cfg["ingest"]["vlm"])
    first = client._image_url(image)
    original = type(image).read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(type(image), "read_bytes", read)
    assert client._image_url(image) == first
    assert reads == []
    image.write_bytes(b"changed image")
    assert client._image_url(image) != first
    assert reads == [image]


def test_request_usage_and_errors_are_recorded_without_credentials(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    monkeypatch.setattr("time.sleep", lambda _: None)
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    calls = []

    def request(req, timeout):
        calls.append(req)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "fake-key", {}, None)
        return io.BytesIO(json.dumps({"usage": {"prompt_tokens": 8, "completion_tokens": 2,
                                               "total_tokens": 10}, "choices": [{
            "finish_reason": "stop", "message": {"content": "A person."}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", request)
    client = VLMClient(cfg["ingest"]["vlm"])
    assert client.caption(image) == "A person."
    assert client.last_requests[0]["http_status"] == 429
    assert client.last_requests[1]["usage"]["total_tokens"] == 10
    assert client.last_requests[1]["elapsed_sec"] >= 0
    assert "fake-key" not in json.dumps(client.last_requests)
