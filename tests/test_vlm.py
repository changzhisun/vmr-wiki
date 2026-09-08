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
    assert len(calls) == 3
    assert "secret" not in str(exc.value)
