from pathlib import Path

import pytest

from docker.egress_proxy import ProxyHandler
from harness.common import HarnessError
from harness.config import hostname


ROOT = Path(__file__).resolve().parents[1]


def test_proxy_script_is_in_docker_build_context():
    rules = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "!docker/egress_proxy.py" in rules
    assert "COPY --chown=node:node docker/egress_proxy.py" in (
        ROOT / "docker/Dockerfile").read_text(encoding="utf-8")


@pytest.mark.parametrize("payload", [
    b"CONNECT attacker.example:443 HTTP/1.1\r\nHost: attacker.example\r\n\r\n",
    b"CONNECT api.openai.com:80 HTTP/1.1\r\nHost: api.openai.com\r\n\r\n",
    b"GET https://api.openai.com/ HTTP/1.1\r\nHost: api.openai.com\r\n\r\n",
])
def test_proxy_rejects_unlisted_hosts_ports_and_methods(payload):
    class FakeSocket:
        def __init__(self):
            self.chunks = [payload]
            self.sent = bytearray()

        def recv(self, size):
            return self.chunks.pop(0) if self.chunks else b""

        def sendall(self, data):
            self.sent.extend(data)

    ProxyHandler.allowed_hosts = frozenset({"api.openai.com"})
    handler = ProxyHandler.__new__(ProxyHandler)
    handler.request = FakeSocket()
    handler.handle()
    assert handler.request.sent.startswith((b"HTTP/1.1 400", b"HTTP/1.1 403"))


@pytest.mark.parametrize("value", ["127.0.0.1", "localhost", "*.example.com", "-api.example.com"])
def test_egress_config_requires_exact_public_dns_names(value):
    with pytest.raises(HarnessError):
        hostname(value, "allowed host")


def test_egress_config_normalizes_hostnames():
    assert hostname("API.OpenAI.Com", "allowed host") == "api.openai.com"
