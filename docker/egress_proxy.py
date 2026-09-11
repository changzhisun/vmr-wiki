"""Minimal allowlisted HTTP CONNECT proxy for the isolated agent network."""
from __future__ import annotations

import argparse
import select
import socket
import socketserver


MAX_HEADER = 64 * 1024
# Poll interval only. Idle CONNECT tunnels must stay up: Claude Code streams
# SSE with long TTFB, and agentic ingest can spend minutes in ffmpeg between
# requests. Closing here produced "socket connection was closed unexpectedly".
IDLE_SELECT_SEC = 60


def enable_keepalive(sock: socket.socket) -> None:
    """Detect dead peers without tearing down a quiet live tunnel."""
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    except OSError:
        return  # Keepalive is diagnostic; failure must not break the tunnel.
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return
    for option_name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPINTVL", 10),
                               ("TCP_KEEPCNT", 6)):
        option = getattr(socket, option_name, None)
        if option is not None:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, option, value)
            except OSError:
                pass


def relay_tunnel(client: socket.socket, upstream: socket.socket,
                 *, idle_select: float = IDLE_SELECT_SEC) -> None:
    sockets = (client, upstream)
    while True:
        try:
            readable, _, _ = select.select(sockets, (), (), idle_select)
        except (OSError, ValueError):
            return
        if not readable:
            continue
        for source in readable:
            try:
                data = source.recv(64 * 1024)
            except OSError:
                return
            if not data:
                return
            destination = upstream if source is client else client
            try:
                destination.sendall(data)
            except OSError:
                return


class ProxyHandler(socketserver.BaseRequestHandler):
    allowed_hosts: frozenset[str] = frozenset()

    def handle(self) -> None:
        header = bytearray()
        while b"\r\n\r\n" not in header:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            header.extend(chunk)
            if len(header) > MAX_HEADER:
                self.request.sendall(b"HTTP/1.1 431 Request Header Fields Too Large\r\n\r\n")
                return
        first_line = bytes(header).split(b"\r\n", 1)[0]
        try:
            method, authority, _ = first_line.decode("ascii").split(" ", 2)
            host, separator, port_text = authority.rpartition(":")
            port = int(port_text) if separator else -1
        except (UnicodeDecodeError, ValueError):
            self.request.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            return
        if method != "CONNECT" or host.lower() not in self.allowed_hosts or port != 443:
            self.request.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            return
        try:
            upstream = socket.create_connection((host, port), timeout=15)
        except OSError:
            self.request.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            return
        with upstream:
            enable_keepalive(self.request)
            enable_keepalive(upstream)
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            relay_tunnel(self.request, upstream)


class ThreadingProxy(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--allow-host", action="append", required=True)
    args = parser.parse_args()
    ProxyHandler.allowed_hosts = frozenset(host.lower() for host in args.allow_host)
    with ThreadingProxy((args.listen, args.port), ProxyHandler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
