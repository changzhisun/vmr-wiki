"""Minimal allowlisted HTTP CONNECT proxy for the isolated agent network."""
from __future__ import annotations

import argparse
import select
import socket
import socketserver


MAX_HEADER = 64 * 1024


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
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            sockets = (self.request, upstream)
            while True:
                readable, _, _ = select.select(sockets, (), (), 60)
                if not readable:
                    return
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)


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
