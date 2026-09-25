from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agents import claude_code, codex
from vmr.core.errors import HarnessError


from vmr.runtime.types import AgentResult


from vmr.runtime.types import FatalAgentError


from vmr.runtime.types import AgentCancelled


from vmr.runtime.trace import trace_path


from vmr.runtime.trace import _trace_lines


from vmr.runtime.trace import _event_label


from vmr.runtime.trace import readable_trace
from vmr.runtime.trace import redact_trace_line


class _ContainerRunner:
    """Shared Docker lifecycle: the host repository, credentials directory, and
    Docker socket are never mounted, and the agent gets no direct network route."""

    DESTINATION_KEY = {"codex": "CODEX_API_KEY", "claude_code": "ANTHROPIC_API_KEY"}
    BASE_URL_KEY = {"codex": "OPENAI_BASE_URL", "claude_code": "ANTHROPIC_BASE_URL"}

    def _configure(
        self,
        *,
        agent: str,
        key_env: str,
        model: str,
        image: str,
        allowed_hosts,
        base_url,
        timeout_sec: float,
    ) -> None:
        self.agent = agent
        self.model = model
        key = os.environ.get(key_env)
        if not key:
            raise FatalAgentError(f"Set {key_env} before running the {agent} agent")
        # Pin an image ID for the whole batch; tag changes cannot affect later work.
        self.image = self._inspect(image)
        self.destination_key = self.DESTINATION_KEY[agent]
        self.base_url_key = self.BASE_URL_KEY[agent]
        self.env = {**os.environ, self.destination_key: key}
        self.allowed_hosts = tuple(allowed_hosts)
        self.base_url = base_url
        self.timeout_sec = timeout_sec
        self.provenance = {
            "runtime": "docker",
            "image_id": self.image,
            "agent_command": self.agent_command(),
            "api_base_url": self.base_url,
            "egress": {"mode": "allowlist-proxy", "hosts": list(self.allowed_hosts)},
        }

    @staticmethod
    def _inspect(image: str) -> str:
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", "--format", "{{.Id}}", image],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            # Docker Desktop's containerd image store can transiently list and
            # run a local image while its image-inspect endpoint returns 404.
            # Resolve the exact reference through the image listing as a safe
            # fallback; the returned value is still the immutable image ID used
            # for every container in this batch.
            try:
                listed = subprocess.run(
                    ["docker", "image", "ls", "--no-trunc", "--quiet", image],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
            except (
                subprocess.CalledProcessError,
                subprocess.TimeoutExpired,
            ) as fallback_exc:
                raise FatalAgentError(
                    "Docker/image unavailable. Start Docker and build the agent image; see README."
                ) from fallback_exc
            digests = {
                line.strip() for line in listed.stdout.splitlines() if line.strip()
            }
            if len(digests) != 1:
                raise FatalAgentError(
                    "Docker/image unavailable. Start Docker and build the agent image; see README."
                ) from exc
            digest = digests.pop()
        else:
            digest = result.stdout.strip()
        if not digest.startswith("sha256:"):
            raise FatalAgentError("Docker did not return an immutable image ID")
        return digest

    def agent_command(self) -> list[str]:
        adapter = codex if self.agent == "codex" else claude_code
        return adapter.command(self.model)

    @staticmethod
    def _bind(source: Path, destination: str, *, readonly: bool = False) -> list[str]:
        if "," in str(source):
            raise HarnessError("Docker bind paths must not contain commas")
        spec = f"type=bind,src={source},dst={destination}"
        return ["--mount", (spec + ",readonly") if readonly else spec]

    def _shared_options(self, name: str, network: str) -> list[str]:
        proxy = "http://egress-proxy:8080"
        # The endpoint is provenance, not a credential, so it is passed inline.
        endpoint = (
            ["--env", f"{self.base_url_key}={self.base_url}"] if self.base_url else []
        )
        return [
            "docker",
            "run",
            "--rm",
            "--interactive",
            "--init",
            "--name",
            name,
            "--network",
            network,
            # Keep Docker's local service discovery for the proxy alias but
            # give its embedded resolver no usable external upstream.
            "--dns",
            "127.0.0.1",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            "1000:1000",
            *endpoint,
            "--env",
            self.destination_key,
            "--env",
            f"HTTPS_PROXY={proxy}",
            "--env",
            f"HTTP_PROXY={proxy}",
            "--env",
            f"ALL_PROXY={proxy}",
            "--env",
            "NO_PROXY=",
            "--env",
            "DISABLE_AUTOUPDATER=1",
            "--env",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1",
        ]

    @staticmethod
    def _docker(
        command: list[str], message: str, *, timeout: int = 30
    ) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                command, capture_output=True, text=True, check=True, timeout=timeout
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
            OSError,
        ) as exc:
            raise HarnessError(message) from exc

    def _start_egress_proxy(self, network: str, name: str) -> None:
        command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            name,
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            "1000:1000",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "0.5",
            "--tmpfs",
            "/tmp:rw,nosuid,size=16m,mode=1777",
            self.image,
            "python3",
            "/opt/vmr/egress_proxy.py",
        ]
        for host in self.allowed_hosts:
            command.extend(["--allow-host", host])
        self._docker(command, "Could not start the allowlisted egress proxy")
        self._docker(
            ["docker", "network", "connect", "--alias", "egress-proxy", network, name],
            "Could not attach the egress proxy to the isolated network",
        )
        # Probe through the same isolated network and DNS alias the agent will
        # use. Besides exercising the actual route, this avoids Docker
        # Desktop's occasionally stale container-name lookup in `docker exec`.
        probe_script = (
            "import socket,time\n"
            "for attempt in range(50):\n"
            " try:\n"
            "  socket.create_connection(('egress-proxy',8080),1).close(); break\n"
            " except OSError:\n"
            "  if attempt == 49: raise\n"
            "  time.sleep(.1)\n"
        )
        probe = [
            "docker",
            "run",
            "--rm",
            "--network",
            network,
            "--dns",
            "127.0.0.1",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user",
            "1000:1000",
            "--pids-limit",
            "16",
            "--memory",
            "64m",
            "--cpus",
            "0.25",
            self.image,
            "python3",
            "-c",
            probe_script,
        ]
        self._docker(
            probe,
            "Allowlisted egress proxy did not become reachable from the isolated network",
            timeout=30,
        )

    @staticmethod
    def _remove_container(name: str) -> None:
        removed = subprocess.run(
            ["docker", "rm", "--force", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if removed.returncode and "No such container" not in removed.stderr:
            raise HarnessError(f"Could not confirm cleanup of container {name}")

    def _wait_for_agent(
        self,
        process: subprocess.Popen,
        prompt: str,
        cancel_event,
        *,
        timeout_sec: float | None = None,
    ) -> AgentResult:
        if cancel_event is not None and cancel_event.is_set():
            raise AgentCancelled("Agent cancelled")
        assert process.stdin is not None
        try:
            process.stdin.write(prompt.encode("utf-8"))
            process.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            with suppress(BrokenPipeError):
                process.stdin.close()
            process.stdin = None
        deadline = time.monotonic() + (
            self.timeout_sec if timeout_sec is None else timeout_sec
        )
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AgentCancelled("Agent cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return AgentResult(None, timed_out=True)
            try:
                return AgentResult(process.wait(timeout=min(0.25, remaining)))
            except subprocess.TimeoutExpired:
                continue

    def _write_readable_stdout(self, trace: Path, stdout: Path) -> None:
        try:
            content = readable_trace(self.agent, trace)
        except Exception as exc:
            content = (
                f"[could not derive readable stdout: {type(exc).__name__}: {exc}]\n"
            ).encode("utf-8", errors="replace")
        try:
            stdout.write_bytes(content)
        except OSError:
            pass  # Never hide the agent or cleanup failure with a derived-log failure.

    def _relay_trace(self, stream, destination) -> None:
        """Copy CLI events into the trace without embedded image bytes."""
        paths = {}
        try:
            for line in iter(stream.readline, b""):
                try:
                    destination.write(redact_trace_line(line, paths))
                except Exception:
                    destination.write(
                        b'{"type":"harness.trace","omitted":"unreadable event"}\n'
                    )
            destination.flush()
        except OSError:
            pass
        finally:
            stream.close()

    def run(
        self,
        workspace: Path,
        prompt: str,
        stdout: Path,
        stderr: Path,
        *,
        cancel_event=None,
        timeout_sec: float | None = None,
        **extra,
    ) -> AgentResult:
        name = f"vmr-{uuid.uuid4().hex}"
        network = f"{name}-internal"
        proxy_name = f"{name}-proxy"
        trace = trace_path(stdout)
        process = None
        network_created = False
        result = None
        events = None
        err = None
        copier = None
        try:
            with trace.open("wb") as header:
                record = {
                    "type": "harness.input",
                    "version": 1,
                    "time": time.time(),
                    "agent": self.agent,
                    "model": self.model,
                    "prompt": prompt,
                }
                header.write(
                    (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
                )
            stderr.write_bytes(b"")
            self._docker(
                ["docker", "network", "create", "--internal", network],
                "Could not create an isolated agent network",
            )
            network_created = True
            self._start_egress_proxy(network, proxy_name)
            command = self.docker_command(workspace, name, network, **extra)
            events = trace.open("ab")
            err = stderr.open("wb")
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=err,
                env=self.env,
                start_new_session=True,
            )
            copier = threading.Thread(
                target=self._relay_trace,
                args=(process.stdout, events),
                daemon=True,
            )
            copier.start()
            result = self._wait_for_agent(
                process, prompt, cancel_event, timeout_sec=timeout_sec
            )
        finally:
            try:
                # Kill the container as well as its client, including on Ctrl-C.
                # This prevents descendants from writing output after timeout/cleanup.
                try:
                    self._remove_container(name)
                finally:
                    if process is not None and process.poll() is None:
                        # docker rm already terminates the workload's process tree.
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=10)
                    try:
                        self._remove_container(proxy_name)
                    finally:
                        if network_created:
                            removed = subprocess.run(
                                ["docker", "network", "rm", network],
                                capture_output=True,
                                text=True,
                                timeout=30,
                            )
                            if (
                                removed.returncode
                                and "not found" not in removed.stderr.lower()
                            ):
                                raise HarnessError(
                                    f"Could not remove isolated network {network}"
                                )
            finally:
                if copier is not None:
                    copier.join(timeout=30)
                if events is not None:
                    events.close()
                if err is not None:
                    err.close()
                self._write_readable_stdout(trace, stdout)
        if result is None:
            raise HarnessError("Agent process ended without a result")
        return result


class DockerRunner(_ContainerRunner):
    """Query agent: one read-only input in, one prediction out."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        agent = cfg["query"]["agent"]
        if cfg["query"]["model"] in ("AGENT_MODEL",) or cfg["query"][
            "model"
        ].startswith("REPLACE_"):
            raise HarnessError("Set AGENT_MODEL or configure an explicit query model")
        self._configure(
            agent=agent,
            key_env=cfg["query"]["api_key_env"][agent],
            model=cfg["query"]["model"],
            image=cfg["query"]["container_image"],
            allowed_hosts=cfg["query"]["egress_allowed_hosts"][agent],
            base_url=cfg["query"]["base_url"][agent],
            timeout_sec=cfg["query"]["timeout_sec"],
        )

    def docker_command(self, workspace: Path, name: str, network: str) -> list[str]:
        # A read-only root mount plus a nested writable output mount is a filesystem
        # boundary, unlike chmod or an agent's workspace-write sandbox alone.
        return [
            *self._shared_options(name, network),
            "--pids-limit",
            "256",
            "--memory",
            "4g",
            "--cpus",
            "2",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m,mode=1777",
            "--tmpfs",
            "/home/node:rw,nosuid,size=256m,uid=1000,gid=1000,mode=700",
            *self._bind(workspace, "/workspace", readonly=True),
            *self._bind(workspace / "output", "/workspace/output"),
            "--workdir",
            "/workspace",
            self.image,
            *self.agent_command(),
        ]


class AgentIngestRunner(_ContainerRunner):
    """Agentic ingest: one read-only video in, one wiki out.

    The video is the only data mounted, so the container holds no query, no
    ground truth and no public dataset identifier to recognize.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        options = cfg["ingest"]["agentic"]
        self.options = options
        agent = options["agent"]
        self._configure(
            agent=agent,
            key_env=options["api_key_env"][agent],
            model=options["model"],
            image=options["container_image"],
            allowed_hosts=options["egress_allowed_hosts"][agent],
            base_url=options["base_url"][agent],
            timeout_sec=options["timeout_sec"],
        )
        self.provenance = {**self.provenance, "role": "agentic_ingest"}

    def docker_command(
        self, workspace: Path, name: str, network: str, *, video: Path, scratch: Path
    ) -> list[str]:
        options = self.options
        # Frame extraction needs real disk, so scratch is a host bind rather
        # than a tmpfs that would count against the container memory limit.
        return [
            *self._shared_options(name, network),
            "--pids-limit",
            str(options["pids_limit"]),
            "--memory",
            f"{options['memory_gb']}g",
            "--cpus",
            str(options["cpus"]),
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m,mode=1777",
            "--tmpfs",
            "/home/node:rw,nosuid,size=256m,uid=1000,gid=1000,mode=700",
            *self._bind(workspace, "/workspace", readonly=True),
            *self._bind(workspace / "output", "/workspace/output"),
            *self._bind(video, "/input/video.mp4", readonly=True),
            *self._bind(scratch, "/scratch"),
            "--workdir",
            "/workspace",
            self.image,
            *self.agent_command(),
        ]
