from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agents import claude_code, codex
from harness.common import HarnessError


@dataclass
class AgentResult:
    exit_code: int | None
    timed_out: bool = False


class FatalAgentError(HarnessError):
    """Shared credential, image or endpoint failure, not one video's fault."""


class AgentCancelled(HarnessError):
    """A running agent was stopped cooperatively after batch cancellation."""


def trace_path(stdout: Path) -> Path:
    """Return the raw event-stream path paired with a readable stdout log."""
    suffix = ".stdout.log"
    if stdout.name.endswith(suffix):
        return stdout.with_name(stdout.name[:-len(suffix)] + ".trace.jsonl")
    return stdout.with_name(stdout.name + ".trace.jsonl")


def _trace_lines(raw: bytes | Path) -> Iterable[bytes]:
    if isinstance(raw, Path):
        with raw.open("rb") as stream:
            yield from stream
    else:
        yield from raw.splitlines()


def _event_label(event: dict) -> str:
    label = str(event.get("type") or "unknown")
    subtype = event.get("subtype")
    return f"{label}/{subtype}" if subtype else label


def readable_trace(agent: str, raw: bytes | Path) -> bytes:
    """Stream a trace and extract its final answer or a compact diagnostic.

    Codex and Claude emit different JSONL event schemas. The complete stream is
    the audit artifact; stdout remains a small human-readable diagnostic.
    """
    final_answer = None
    assistant_text = None
    fallback: deque[str] = deque(maxlen=32)
    last_event = None
    for raw_line in _trace_lines(raw):
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            fallback.append(line[-8192:])
            continue
        if not isinstance(event, dict) or event.get("type") == "harness.input":
            continue
        last_event = _event_label(event)
        if agent == "claude_code" and event.get("type") == "result":
            result = event.get("result")
            if isinstance(result, str) and result:
                final_answer = result
        elif agent == "claude_code" and event.get("type") == "assistant":
            message = event.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                texts = [block.get("text") for block in content
                         if isinstance(block, dict) and block.get("type") == "text"
                         and isinstance(block.get("text"), str) and block.get("text")]
                if texts:
                    assistant_text = "\n".join(texts)
        elif agent == "codex" and event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str) and text:
                    final_answer = text
    if final_answer:
        text = final_answer
    elif assistant_text:
        text = assistant_text
    elif fallback:
        text = "\n".join(fallback)
    elif last_event:
        text = f"[no final answer event; last agent event: {last_event}]"
    else:
        text = "[no agent events were emitted]"
    return (text + ("\n" if text else "")).encode("utf-8")


class _ContainerRunner:
    """Shared Docker lifecycle: the host repository, credentials directory, and
    Docker socket are never mounted, and the agent gets no direct network route."""

    DESTINATION_KEY = {"codex": "CODEX_API_KEY", "claude_code": "ANTHROPIC_API_KEY"}
    BASE_URL_KEY = {"codex": "OPENAI_BASE_URL", "claude_code": "ANTHROPIC_BASE_URL"}

    def _configure(self, *, agent: str, key_env: str, model: str, image: str,
                   allowed_hosts, base_url, timeout_sec: float) -> None:
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
        self.provenance = {"runtime": "docker", "image_id": self.image,
                           "agent_command": self.agent_command(),
                           "api_base_url": self.base_url,
                           "egress": {"mode": "allowlist-proxy", "hosts": list(self.allowed_hosts)}}

    @staticmethod
    def _inspect(image: str) -> str:
        try:
            result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                    capture_output=True, text=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise FatalAgentError("Docker/image unavailable. Start Docker and build the agent image; see README.") from exc
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
        endpoint = ["--env", f"{self.base_url_key}={self.base_url}"] if self.base_url else []
        return ["docker", "run", "--rm", "--interactive", "--init", "--name", name,
                "--network", network,
                # Keep Docker's local service discovery for the proxy alias but
                # give its embedded resolver no usable external upstream.
                "--dns", "127.0.0.1",
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--user", "1000:1000", *endpoint,
                "--env", self.destination_key,
                "--env", f"HTTPS_PROXY={proxy}", "--env", f"HTTP_PROXY={proxy}",
                "--env", f"ALL_PROXY={proxy}", "--env", "NO_PROXY=",
                "--env", "DISABLE_AUTOUPDATER=1", "--env", "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"]

    @staticmethod
    def _docker(command: list[str], message: str, *, timeout: int = 30) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True, timeout=timeout)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            raise HarnessError(message) from exc

    def _start_egress_proxy(self, network: str, name: str) -> None:
        command = ["docker", "run", "--detach", "--rm", "--name", name,
                   "--network", "bridge", "--read-only", "--cap-drop=ALL",
                   "--security-opt=no-new-privileges", "--user", "1000:1000",
                   "--pids-limit", "64", "--memory", "256m", "--cpus", "0.5",
                   "--tmpfs", "/tmp:rw,nosuid,size=16m,mode=1777", self.image,
                   "python3", "/opt/vmr/egress_proxy.py"]
        for host in self.allowed_hosts:
            command.extend(["--allow-host", host])
        self._docker(command, "Could not start the allowlisted egress proxy")
        self._docker(["docker", "network", "connect", "--alias", "egress-proxy", network, name],
                     "Could not attach the egress proxy to the isolated network")
        probe = ["docker", "exec", name, "python3", "-c",
                 "import socket; socket.create_connection(('127.0.0.1',8080),1).close()"]
        for _ in range(50):
            result = subprocess.run(probe, capture_output=True, text=True)
            if result.returncode == 0:
                return
            time.sleep(0.1)
        raise HarnessError("Allowlisted egress proxy did not become ready; rebuild the agent image")

    @staticmethod
    def _remove_container(name: str) -> None:
        removed = subprocess.run(["docker", "rm", "--force", name],
                                 capture_output=True, text=True, timeout=30)
        if removed.returncode and "No such container" not in removed.stderr:
            raise HarnessError(f"Could not confirm cleanup of container {name}")

    def _wait_for_agent(self, process: subprocess.Popen, prompt: str, cancel_event) -> AgentResult:
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
        deadline = time.monotonic() + self.timeout_sec
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
            content = (f"[could not derive readable stdout: {type(exc).__name__}: {exc}]\n"
                       ).encode("utf-8", errors="replace")
        try:
            stdout.write_bytes(content)
        except OSError:
            pass  # Never hide the agent or cleanup failure with a derived-log failure.

    def run(self, workspace: Path, prompt: str, stdout: Path, stderr: Path, *,
            cancel_event=None, **extra) -> AgentResult:
        name = f"vmr-{uuid.uuid4().hex}"
        network = f"{name}-internal"
        proxy_name = f"{name}-proxy"
        trace = trace_path(stdout)
        process = None
        network_created = False
        result = None
        try:
            with trace.open("wb") as events:
                header = {"type": "harness.input", "version": 1, "time": time.time(),
                          "agent": self.agent, "model": self.model, "prompt": prompt}
                events.write((json.dumps(header, ensure_ascii=False) + "\n").encode("utf-8"))
            stderr.write_bytes(b"")
            self._docker(["docker", "network", "create", "--internal", network],
                         "Could not create an isolated agent network")
            network_created = True
            self._start_egress_proxy(network, proxy_name)
            command = self.docker_command(workspace, name, network, **extra)
            with trace.open("ab") as events, stderr.open("wb") as err:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=err,
                                           env=self.env, start_new_session=True)
                result = self._wait_for_agent(process, prompt, cancel_event)
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
                            removed = subprocess.run(["docker", "network", "rm", network],
                                                     capture_output=True, text=True, timeout=30)
                            if removed.returncode and "not found" not in removed.stderr.lower():
                                raise HarnessError(
                                    f"Could not remove isolated network {network}")
            finally:
                self._write_readable_stdout(trace, stdout)
        if result is None:
            raise HarnessError("Agent process ended without a result")
        return result


class DockerRunner(_ContainerRunner):
    """Query agent: one frozen wiki in, one prediction out."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        agent = cfg["query"]["agent"]
        if cfg["query"]["model"].startswith("REPLACE_"):
            raise HarnessError("Configure an explicit query model")
        self._configure(agent=agent, key_env=cfg["query"]["api_key_env"][agent],
                        model=cfg["query"]["model"], image=cfg["query"]["container_image"],
                        allowed_hosts=cfg["query"]["egress_allowed_hosts"][agent],
                        base_url=cfg["query"]["base_url"][agent],
                        timeout_sec=cfg["query"]["timeout_sec"])

    def docker_command(self, workspace: Path, name: str, network: str) -> list[str]:
        # A read-only root mount plus a nested writable output mount is a filesystem
        # boundary, unlike chmod or an agent's workspace-write sandbox alone.
        return [*self._shared_options(name, network),
                "--pids-limit", "256", "--memory", "4g", "--cpus", "2",
                "--tmpfs", "/tmp:rw,nosuid,size=512m,mode=1777",
                "--tmpfs", "/home/node:rw,nosuid,size=256m,uid=1000,gid=1000,mode=700",
                *self._bind(workspace, "/workspace", readonly=True),
                *self._bind(workspace / "output", "/workspace/output"),
                "--workdir", "/workspace", self.image, *self.agent_command()]


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
        self._configure(agent=agent, key_env=options["api_key_env"][agent],
                        model=options["model"], image=options["container_image"],
                        allowed_hosts=options["egress_allowed_hosts"][agent],
                        base_url=options["base_url"][agent],
                        timeout_sec=options["timeout_sec"])
        self.provenance = {**self.provenance, "role": "agentic_ingest"}

    def docker_command(self, workspace: Path, name: str, network: str, *,
                       video: Path, scratch: Path) -> list[str]:
        options = self.options
        # Frame extraction needs real disk, so scratch is a host bind rather
        # than a tmpfs that would count against the container memory limit.
        return [*self._shared_options(name, network),
                "--pids-limit", str(options["pids_limit"]),
                "--memory", f"{options['memory_gb']}g", "--cpus", str(options["cpus"]),
                "--tmpfs", "/tmp:rw,nosuid,size=512m,mode=1777",
                "--tmpfs", "/home/node:rw,nosuid,size=256m,uid=1000,gid=1000,mode=700",
                *self._bind(workspace, "/workspace", readonly=True),
                *self._bind(workspace / "output", "/workspace/output"),
                *self._bind(video, "/input/video.mp4", readonly=True),
                *self._bind(scratch, "/scratch"),
                "--workdir", "/workspace", self.image, *self.agent_command()]
