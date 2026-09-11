from __future__ import annotations

import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from agents import claude_code, codex
from harness.common import HarnessError


@dataclass
class AgentResult:
    exit_code: int | None
    timed_out: bool = False


class FatalAgentError(HarnessError):
    """Shared credential, image or endpoint failure, not one video's fault."""


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

    def run(self, workspace: Path, prompt: str, stdout: Path, stderr: Path, **extra) -> AgentResult:
        name = f"vmr-{uuid.uuid4().hex}"
        network = f"{name}-internal"
        proxy_name = f"{name}-proxy"
        process = None
        network_created = False
        try:
            self._docker(["docker", "network", "create", "--internal", network],
                         "Could not create an isolated agent network")
            network_created = True
            self._start_egress_proxy(network, proxy_name)
            command = self.docker_command(workspace, name, network, **extra)
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                           env=self.env, start_new_session=True)
                try:
                    process.communicate(prompt.encode("utf-8"), timeout=self.timeout_sec)
                    return AgentResult(process.returncode)
                except subprocess.TimeoutExpired:
                    return AgentResult(None, timed_out=True)
        finally:
            # Kill the container as well as its client, including on Ctrl-C.
            # This prevents descendants from writing output after timeout/cleanup.
            try:
                self._remove_container(name)
            finally:
                if process is not None and process.poll() is None:
                    # docker rm already terminates the workload's process tree.
                    # Allow its client to observe that exit before signaling it;
                    # killpg can race with the client's teardown on macOS.
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
                            raise HarnessError(f"Could not remove isolated network {network}")


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
