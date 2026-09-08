from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from agents import claude_code, codex
from harness.common import HarnessError


@dataclass
class AgentResult:
    exit_code: int | None
    timed_out: bool = False


class DockerRunner:
    """The host repository, credentials directory, and Docker socket are never mounted."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.agent = cfg["query"]["agent"]
        key_env = cfg["query"]["api_key_env"][self.agent]
        key = os.environ.get(key_env)
        if not key:
            raise HarnessError(f"Set {key_env} before running queries")
        if cfg["query"]["model"].startswith("REPLACE_"):
            raise HarnessError("Configure an explicit query model")
        # Pin an image ID for the whole experiment; tag changes cannot affect later queries.
        self.image = self._inspect(cfg["query"]["container_image"])
        destination = "CODEX_API_KEY" if self.agent == "codex" else "ANTHROPIC_API_KEY"
        self.env = {**os.environ, destination: key}
        self.destination_key = destination
        self.provenance = {"runtime": "docker", "image_id": self.image,
                           "agent_command": self.agent_command()}

    @staticmethod
    def _inspect(image: str) -> str:
        try:
            result = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                    capture_output=True, text=True, check=True, timeout=30)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise HarnessError("Docker/image unavailable. Start Docker and build the agent image; see README.") from exc
        digest = result.stdout.strip()
        if not digest.startswith("sha256:"):
            raise HarnessError("Docker did not return an immutable image ID")
        return digest

    def agent_command(self) -> list[str]:
        adapter = codex if self.agent == "codex" else claude_code
        return adapter.command(self.cfg["query"]["model"])

    def docker_command(self, workspace: Path, name: str) -> list[str]:
        # A read-only root mount plus a nested writable output mount is a filesystem
        # boundary, unlike chmod or an agent's workspace-write sandbox alone.
        if "," in str(workspace):
            raise HarnessError("Docker bind paths must not contain commas")
        return ["docker", "run", "--rm", "--interactive", "--init", "--name", name,
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--user", "1000:1000", "--pids-limit", "256", "--memory", "4g", "--cpus", "2",
                "--tmpfs", "/tmp:rw,nosuid,size=512m,mode=1777",
                "--tmpfs", "/home/node:rw,nosuid,size=256m,uid=1000,gid=1000,mode=700",
                "--mount", f"type=bind,src={workspace},dst=/workspace,readonly",
                "--mount", f"type=bind,src={workspace / 'output'},dst=/workspace/output",
                "--workdir", "/workspace", "--env", self.destination_key,
                "--env", "DISABLE_AUTOUPDATER=1", "--env", "CLAUDE_CODE_DISABLE_AUTO_MEMORY=1",
                self.image, *self.agent_command()]

    def run(self, workspace: Path, prompt: str, stdout: Path, stderr: Path) -> AgentResult:
        name = f"vmr-{uuid.uuid4().hex}"
        command = self.docker_command(workspace, name)
        process = None
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out, stderr=err,
                                           env=self.env, start_new_session=True)
                try:
                    process.communicate(prompt.encode("utf-8"), timeout=self.cfg["query"]["timeout_sec"])
                    return AgentResult(process.returncode)
                except subprocess.TimeoutExpired:
                    return AgentResult(None, timed_out=True)
        finally:
            # Kill the container as well as its client, including on Ctrl-C.
            # This prevents descendants from writing output after timeout/cleanup.
            try:
                removed = subprocess.run(["docker", "rm", "--force", name],
                                         capture_output=True, text=True, timeout=30)
                if removed.returncode and "No such container" not in removed.stderr:
                    raise HarnessError(f"Could not confirm cleanup of container {name}")
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
