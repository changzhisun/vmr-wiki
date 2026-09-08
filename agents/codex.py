"""Only used inside an isolated container with read-only task/wiki mounts."""


def command(model: str) -> list[str]:
    return ["codex", "--ask-for-approval", "never", "exec", "--ephemeral",
            "--ignore-user-config", "--skip-git-repo-check", "--sandbox", "danger-full-access",
            "--model", model, "-"]
