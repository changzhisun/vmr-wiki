"""Explicitly load the shared instructions; Claude does not auto-load AGENTS.md."""


def command(model: str) -> list[str]:
    return ["claude", "--bare", "--print", "--no-session-persistence", "--model", model,
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands", "--tools", "Bash,Read,Write,Edit,Glob,Grep",
            "--permission-mode", "dontAsk", "--allowedTools", "Bash,Read,Write,Edit,Glob,Grep",
            "--append-system-prompt-file", "/workspace/AGENTS.md"]
