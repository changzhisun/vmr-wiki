"""Explicitly load the shared instructions; Claude does not auto-load AGENTS.md."""

# Bash, Read and file editing are enough to search the wiki and write one
# prediction. --tools is an allowlist, so a tool added by a future Claude Code
# release cannot reach the agent without changing this list.
TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

# --tools declares these two regardless of the allowlist.
ALWAYS_ON = ("Monitor", "PushNotification")


def command(model: str) -> list[str]:
    # --bare is deliberately not used: it pins the tools to Bash, Edit and Read
    # whatever --tools says, and the agent then cannot write its prediction. Its
    # other protections are covered here and by the container: empty
    # --setting-sources drops hooks and settings, --strict-mcp-config with no
    # servers drops MCP, CLAUDE_CODE_DISABLE_AUTO_MEMORY is set in the
    # environment, and the read-only workspace holds no CLAUDE.md to discover.
    return ["claude", "--print", "--no-session-persistence", "--model", model,
            "--setting-sources", "", "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
            "--disable-slash-commands", "--tools", TOOLS,
            "--permission-mode", "dontAsk", "--allowedTools", TOOLS,
            "--disallowedTools", *ALWAYS_ON,
            "--append-system-prompt-file", "/workspace/AGENTS.md"]
