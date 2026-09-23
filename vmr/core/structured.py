import re

# Whole-response Markdown fence only. Prose around a fence, or a fence in the
# middle of other text, is still invalid: we unwrap the wrapper, not the JSON.
_MARKDOWN_JSON_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*?)[ \t]*\r?\n?```\Z",
    re.IGNORECASE | re.DOTALL,
)


def unwrap_markdown_json_fence(text: str) -> str:
    """Strip a surrounding ```json fence. Do not extract JSON from other prose."""
    stripped = text.strip()
    match = _MARKDOWN_JSON_FENCE.fullmatch(stripped)
    return match.group("body").strip() if match else stripped
