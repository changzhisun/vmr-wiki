import json

from vmr.runtime.trace import redact_trace_line


def test_trace_redaction_replaces_image_bytes_with_the_tool_path():
    payload = "A" * 400
    paths = {}
    call = {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "Read",
                    "input": {"file_path": "/scratch/sheet_00.jpg"},
                }
            ]
        },
    }
    call_line = (json.dumps(call) + "\n").encode()
    assert redact_trace_line(call_line, paths) == call_line
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": payload,
                            },
                        }
                    ],
                }
            ]
        },
        "tool_use_result": {
            "type": "image",
            "file": {"base64": payload, "type": "image/jpeg", "originalSize": 300},
        },
    }
    redacted = json.loads(redact_trace_line((json.dumps(event) + "\n").encode(), paths))
    source = redacted["message"]["content"][0]["content"][0]["source"]
    saved = redacted["tool_use_result"]["file"]
    assert source["data"] == "/scratch/sheet_00.jpg"
    assert source["media_type"] == "image/jpeg"
    assert saved["base64"] == "/scratch/sheet_00.jpg"
    assert saved["originalSize"] == 300
    assert payload not in json.dumps(redacted)


def test_trace_redaction_omits_image_bytes_when_the_path_is_unknown():
    payload = "A" * 400
    event = {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": payload,
                            },
                        }
                    ],
                }
            ]
        },
        "tool_use_result": {
            "type": "image",
            "file": {"base64": payload, "type": "image/jpeg", "originalSize": 300},
        },
    }
    line = (json.dumps(event) + "\n").encode()
    redacted = json.loads(redact_trace_line(line))
    source = redacted["message"]["content"][0]["content"][0]["source"]
    saved = redacted["tool_use_result"]["file"]
    assert payload not in json.dumps(redacted)
    assert source["data"] == f"<omitted image/jpeg, {len(payload)} base64 chars>"
    assert source["media_type"] == "image/jpeg"
    assert saved["base64"] == source["data"]
    assert saved["originalSize"] == 300
    assert redacted["message"]["content"][0]["tool_use_id"] == "call-1"
    assert len(redact_trace_line(line)) < len(line) // 2


def test_trace_redaction_leaves_events_without_images_unchanged():
    line = (
        b'{"type":"assistant","message":{"content":[{"type":"text","text":"seen"}]}}\n'
    )
    assert redact_trace_line(line) == line


def test_trace_redaction_replaces_data_urls_outside_json():
    line = b"see data:image/png;base64," + b"A" * 300 + b" done\n"
    redacted = redact_trace_line(line)
    assert b"data:image" not in redacted
    assert b"<omitted data-url>" in redacted
    assert redacted.endswith(b" done\n")
