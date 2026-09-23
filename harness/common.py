"""Compatibility exports; new code imports vmr.core directly."""
from vmr.core.errors import HarnessError, RunFailure
from vmr.core.validation import identifier, number, positive_int, nonempty, unique_index, cli
from vmr.core.jsonio import _object, parse_json, read_json, read_jsonl, atomic_text, write_json, write_jsonl
from vmr.core.hashing import canonical, object_hash, file_hash
from vmr.core.time import now
from vmr.core.errors import AGENT_FAILURE_KINDS, HARNESS_FAILURE_KINDS, FAILURE_KINDS

def ingest_content_hash(cfg):
    from vmr.compat.identity import compile_content_hash
    return compile_content_hash(cfg)

def ingest_content_diff(stored, cfg):
    from vmr.compat.identity import ingest_content_diff as diff
    return diff(stored, cfg)
