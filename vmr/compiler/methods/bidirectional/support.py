"""Bidirectional temporal parsing using one VLM and deterministic time logic."""

from __future__ import annotations

from copy import deepcopy
import json
import logging

from vmr.compiler.methods.bidirectional.config import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    budgets,
    settings,
)
from vmr.compiler.methods.bidirectional.io import (
    FrameIndex,
    RequestJournal,
    batches,
    stream_jsonl,
    windows,
)
from vmr.core.errors import HarnessError
from vmr.core.hashing import canonical, file_hash, object_hash
from vmr.core.time import now
from vmr.core.jsonio import write_json
from vmr.artifact.integrity import remove_tree, tree_hashes
from vmr.compiler.methods.bidirectional.graph import (
    NODE_EXAMPLE,
    OPERATIONS,
    SEMANTICS,
    TemporalGraph,
    cite_ids,
    clip_to_range,
    combine_evidence,
    normalize_node,
    strings,
)

LOG = logging.getLogger(__name__)
NODE_SCHEMA = json.dumps(NODE_EXAMPLE)
EDIT_INSTRUCTION = (
    """Compare sources without assuming top-down is correct. Preserve supported short moments.
Return JSON with "operations" and "conflicts" arrays (either may be empty). Extra keys are ignored.
Visual review must also include "refuted_observation_ids" (array, possibly empty) and "resolved" (boolean).
Allowed operations only: KEEP, INSERT, DELETE, SPLIT, MERGE, SHIFT, RELABEL, REPARENT.
Every operation has op, node_ids (list of existing IDs; [] if none), observation_ids (list of supplied evidence IDs), reason.
Cite only IDs present in this INPUT records list. Temporary INSERT refs may use "$name".
KEEP associates observations with one or more existing nodes.
INSERT has new_node with all node-schema fields and parent_id (existing ID or null).
SPLIT has node_ids with exactly one existing parent ID and new_nodes with at least two children;
keep the original as their broader parent.
MERGE has peer node_ids with equal granularity/parent and new_node with combined semantics;
its time range is the union envelope, inherited children and original evidence are preserved.
SHIFT has updates containing start, end, boundary_uncertainty; requires visual boundary review.
RELABEL has updates containing only semantic fields (including type, retrieval_text, confidence).
REPARENT has node_ids with exactly one ID, parent_id, and optional relations; keep child granularity above parent.
INSERT may assign a temporary ref such as "$parent" for subsequent operations in this batch.
DELETE always needs new visual evidence. Never delete because the other pass omitted a moment.
For visual contradictions (e.g. salt vs sugar) use conflicts with node_ids, observation_ids, reason;
do not guess or silently settle them from text. A conflict must cite at least one supplied reference.
Overlaps and gaps are legal when semantically meaningful. Do not force a time partition or fixed ontology.
Unknown facts stay unknown; no audio, external lookup, embeddings, detector, or other model is available.
boundary_uncertainty.start/end are [lo, hi] in seconds and must satisfy 0 <= lo <= that endpoint <= hi <= video duration.
Never copy example timestamps; they are shape only.
Node schema: """
    + NODE_SCHEMA
)


def is_unusable_response(exc):
    return isinstance(exc, HarnessError) and "response repair budget exhausted" in str(
        exc
    )


def view(node):
    # Evidence/history remain in the disk index. Text comparison receives the
    # semantic record and stable references, not unbounded frame/history arrays.
    keys = SEMANTICS | {
        "node_id",
        "observation_id",
        "parent_id",
        "granularity",
        "start",
        "end",
        "boundary_uncertainty",
        "review_status",
        "issues",
        "group_id",
    }
    return {key: value for key, value in node.items() if key in keys}
