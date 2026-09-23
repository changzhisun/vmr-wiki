from copy import deepcopy
import pytest
from vmr.compiler.methods.bidirectional.edits import GraphState, apply_edit
from vmr.compiler.methods.bidirectional.model import NODE_EXAMPLE, normalize_node
from vmr.core.errors import HarnessError


def state():
    node = normalize_node({**deepcopy(NODE_EXAMPLE), "node_id": "n1"}, 3)
    return GraphState({"n1": node}, {}, (), 3, 100)


def test_reducer_returns_new_state_without_mutating_input():
    before = state()
    version = before.version
    after, logs = apply_edit(
        before,
        [
            dict(
                op="RELABEL",
                node_ids=["n1"],
                observation_ids=[],
                reason="clearer description",
                updates={"title": "Updated"},
            )
        ],
        expected_version=version,
        request_id="review/1",
    )
    assert before.version == version
    assert before.nodes["n1"]["title"] != after.nodes["n1"]["title"]
    assert after.nodes["n1"]["title"] == "Updated"
    assert logs[0]["before_version"] == version
    assert logs[0]["after_version"] == after.version


def test_invalid_edit_is_atomic_without_a_database():
    before = state()
    snapshot = deepcopy(before)
    with pytest.raises(HarnessError):
        apply_edit(
            before,
            [
                dict(
                    op="RELABEL",
                    node_ids=["n1"],
                    reason="first",
                    updates={"title": "Changed"},
                ),
                dict(op="REPARENT", node_ids=["n1"], reason="cycle", parent_id="n1"),
            ],
            expected_version=before.version,
            request_id="review/1",
        )
    assert before == snapshot
