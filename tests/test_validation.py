import copy

import pytest

from harness.common import HarnessError, identifier, parse_json
from harness.validate import validate_prediction


def prediction():
    return {"query_id": "q1", "video_id": "v1", "moments": [
        {"start_sec": 0.0, "end_sec": 1.0, "score": 0.9, "evidence": "first"},
        {"start_sec": 2.0, "end_sec": 3.0, "score": 0.8, "evidence": "second"},
    ]}


def test_valid_multi_and_abstention():
    assert len(validate_prediction(prediction())["moments"]) == 2
    empty = prediction()
    empty["moments"] = []
    assert validate_prediction(empty)["moments"] == []


@pytest.mark.parametrize("field,value", [
    ("start_sec", -1), ("start_sec", True), ("start_sec", "0"),
    ("end_sec", 0), ("end_sec", float("inf")), ("score", float("nan")),
    ("score", -0.1), ("score", 1.01), ("evidence", None),
])
def test_invalid_moment(field, value):
    row = prediction()
    row["moments"][0][field] = value
    with pytest.raises(HarnessError):
        validate_prediction(row)


def test_ids_count_duration_order_and_missing_fields():
    for kwargs in ({"query_id": "q2"}, {"video_id": "v2"},
                   {"max_predictions": 1}, {"duration": 2}):
        with pytest.raises(HarnessError):
            validate_prediction(prediction(), **kwargs)
    row = prediction()
    row["moments"].reverse()
    with pytest.raises(HarnessError):
        validate_prediction(row)
    row = prediction()
    del row["moments"][0]["evidence"]
    with pytest.raises(HarnessError):
        validate_prediction(row)


@pytest.mark.parametrize("text", ['{"a": NaN}', '{"a": Infinity}', '{"a":1,"a":2}', '```json\n{}\n```'])
def test_strict_json(text):
    with pytest.raises(HarnessError):
        parse_json(text)


@pytest.mark.parametrize("value", ["../x", "/absolute", "a/b", "", "..", "a\nb", None])
def test_unsafe_identifiers(value):
    with pytest.raises(HarnessError):
        identifier(value)

