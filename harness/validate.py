from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

from harness.common import (HarnessError, cli, identifier, nonempty, number,
                            positive_int, read_json)


def validate_moment(moment: dict, duration: float | None = None) -> None:
    if not isinstance(moment, dict):
        raise HarnessError("Each moment must be an object")
    start = number(moment.get("start_sec"), "start_sec", 0)
    end = number(moment.get("end_sec"), "end_sec", 0)
    if end <= start:
        raise HarnessError("end_sec must exceed start_sec")
    if duration is not None and end > duration:
        raise HarnessError(f"end_sec exceeds video duration ({duration})")


def validate_query(query: dict) -> None:
    if "split" not in query:
        raise HarnessError("Query has no split. Re-run the Dataset Adapter.")
    if set(query) != {"query_id", "video_id", "query", "split"}:
        raise HarnessError("Query must contain only query_id, video_id, query, split")
    identifier(query["query_id"], "query_id")
    identifier(query["video_id"], "video_id")
    nonempty(query["query"], "query")
    nonempty(query["split"], "split")


def validate_prediction(prediction: dict, *, query_id: str | None = None,
                        video_id: str | None = None, max_predictions: int = 5,
                        duration: float | None = None, split: str | None = None) -> dict:
    positive_int(max_predictions, "max_predictions")
    required = {"query_id", "video_id", "split", "moments"}
    if not isinstance(prediction, dict) or not required <= prediction.keys() or prediction.keys() - required - {"evidence"}:
        raise HarnessError("Prediction requires query_id, video_id, split, moments; evidence is optional")
    nonempty(prediction["split"], "split")
    if split is not None and prediction["split"] != split:
        raise HarnessError(f"Mismatched split: expected {split!r}, got {prediction['split']!r}")
    if "evidence" in prediction and not isinstance(prediction["evidence"], str):
        raise HarnessError("evidence must be a string")
    for field, expected in (("query_id", query_id), ("video_id", video_id)):
        identifier(prediction[field], field)
        if expected is not None and prediction[field] != expected:
            raise HarnessError(f"Mismatched {field}: expected {expected}")
    moments = prediction["moments"]
    if not isinstance(moments, list) or len(moments) > max_predictions:
        raise HarnessError(f"moments must be an array of at most {max_predictions} entries")
    previous_score = 1.0
    for moment in moments:
        validate_moment(moment, duration)
        required_moment = {"start_sec", "end_sec", "score"}
        if not required_moment <= moment.keys() or moment.keys() - required_moment - {"evidence"}:
            raise HarnessError("Moment requires start_sec, end_sec, score; evidence is optional")
        score = number(moment["score"], "score", 0)
        if score > 1 or score > previous_score:
            raise HarnessError("Scores must be in [0, 1] and sorted descending")
        previous_score = score
        if "evidence" in moment and not isinstance(moment["evidence"], str):
            raise HarnessError("evidence must be a string")
    return prediction


def main():
    parser = argparse.ArgumentParser(description="Validate an agent prediction without repairing it")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--query-id")
    parser.add_argument("--video-id")
    parser.add_argument("--split", help="Expected exact split string (no alias mapping)")
    parser.add_argument("--max-predictions", type=int, default=5)
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    validate_prediction(read_json(args.prediction), query_id=args.query_id,
                        video_id=args.video_id, max_predictions=args.max_predictions,
                        duration=args.duration, split=args.split)
    print("valid")


if __name__ == "__main__":
    cli(main)
