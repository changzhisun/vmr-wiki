from harness.common import HarnessError


def evaluate_predictions(predictions, ground_truth, *, official_root=None):
    if official_root is not None:
        raise HarnessError("The generic evaluator has no official-root implementation")
    return {"implementation": "generic", "gt_semantics": "any_acceptable_moment"}
