END_CLAMP_TOLERANCE_SEC = 0.05
FLOAT_NOISE_TOLERANCE_SEC = 1e-9


def clamp_prediction_ends(prediction: dict, *, duration: float) -> list[dict]:
    """Clip only small endpoint overruns caused by duration precision differences."""
    adjustments = []
    for index, moment in enumerate(prediction["moments"]):
        end = moment["end_sec"]
        excess = end - duration
        if 0 < excess <= FLOAT_NOISE_TOLERANCE_SEC:
            moment["end_sec"] = duration
        elif FLOAT_NOISE_TOLERANCE_SEC < excess <= END_CLAMP_TOLERANCE_SEC:
            moment["end_sec"] = duration
            adjustments.append(
                {
                    "kind": "clamp_end_to_effective_duration",
                    "moment_index": index,
                    "original_end_sec": end,
                    "adjusted_end_sec": duration,
                    "delta_sec": round(excess, 9),
                    "reason": "end_sec exceeded the authoritative prediction duration",
                }
            )
    return adjustments
