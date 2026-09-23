from pathlib import Path


def run_status(result: dict, root: Path) -> str:
    """Useful CLI diagnostics without opening large metadata files."""
    qid = result["query_id"]
    message = f"{qid}: {result['status']}"
    adjustments = result.get("output_adjustments") or []
    if adjustments:
        message += f" [adjusted] clipped {len(adjustments)} end_sec value(s) to the effective duration"
    if result["status"] != "success":
        reason = " ".join(
            str(result.get("error") or "No error detail recorded").split()
        )[:300]
        message += f" [{result.get('failure_kind') or 'unclassified'}] {reason}"
        message += f"\n  metadata: {root / 'run_metadata' / (qid + '.json')}"
        message += f"\n  logs: {root / 'logs' / (qid + '.stderr.log')}"
        message += f"\n        {root / 'logs' / (qid + '.stdout.log')}"
        relative_trace = result.get("trace_path", f"logs/{qid}.trace.jsonl")
        message += f"\n        {root / relative_trace}"
    return message
