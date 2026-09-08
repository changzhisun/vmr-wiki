from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.common import cli
from harness.run_query import Experiment, arguments, configured


def main():
    args = arguments(batch=True)
    failed = 0
    with Experiment(configured(args), args.experiment) as experiment:
        for query in experiment.queries:
            result = experiment.run(query)
            print(f"{query['query_id']}: {result['status']}", flush=True)
            failed += result["status"] != "success"
    print(f"Finished: {failed} failed/unresolved runs")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli(main)
