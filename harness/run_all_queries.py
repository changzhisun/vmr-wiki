from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.common import HarnessError, cli
from harness.progress import ProgressBar
from harness.run_query import Experiment, arguments, configured


def main():
    args = arguments(batch=True)
    failed = completed = 0
    with Experiment(configured(args), args.experiment) as experiment:
        total = len(experiment.queries)
        # Per-query status stays on stdout, where it has always been.
        bar = ProgressBar(total, desc="Running queries", unit="queries", log_stream="stdout")
        bar.start()
        try:
            for query in experiment.queries:
                try:
                    result = experiment.run(query)
                except (HarnessError, OSError) as exc:
                    # A harness fault is not a measurement. Stop here rather than
                    # record the rest of the split as zeros; the runs that already
                    # finished are kept, and this query is retried after a fix.
                    raise HarnessError(
                        f"{query['query_id']}: harness failure after {completed} of {total} "
                        f"queries; the remaining queries were not scored.\n{exc}") from exc
                completed += 1
                failed += result["status"] != "success"
                bar.update()
                bar.log(f"{query['query_id']}: {result['status']}")
        finally:
            bar.finish()
    print(f"Finished: {failed} failed/unresolved runs")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli(main)
