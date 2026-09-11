from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from harness.common import HarnessError, cli, positive_int
from harness.progress import ProgressBar
from harness.run_query import Experiment, arguments, configured, run_status


def _record(result: dict, experiment: Experiment, bar: ProgressBar) -> bool:
    bar.update()
    bar.log(run_status(result, experiment.root))
    return result["status"] != "success"


def run_queries(experiment: Experiment, jobs: int, bar: ProgressBar) -> tuple[int, int]:
    """Run distinct queries with at most ``jobs`` live Agent containers."""
    positive_int(jobs, "jobs")
    queries = experiment.queries
    total = len(queries)
    completed = failed = 0

    if jobs == 1:
        for query in queries:
            try:
                result = experiment.run(query)
            except (HarnessError, OSError) as exc:
                raise HarnessError(
                    f"{query['query_id']}: harness failure after {completed} of {total} "
                    f"queries; the remaining queries were not scored.\n{exc}") from exc
            completed += 1
            failed += _record(result, experiment, bar)
        return completed, failed

    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = {}
    next_index = 0
    harness_failure = None
    try:
        # Keep only one wave in flight so a harness fault stops new submissions
        # instead of queueing the rest of the split behind doomed work.
        while futures or (next_index < total and harness_failure is None):
            while harness_failure is None and next_index < total and len(futures) < jobs:
                query = queries[next_index]
                futures[executor.submit(experiment.run, query)] = query
                next_index += 1
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                query = futures.pop(future)
                try:
                    result = future.result()
                except (HarnessError, OSError) as exc:
                    if harness_failure is None:
                        harness_failure = (query, exc)
                    continue
                completed += 1
                failed += _record(result, experiment, bar)
    except BaseException:
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    if harness_failure is not None:
        query, exc = harness_failure
        not_started = total - next_index
        raise HarnessError(
            f"{query['query_id']}: harness failure after {completed} of {total} completed "
            f"queries; {not_started} queries were not started.\n{exc}") from exc
    return completed, failed


def main():
    args = arguments(batch=True)
    with Experiment(configured(args), args.experiment) as experiment:
        total = len(experiment.queries)
        # Per-query status stays on stdout, where it has always been.
        bar = ProgressBar(total, desc="Running queries", unit="queries", log_stream="stdout")
        bar.start()
        try:
            completed, failed = run_queries(experiment, args.jobs, bar)
        finally:
            bar.finish()
    print(f"Finished: {failed} failed/unresolved runs")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli(main)
