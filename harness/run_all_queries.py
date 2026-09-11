from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import signal
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from harness.common import HarnessError, cli, positive_int
from harness.progress import ProgressBar
from harness.run_query import Experiment, arguments, configured, run_status


MAX_QUERY_JOBS = 16
_active_bar: "ProgressBar | None" = None
_active_cancel: "threading.Event | None" = None


def _handle_interrupt(signum: int, frame: object) -> None:
    if _active_cancel is not None:
        _active_cancel.set()
    if _active_bar is not None:
        _active_bar.finish()
    raise KeyboardInterrupt


def validate_jobs(jobs: int) -> int:
    positive_int(jobs, "jobs")
    if jobs > MAX_QUERY_JOBS:
        raise HarnessError(f"jobs must not exceed {MAX_QUERY_JOBS}")
    return jobs


def _record(result: dict, experiment: Experiment, bar: ProgressBar) -> bool:
    bar.update()
    bar.log(run_status(result, experiment.root))
    return result["status"] != "success"


def run_queries(experiment: Experiment, jobs: int, bar: ProgressBar, *,
                cancel_event: threading.Event | None = None) -> tuple[int, int]:
    """Run distinct queries with at most ``jobs`` live Agent containers."""
    jobs = validate_jobs(jobs)
    cancel_event = cancel_event or threading.Event()
    queries = experiment.queries
    total = len(queries)
    completed = failed = 0
    executor = ThreadPoolExecutor(max_workers=jobs)
    futures = {}
    next_index = 0
    errors: list[tuple[dict, BaseException]] = []
    shut_down = False
    try:
        # Keep only one wave in flight so a harness fault stops new submissions
        # instead of queueing the rest of the split behind doomed work.
        while futures or (next_index < total and not errors):
            while not errors and next_index < total and len(futures) < jobs:
                query = queries[next_index]
                futures[executor.submit(experiment.run, query,
                                        cancel_event=cancel_event)] = query
                next_index += 1
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                query = futures.pop(future)
                try:
                    result = future.result()
                except (HarnessError, OSError) as exc:
                    errors.append((query, exc))
                    cancel_event.set()
                    continue
                completed += 1
                failed += _record(result, experiment, bar)
    except KeyboardInterrupt:
        cancel_event.set()
        # Once cooperative cleanup starts, ignore a second SIGINT so it cannot
        # abandon live containers, networks, or the experiment lock.
        restore_handler = None
        try:
            current_handler = signal.getsignal(signal.SIGINT)
            if current_handler is not signal.SIG_IGN:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                if current_handler is not _handle_interrupt:
                    restore_handler = current_handler
        except (ValueError, OSError):
            pass
        try:
            executor.shutdown(wait=True, cancel_futures=True)
            shut_down = True
        finally:
            if restore_handler is not None:
                signal.signal(signal.SIGINT, restore_handler)
        raise
    except BaseException:
        cancel_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
        shut_down = True
        raise
    finally:
        if not shut_down:
            executor.shutdown(wait=True)

    if errors:
        not_started = total - next_index
        shown = "; ".join(f"{query['query_id']}: {exc}" for query, exc in errors[:4])
        if len(errors) > 4:
            shown += f"; ... (+{len(errors) - 4})"
        raise HarnessError(
            f"{len(errors)} harness/interrupted query failure(s); {completed} of {total} "
            f"queries produced a result, {next_index} started, and {not_started} queries were not started.\n"
            f"{shown}") from errors[0][1]
    return completed, failed


def main():
    args = arguments(batch=True)
    jobs = validate_jobs(args.jobs)  # Validate before creating an experiment directory.
    cancel_event = threading.Event()
    global _active_bar, _active_cancel
    _active_cancel = cancel_event
    old_handler = None
    try:
        try:
            old_handler = signal.signal(signal.SIGINT, _handle_interrupt)
        except (ValueError, OSError):
            pass
        with Experiment(configured(args), args.experiment) as experiment:
            total = len(experiment.queries)
            # Per-query status stays on stdout, where it has always been.
            _active_bar = ProgressBar(total, desc="Running queries", unit="queries",
                                      log_stream="stdout")
            _active_bar.start()
            try:
                completed, failed = run_queries(
                    experiment, jobs, _active_bar, cancel_event=cancel_event)
            finally:
                _active_bar.finish()
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
        _active_bar = None
        _active_cancel = None
    print(f"Finished: {failed} failed/unresolved runs")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    cli(main)
