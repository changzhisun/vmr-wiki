from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import signal
from .experiment import Experiment
from .status import run_status
from vmr.core.progress import ProgressBar
from vmr.core.validation import positive_int
from vmr.core.errors import HarnessError

MAX_QUERY_JOBS = 16
_active_bar = None
_active_cancel = None


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


def run_queries(
    experiment: Experiment,
    jobs: int,
    bar: ProgressBar,
    *,
    cancel_event: threading.Event | None = None,
) -> tuple[int, int]:
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
                futures[
                    executor.submit(experiment.run, query, cancel_event=cancel_event)
                ] = query
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
            f"{shown}"
        ) from errors[0][1]
    return completed, failed


def run_batch(experiment, *, jobs=1, query_id=None):
    global _active_bar, _active_cancel
    validate_jobs(jobs)
    if query_id is not None:
        selected = [q for q in experiment.queries if q["query_id"] == query_id]
        if not selected:
            raise HarnessError(f"Unknown query: {query_id}")
        experiment.queries = selected
    cancel = threading.Event()
    _active_cancel = cancel
    old = signal.signal(signal.SIGINT, _handle_interrupt)
    bar = ProgressBar(
        len(experiment.queries),
        desc="Querying artifacts",
        unit="queries",
        log_stream="stdout",
    )
    _active_bar = bar
    try:
        bar.start()
        _, failed = run_queries(experiment, jobs, bar, cancel_event=cancel)
        return failed
    finally:
        bar.finish()
        signal.signal(signal.SIGINT, old)
        _active_bar = _active_cancel = None
