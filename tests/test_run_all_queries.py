import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.runner import AgentCancelled
from harness.common import HarnessError
from harness.run_all_queries import MAX_QUERY_JOBS, run_queries


class Bar:
    def __init__(self):
        self.current = 0
        self.messages = []

    def update(self):
        self.current += 1

    def log(self, message):
        self.messages.append(message)

    def finish(self):
        pass


class ParallelExperiment:
    root = Path("results/test")

    def __init__(self, count=4):
        self.queries = [{"query_id": f"q{i}"} for i in range(count)]
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.active = self.max_active = 0

    def run(self, query, *, cancel_event):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active == 2:
                self.release.set()
        assert self.release.wait(2), "two workers were not active concurrently"
        with self.lock:
            self.active -= 1
        return {"query_id": query["query_id"], "status": "success"}


def test_parallel_query_workers_are_bounded_and_all_results_are_recorded():
    experiment = ParallelExperiment()
    bar = Bar()
    assert run_queries(experiment, 2, bar) == (4, 0)
    assert experiment.max_active == 2
    assert bar.current == 4 and len(bar.messages) == 4


def test_default_single_worker_path_is_exercised_and_ordered():
    order = []

    class SerialExperiment:
        root = Path("results/test")
        queries = [{"query_id": f"q{i}"} for i in range(3)]

        def run(self, query, *, cancel_event):
            assert not cancel_event.is_set()
            order.append(query["query_id"])
            return {"query_id": query["query_id"], "status": "success"}

    bar = Bar()
    assert run_queries(SerialExperiment(), 1, bar) == (3, 0)
    assert order == ["q0", "q1", "q2"]
    assert bar.current == 3


def test_parallel_harness_failure_stops_submitting_new_queries():
    started = []
    release = threading.Event()
    timer = threading.Timer(0.2, release.set)

    class Broken:
        root = Path("results/test")
        queries = [{"query_id": f"q{i}"} for i in range(5)]

        def run(self, query, *, cancel_event):
            started.append(query["query_id"])
            if query["query_id"] == "q0":
                raise HarnessError("broken invariant")
            assert release.wait(2)
            return {"query_id": query["query_id"], "status": "success"}

    timer.start()
    try:
        with pytest.raises(HarnessError, match="queries were not started"):
            run_queries(Broken(), 2, Bar())
    finally:
        timer.cancel()
    assert set(started) == {"q0", "q1"}


def test_parallel_drain_reports_all_started_harness_failures():
    barrier = threading.Barrier(2)

    class Broken:
        root = Path("results/test")
        queries = [{"query_id": f"q{i}"} for i in range(4)]

        def run(self, query, *, cancel_event):
            barrier.wait(timeout=2)
            raise HarnessError(f"broken {query['query_id']}")

    with pytest.raises(HarnessError, match=r"2 harness/interrupted.*2 started.*2 queries were not started") as exc:
        run_queries(Broken(), 2, Bar())
    assert "q0: broken q0" in str(exc.value)
    assert "q1: broken q1" in str(exc.value)


def test_jobs_must_be_positive():
    with pytest.raises(HarnessError, match="positive integer"):
        run_queries(ParallelExperiment(0), 0, Bar())
    with pytest.raises(HarnessError, match=f"must not exceed {MAX_QUERY_JOBS}"):
        run_queries(ParallelExperiment(0), MAX_QUERY_JOBS + 1, Bar())


def test_main_rejects_jobs_before_creating_experiment(monkeypatch):
    monkeypatch.setattr("harness.run_all_queries.arguments",
                        lambda **kwargs: SimpleNamespace(jobs=0))
    monkeypatch.setattr("harness.run_all_queries.Experiment", lambda *args, **kwargs:
                        pytest.fail("Experiment must not be constructed"))
    with pytest.raises(HarnessError, match="positive integer"):
        __import__("harness.run_all_queries", fromlist=["main"]).main()


def test_keyboard_interrupt_cancels_and_waits_for_running_workers(monkeypatch):
    started = threading.Event()
    stopped = threading.Event()

    class CancellableExperiment:
        root = Path("results/test")
        queries = [{"query_id": "q0"}]

        def run(self, query, *, cancel_event):
            started.set()
            assert cancel_event.wait(2), "worker did not receive cancellation"
            stopped.set()
            raise AgentCancelled("cancelled")

    def interrupt(*args, **kwargs):
        assert started.wait(2), "worker did not start"
        raise KeyboardInterrupt

    monkeypatch.setattr("harness.run_all_queries.wait", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_queries(CancellableExperiment(), 1, Bar())
    assert stopped.is_set(), "run_queries returned before the worker cleaned up"
