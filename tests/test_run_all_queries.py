import threading
from pathlib import Path

import pytest

from harness.common import HarnessError
from harness.run_all_queries import run_queries


class Bar:
    def __init__(self):
        self.current = 0
        self.messages = []

    def update(self):
        self.current += 1

    def log(self, message):
        self.messages.append(message)


class ParallelExperiment:
    root = Path("results/test")

    def __init__(self, count=4):
        self.queries = [{"query_id": f"q{i}"} for i in range(count)]
        self.lock = threading.Lock()
        self.release = threading.Event()
        self.active = self.max_active = 0

    def run(self, query):
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


def test_parallel_harness_failure_stops_submitting_new_queries():
    started = []
    release = threading.Event()
    timer = threading.Timer(0.2, release.set)

    class Broken:
        root = Path("results/test")
        queries = [{"query_id": f"q{i}"} for i in range(5)]

        def run(self, query):
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


def test_jobs_must_be_positive():
    with pytest.raises(HarnessError, match="positive integer"):
        run_queries(ParallelExperiment(0), 0, Bar())
