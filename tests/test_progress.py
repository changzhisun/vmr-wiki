import io

import pytest

from harness.progress import ProgressBar


class TTYBuffer(io.StringIO):
    def isatty(self):
        return True


def test_progress_bar_clears_full_rendered_tty_line(monkeypatch):
    stdout = TTYBuffer()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    now = [0.0]
    bar = ProgressBar(100, desc="Ingesting videos", unit="videos", clock=lambda: now[0])

    now[0] = 10.0
    bar.update(10)
    rendered = (
        "Ingesting videos: [===---------------------------] 10/100 (10%)"
        " | 1.00 videos/s | ETA 00:01:30"
    )
    bar.log("done")
    assert "\r" + " " * len(rendered) + "\r" in stdout.getvalue()

    now[0] = 100.0
    bar.update(90)
    rendered = (
        "Ingesting videos: [==============================] 100/100 (100%)"
        " | 1.00 videos/s | ETA 00:00:00"
    )
    bar.log("finished")
    assert "\r" + " " * len(rendered) + "\r" in stdout.getvalue()


def test_progress_bar_non_tty_summary_includes_rate_and_elapsed(monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    now = [5.0]
    bar = ProgressBar(4, desc="Running queries", unit="queries", clock=lambda: now[0])
    now[0] = 7.0
    bar.update(4)
    bar.finish()
    assert stdout.getvalue() == (
        "Running queries: [==============================] 4/4 (100%)"
        " | 2.00 queries/s | elapsed 00:00:02\n"
    )


def test_progress_bar_uses_readable_units_for_slow_work(monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    now = [0.0]
    bar = ProgressBar(10, desc="Ingesting videos", unit="videos", clock=lambda: now[0])
    now[0] = 300.0
    bar.update()
    bar.finish()
    assert "0.20 videos/min" in stdout.getvalue()


def test_progress_bar_logs_to_requested_stream(monkeypatch):
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("sys.stderr", stderr)
    ProgressBar(1, unit="queries", log_stream="stdout").log("qid: success")
    assert stdout.getvalue() == "qid: success\n"
    assert stderr.getvalue() == ""


def test_progress_bar_rejects_unknown_log_stream():
    with pytest.raises(ValueError):
        ProgressBar(1, log_stream="syslog")


def test_progress_bar_finish_from_interrupted_render_does_not_deadlock(monkeypatch):
    """An interrupt handler runs on the thread it interrupts.

    If it calls finish() while that thread is mid-render and holding the
    lock, a non-reentrant lock would deadlock instead of closing the bar.
    """
    class InterruptingBuffer(TTYBuffer):
        def write(self, data):
            if not interrupted:
                interrupted.append(True)
                bar.finish()
            return super().write(data)

    interrupted = []
    stdout = InterruptingBuffer()
    monkeypatch.setattr("sys.stdout", stdout)
    now = [0.0]
    bar = ProgressBar(2, desc="Ingesting videos", unit="videos", clock=lambda: now[0])

    now[0] = 1.0
    bar.update()

    assert interrupted
    assert "elapsed 00:00:01" in stdout.getvalue()
