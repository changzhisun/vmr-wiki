import io

import pytest

from harness.progress import ProgressBar
from harness.common import HarnessError
from harness.freeze import freeze_dataset


@pytest.fixture
def freeze_inputs(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr("harness.freeze.dataset_context", lambda *args: (tmp_path, {}, "train"))
    monkeypatch.setattr("harness.freeze.load_videos", lambda *args: {"first": {}, "second": {}})
    monkeypatch.setattr("harness.freeze.wiki_readiness", lambda *args: ([], []))
    monkeypatch.setattr("harness.freeze.read_json", lambda path: {
        "video_id": path.parent.name, "ingest_config": cfg["ingest"]})
    return cfg


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


@pytest.mark.parametrize("tty", [True, False])
def test_freeze_progress_reports_both_stages(freeze_inputs, monkeypatch, tty):
    stdout = TTYBuffer() if tty else io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    monkeypatch.setattr("harness.freeze.freeze_wiki", lambda path: {"wiki_hash": path.name})
    manifest = freeze_dataset(freeze_inputs)
    assert manifest["videos"] == {"first": "first", "second": "second"}
    text = stdout.getvalue()
    for desc in ("Checking freeze settings", "Freezing/verifying videos"):
        assert desc in text
    assert text.count("2/2 (100%)") >= 2
    assert "videos/s" in text and text.count("elapsed") == 2
    assert ("ETA" in text) == tty
    if not tty:
        assert len(text.splitlines()) == 2


def test_freeze_preflight_error_closes_bar_before_freezing(freeze_inputs, monkeypatch):
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)
    calls = iter([[], ["fixture mismatch"]])
    monkeypatch.setattr("harness.freeze.ingest_content_diff", lambda *args: next(calls))
    monkeypatch.setattr("harness.freeze.freeze_wiki", lambda *args: pytest.fail("freeze before preflight"))
    with pytest.raises(HarnessError, match="fixture mismatch"):
        freeze_dataset(freeze_inputs)
    text = stdout.getvalue()
    assert "1/2 (50%)" in text and text.endswith("\n")
    assert "Freezing/verifying" not in text


@pytest.mark.parametrize("error", [HarnessError, KeyboardInterrupt])
def test_freeze_error_or_interrupt_closes_partial_bar(freeze_inputs, monkeypatch, error):
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout)

    def freeze(path):
        if path.name == "second":
            raise error("fixture")
        return {"wiki_hash": path.name}

    monkeypatch.setattr("harness.freeze.freeze_wiki", freeze)
    with pytest.raises(error):
        freeze_dataset(freeze_inputs)
    final = stdout.getvalue().splitlines()[-1]
    assert "Freezing/verifying videos" in final
    assert "1/2 (50%)" in final and "elapsed" in final
