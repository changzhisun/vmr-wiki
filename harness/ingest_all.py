from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import signal
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness.common import HarnessError, cli, identifier
from harness.config import dataset_path, load_config
from harness.dataset import dataset_context, load_videos
from harness.freeze import freeze_dataset
from harness.ingest import ingest_video

logger = logging.getLogger(__name__)

# Module-level handle so an interrupt can close the progress bar cleanly.
_active_bar: "ProgressBar | None" = None


def _handle_interrupt(signum: int, frame: object) -> None:
    if _active_bar is not None:
        _active_bar.finish()
    raise KeyboardInterrupt


class ProgressBar:
    """Thread-safe, dependency-free progress bar rendered on stdout.

    On a TTY it draws an updating bar; log lines emitted through ``log()``
    are written above the bar without corrupting it. On a non-TTY stdout it
    stays silent during the run and prints a single summary line at the end.
    """

    def __init__(self, total: int, desc: str = "Ingesting") -> None:
        self.total = total
        self.desc = desc
        self.current = 0
        self._lock = threading.Lock()
        self._width = 30
        self._finished = False
        self._is_tty = sys.stdout.isatty()
        # Width wide enough to fully overwrite the longest possible bar line.
        self._line_width = len(desc) + self._width + len(f"{total}/{total}") + 10

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.current += n
            if self._is_tty and not self._finished:
                self._render_unlocked()

    def log(self, message: str) -> None:
        """Write a log line above the bar without corrupting either."""
        with self._lock:
            if self._is_tty and not self._finished:
                self._clear_unlocked()
                sys.stderr.write(message + "\n")
                sys.stderr.flush()
                self._render_unlocked()
            else:
                sys.stderr.write(message + "\n")
                sys.stderr.flush()

    def _render_unlocked(self) -> None:
        fraction = min(self.current / self.total, 1.0)
        filled = int(self._width * fraction)
        bar = "=" * filled + "-" * (self._width - filled)
        percent = int(100 * fraction)
        line = f"{self.desc}: [{bar}] {self.current}/{self.total} ({percent}%)"
        sys.stdout.write("\r" + line.ljust(self._line_width) + "\r")
        sys.stdout.flush()

    def _clear_unlocked(self) -> None:
        sys.stdout.write("\r" + " " * self._line_width + "\r")
        sys.stdout.flush()

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            percent = int(100 * min(self.current / self.total, 1.0))
            if self._is_tty:
                self._clear_unlocked()
            sys.stdout.write(f"{self.desc}: {self.current}/{self.total} ({percent}%)\n")
            sys.stdout.flush()


class _BarAwareHandler(logging.Handler):
    """Logging handler that writes log lines above the progress bar."""

    def __init__(self, bar: ProgressBar) -> None:
        super().__init__()
        self._bar = bar
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bar.log(self.format(record))
        except Exception:
            self.handleError(record)


def _setup_logging(bar: ProgressBar, verbose: bool = False) -> None:
    handler = _BarAwareHandler(bar)
    level = logging.DEBUG if verbose else logging.INFO
    handler.setLevel(level)
    logging.basicConfig(level=level, handlers=[handler], force=True)


def _ingest_one(
    vid: str,
    video: dict,
    video_path: Path,
    output: Path,
    cfg: dict,
    captioner=None,
) -> tuple[dict, bool]:
    """Ingest a single video.

    Returns ``(metadata, was_existing)``. Raises ``HarnessError`` on failure;
    any partial output is cleaned up inside ``ingest_video``.
    """
    was_existing = output.exists()
    if was_existing:
        logger.info("%s: verifying existing ingest", vid)
    else:
        logger.info("%s: ingesting new video (%.1fs)", vid, video["duration"])

    metadata = ingest_video(video_path, vid, output, cfg, captioner=captioner)

    if abs(metadata["duration"] - video["duration"]) > 0.1:
        raise HarnessError(
            f"{vid}: annotation/media durations differ by more than 0.1s "
            f"({metadata['duration']}s vs {video['duration']}s)"
        )

    if was_existing:
        logger.info("%s: verified OK (%.1fs)", vid, metadata["duration"])
    else:
        logger.info("%s: ingest OK (%.1fs)", vid, metadata["duration"])
    return metadata, was_existing


def _run_sequential(
    tasks: list[tuple[str, dict, Path, Path]],
    cfg: dict,
    captioner,
    bar: ProgressBar,
) -> list[dict]:
    results: list[dict] = []
    for vid, video, path, output in tasks:
        metadata, _ = _ingest_one(vid, video, path, output, cfg, captioner=captioner)
        results.append(metadata)
        bar.update()
    return results


def _run_parallel(
    tasks: list[tuple[str, dict, Path, Path]],
    cfg: dict,
    captioner,
    jobs: int,
    bar: ProgressBar,
) -> list[dict]:
    results: list[dict] = [None] * len(tasks)  # type: ignore[list-item]
    failures: list[tuple[str, str]] = []
    recaptioned = 0
    verified = 0

    executor = ThreadPoolExecutor(max_workers=jobs)
    shut_down = False
    try:
        # Results are written back by index so the return order matches the input
        # order (same contract as the sequential path), regardless of completion order.
        future_to_idx = {
            executor.submit(_ingest_one, vid, video, path, output, cfg, captioner=captioner): idx
            for idx, (vid, video, path, output) in enumerate(tasks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            vid = tasks[idx][0]
            try:
                metadata, was_existing = future.result()
                results[idx] = metadata
                if was_existing:
                    verified += 1
                else:
                    recaptioned += 1
            except HarnessError as exc:
                failures.append((vid, str(exc)))
                logger.error("%s: FAILED - %s", vid, exc)
            bar.update()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user; cancelling remaining work")
        executor.shutdown(wait=False, cancel_futures=True)
        shut_down = True
        raise
    finally:
        if not shut_down:
            executor.shutdown(wait=True)

    logger.info(
        "Done: %d recaptioned, %d verified, %d failed of %d",
        recaptioned,
        verified,
        len(failures),
        len(tasks),
    )
    if failures:
        _report_failures(failures, len(tasks))
        raise HarnessError(f"{len(failures)} of {len(tasks)} video(s) failed to ingest")

    logger.info("All %d videos ingested/verified successfully", len(tasks))
    return results


def _report_failures(failures: list[tuple[str, str]], total: int) -> None:
    by_error: dict[str, list[str]] = {}
    for vid, error in failures:
        by_error.setdefault(error, []).append(vid)
    for error, vids in by_error.items():
        shown = ", ".join(vids[:8])
        if len(vids) > 8:
            shown += f" ... (+{len(vids) - 8})"
        logger.error("[%d/%d] %s: %s", len(vids), total, shown, error)
    if len(by_error) == 1 and len(failures) == total:
        logger.error(
            "All videos failed identically: likely a shared configuration or "
            "environment issue (API key, VLM base URL, ffmpeg, or dataset path)."
        )


def ingest_all(
    cfg: dict,
    *,
    split: str | None = None,
    captioner=None,
    jobs: int = 4,
    verbose: bool = False,
) -> list[dict]:
    """Ingest every video in the split into the visual wiki.

    ``jobs`` controls how many videos are ingested concurrently. With more than
    one worker, per-video failures are collected and reported in a final
    summary instead of aborting the whole run; a single ``HarnessError`` is
    raised at the end if any video failed. With ``jobs=1``, the original
    sequential, fail-fast behavior is preserved.

    The optional ``captioner`` is shared across worker threads and therefore
    must be thread-safe.
    """
    if jobs < 1:
        raise HarnessError("--jobs must be at least 1")

    dataset, metadata, split = dataset_context(cfg, split)
    videos = load_videos(dataset, metadata, split)
    if not videos:
        raise HarnessError("Dataset contains no videos")

    wiki_root = dataset_path(cfg, "wiki") / "videos"
    tasks: list[tuple[str, dict, Path, Path]] = []
    for vid, video in videos.items():
        path = Path(video["video_path"])
        if not path.is_absolute():
            path = dataset / path
        tasks.append((vid, video, path, wiki_root / vid))

    global _active_bar
    _active_bar = ProgressBar(len(tasks), desc="Ingesting videos")
    _setup_logging(_active_bar, verbose)

    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGINT, _handle_interrupt)
    except (ValueError, OSError):
        pass  # Signals can only be registered from the main thread.

    logger.info(
        "Starting ingest of %d video(s) with %d worker(s) (split=%r)", len(tasks), jobs, split
    )

    try:
        if jobs == 1:
            results = _run_sequential(tasks, cfg, captioner, _active_bar)
        else:
            results = _run_parallel(tasks, cfg, captioner, jobs, _active_bar)
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)
        _active_bar.finish()
        _active_bar = None

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all videos using fixed caption settings")
    parser.add_argument("--dataset")
    parser.add_argument("--split", help="Exact dataset.json split name (or dataset.split in config)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Concurrent videos to ingest (default: 4). Use 1 for the original sequential, fail-fast behavior.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset:
        cfg["dataset"]["name"] = identifier(args.dataset)

    results = ingest_all(cfg, split=args.split, jobs=args.jobs, verbose=args.verbose)

    if args.freeze:
        freeze_dataset(cfg, split=args.split)
    print(f"Ingested/verified {len(results)} videos")


if __name__ == "__main__":
    cli(main)
