from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import threading
from vmr.core.errors import HarnessError
from vmr.core.progress import ProgressBar
from vmr.core.validation import positive_int
from vmr.core.jsonio import write_json
from vmr.core.hashing import object_hash
from vmr.artifact.wikiset import write_wikiset
from vmr.datasets.manifest import load_dataset, select_split, load_videos
from .pipeline import compile_video


def compile_dataset(
    config,
    dataset,
    split,
    output_set,
    *,
    store=None,
    jobs=1,
    captioner=None,
    runtime=None,
):
    positive_int(jobs, "jobs")
    limit = positive_int(
        config.compile.consecutive_failure_limit, "consecutive_failure_limit"
    )
    dataset = Path(dataset)
    metadata = load_dataset(dataset)
    split = select_split(metadata, split)
    videos = load_videos(dataset, metadata, split)
    if not videos:
        raise HarnessError("No videos to compile")
    store = Path(store or config.storage.artifacts)
    rows = iter(videos.items())
    pending, artifacts, failures = {}, {}, []
    consecutive = 0
    cancel = threading.Event()
    bar = ProgressBar(
        len(videos), desc="Compiling videos", unit="videos", log_stream="stdout"
    )
    try:
        bar.start()
        with ThreadPoolExecutor(max_workers=jobs) as pool:

            def submit():
                item = next(rows, None)
                if item is not None:
                    vid, row = item
                    video = Path(row["video_path"])
                    if not video.is_absolute():
                        video = dataset / video
                    future = pool.submit(
                        compile_video,
                        video,
                        config.compile,
                        store,
                        templates=config.storage.templates,
                        captioner=captioner,
                        runtime=runtime,
                        cancel_event=cancel,
                    )
                    pending[future] = vid

            for _ in range(min(jobs, len(videos))):
                submit()
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        vid = pending.pop(future)
                        failure_path = (
                            store
                            / ".compile-failures"
                            / (object_hash([metadata["name"], split, vid]) + ".json")
                        )
                        try:
                            artifacts[vid] = future.result()
                            failure_path.unlink(missing_ok=True)
                            consecutive = 0
                            _record(bar, vid)
                        except (HarnessError, OSError) as exc:
                            if cancel.is_set():
                                raise
                            failures.append(vid)
                            consecutive += 1
                            write_json(
                                failure_path,
                                dict(schema_version=1, video_id=vid, error=str(exc)),
                            )
                            _record(bar, vid, error=exc)
                            from vmr.vlm.transport import FatalVLMError
                            from agents.runner import FatalAgentError

                            cause, visited, fatal = exc, set(), False
                            while cause is not None and id(cause) not in visited:
                                visited.add(id(cause))
                                fatal = fatal or isinstance(
                                    cause, (FatalVLMError, FatalAgentError)
                                )
                                cause = cause.__cause__
                            if fatal or consecutive >= limit:
                                raise HarnessError(
                                    f"Compile circuit stopped after {consecutive} failures: {exc}"
                                ) from exc
                        submit()
            except BaseException:
                cancel.set()
                for future in pending:
                    future.cancel()
                raise
        if failures:
            raise HarnessError(
                f"Compile failed for {len(failures)} videos; sealed successes can be reused"
            )
        return write_wikiset(
            output_set,
            dataset=metadata["name"],
            split=split,
            artifacts=artifacts,
            store=store,
        )
    finally:
        bar.finish()


def _record(bar, video_id, *, error=None):
    bar.update()
    if error is None:
        bar.log(f"{video_id}: sealed")
        return
    reason = " ".join(str(error).split())[:300]
    bar.log(f"{video_id}: failed: {reason}")
