import json
from pathlib import Path

import pytest

from harness.common import HarnessError, ingest_content_hash, read_json, read_jsonl, write_json
from harness.freeze import freeze_wiki, verify_wiki
from harness.ingest import ingest_video


@pytest.fixture
def recovery(cfg, tmp_path, monkeypatch):
    cfg["ingest"].update(caption_mode="dense", caption_window_frames=2,
                         caption_stride_frames=1, caption_max_repairs=0)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"source")
    extracted = []

    def extract(_video, timestamp, path, _cfg):
        extracted.append(timestamp)
        path.write_bytes(b"jpeg")

    monkeypatch.setattr("harness.ingest.extract_frame", extract)
    monkeypatch.setattr("harness.ingest.probe_durations", lambda _: (3.0, 3.0))
    monkeypatch.setattr("harness.ingest.media_command", lambda _: "ffmpeg fixture")
    return cfg, video, tmp_path / "wiki" / "videos" / "clip", extracted


class Captioner:
    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def caption(self, images, *, timestamps, target_timestamps=None, correction=None):
        self.calls.append(timestamps)
        if timestamps[0] == self.fail_at:
            raise HarnessError("temporary server failure")
        return json.dumps({"events": [{"start": timestamps[0], "end": timestamps[-1],
                                      "kind": "action", "caption": "A person walks."}]})


def test_resume_reuses_frames_and_completed_windows_and_preserves_audit(recovery):
    cfg, video, output, extracted = recovery
    with pytest.raises(HarnessError, match="server failure"):
        ingest_video(video, "clip", output, cfg, captioner=Captioner(1.0))
    assert not output.exists()
    assert extracted == [0.0, 1.0, 2.0]
    resumed = Captioner()
    metadata = ingest_video(video, "clip", output, cfg, captioner=resumed)
    assert resumed.calls == [[1.0, 2.0]]
    assert extracted == [0.0, 1.0, 2.0]
    assert metadata["telemetry"]["reused_windows"] == 1
    assert metadata["telemetry"]["reused_frames"] == 3
    assert metadata["telemetry"]["caption_attempts"] == 3
    assert metadata["telemetry"]["rejected_attempts"] == 1
    audit = read_jsonl(output / "caption_audit.jsonl")
    assert len(audit[1]["attempts"]) == 2
    assert audit[1]["attempts"][1]["normalized_events"][0]["start"] == 1.0
    assert "raw_response" in audit[0]["attempts"][0]
    seal = freeze_wiki(output)
    assert "caption_audit.jsonl" in seal["files"]
    assert verify_wiki(output) == seal


@pytest.mark.parametrize("change", ["source", "config"])
def test_checkpoint_identity_prevents_stale_reuse(recovery, change):
    cfg, video, output, extracted = recovery
    with pytest.raises(HarnessError):
        ingest_video(video, "clip", output, cfg, captioner=Captioner(1.0))
    if change == "source":
        video.write_bytes(b"different source")
    else:
        cfg["ingest"]["vlm"]["prompt"] += " Extra instruction."
    resumed = Captioner()
    ingest_video(video, "clip", output, cfg, captioner=resumed)
    assert len(resumed.calls) == 2
    assert len(extracted) == 6


def test_corrupt_checkpoint_is_rejected(recovery):
    cfg, video, output, _ = recovery
    with pytest.raises(HarnessError):
        ingest_video(video, "clip", output, cfg, captioner=Captioner(1.0))
    image = next((output.parent.parent / ".ingest-checkpoints").rglob("000001.jpg"))
    image.write_bytes(b"corrupt")
    with pytest.raises(HarnessError, match="Checkpoint frame changed"):
        ingest_video(video, "clip", output, cfg, captioner=Captioner())
    assert not output.exists()


@pytest.mark.parametrize("corruption", [[], {"data": {"frames": []}, "sha256": "wrong"}])
def test_corrupt_window_checkpoint_is_rejected(recovery, corruption):
    cfg, video, output, _ = recovery
    with pytest.raises(HarnessError):
        ingest_video(video, "clip", output, cfg, captioner=Captioner(1.0))
    window = next((output.parent.parent / ".ingest-checkpoints").rglob("w000001.json"))
    write_json(window, corruption)
    with pytest.raises(HarnessError, match="Checkpoint changed"):
        ingest_video(video, "clip", output, cfg, captioner=Captioner())


def test_checkpoint_cleanup_failure_does_not_fail_published_wiki(recovery, monkeypatch):
    cfg, video, output, _ = recovery

    def fail_cleanup(path):
        raise PermissionError("fixture")

    monkeypatch.setattr("harness.ingest.remove_tree", fail_cleanup)
    metadata = ingest_video(video, "clip", output, cfg, captioner=Captioner())
    assert read_json(output / "ingest.json") == metadata


@pytest.mark.parametrize("mode", ["simple", "dense"])
@pytest.mark.parametrize("error", [KeyboardInterrupt, ConnectionResetError])
def test_resume_after_interrupted_or_unexpected_caption_error(recovery, mode, error):
    cfg, video, output, extracted = recovery
    cfg["ingest"]["caption_mode"] = mode

    class InterruptedCaptioner:
        def caption(self, *args, **kwargs):
            raise error()

    class ResumedCaptioner:
        def caption(self, *args, **kwargs):
            return '{"events":[]}' if mode == "dense" else "A scene."

    with pytest.raises(error):
        ingest_video(video, "clip", output, cfg, captioner=InterruptedCaptioner())
    assert not output.exists()
    metadata = ingest_video(video, "clip", output, cfg, captioner=ResumedCaptioner())
    assert metadata["telemetry"]["rejected_attempts"] == 1
    assert metadata["telemetry"]["caption_attempts"] == 3
    assert metadata["telemetry"]["reused_frames"] == 3
    assert extracted == [0.0, 1.0, 2.0]
    attempt = read_jsonl(output / "caption_audit.jsonl")[0]["attempts"][0]
    assert attempt["status"] == "failed" and attempt["error"] == error.__name__


def test_processing_rules_are_part_of_dense_identity(cfg):
    cfg["ingest"]["caption_mode"] = "dense"
    current = ingest_content_hash(cfg)
    legacy = dict(cfg["ingest"])
    legacy.pop("caption_processing_version")
    legacy.pop("dense_timestamp_mode")
    assert ingest_content_hash({"ingest": legacy}) != current
    cfg["ingest"]["dense_timestamp_mode"] = "frame_index"
    assert ingest_content_hash(cfg) != current


def test_frame_index_mode_records_conversion(recovery):
    cfg, video, output, _ = recovery
    cfg["ingest"]["dense_timestamp_mode"] = "frame_index"
    captioner = Captioner()
    ingest_video(video, "clip", output, cfg, captioner=captioner)
    assert captioner.calls == [[0, 1], [0, 1]]
    rows = read_jsonl(output / "frames.jsonl")
    assert rows[1]["events"][0]["start"] == 1.0
    assert rows[1]["events"][0]["end"] == 2.0
