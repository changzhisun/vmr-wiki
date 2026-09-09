from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from adapters.qvhighlights import QVHighlightsAdapter
from harness.common import write_json, write_jsonl
from harness.config import load_config
from harness.freeze import freeze_dataset, remove_tree
from harness.ingest_all import ingest_all


ROOT = Path(__file__).resolve().parents[1]


def write_evaluation_dataset(directory, truth, *, name="example", default="train"):
    splits = {row["split"]: {"has_ground_truth": True} for row in truth}
    metadata = {"name": name, "splits": splits, "default_eval_split": default}
    write_json(directory / "dataset.json", metadata)
    pairs = sorted({(row["split"], row["video_id"]) for row in truth})
    write_jsonl(directory / "videos.jsonl", [
        {"video_id": vid, "split": split, "video_path": f"{vid}.mp4", "duration": 200}
        for split, vid in pairs])
    write_jsonl(directory / "queries.jsonl", [
        {"query_id": row["query_id"], "video_id": row["video_id"], "split": row["split"], "query": "fixture"}
        for row in truth])


class Captioner:
    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def caption(self, path):
        with self.lock:
            self.calls += 1
        assert path.read_bytes().startswith(b"\xff\xd8")
        return "A red scene containing no people."


@pytest.fixture
def cfg(tmp_path):
    config = load_config(ROOT / "config.yaml")
    for kind in ("datasets", "wiki", "runs", "results"):
        config["paths"][kind] = str(tmp_path / kind)
    config["ingest"]["sample_interval_sec"] = 1.0
    config["ingest"]["image_max_size"] = 64
    config["ingest"]["vlm"]["model"] = "fixture-vlm"
    config["query"]["model"] = "fixture-agent"
    yield config
    for kind in ("wiki", "runs"):
        path = tmp_path / kind
        if path.exists():
            remove_tree(path)


@pytest.fixture
def prepared(cfg, tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe required for video integration tests")
    video_root = tmp_path / "videos"
    video_root.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=80x60:r=10",
                    "-t", "3", "-c:v", "mpeg4", "-y", str(video_root / "video.mp4")], check=True)
    raw = tmp_path / "annotations.jsonl"
    write_jsonl(raw, [
        {"qid": i, "vid": "video", "duration": 3.0, "query": f"Find red scene {i}",
         "split": "train", "relevant_windows": [[0.0, 1.0], [2.0, 3.0]]} for i in range(1, 4)
    ])
    dataset = Path(cfg["paths"]["datasets"]) / "qvhighlights"
    QVHighlightsAdapter().prepare(raw, video_root, dataset, split="train")
    cfg["dataset"]["split"] = "train"
    return cfg, Captioner()


@pytest.fixture
def frozen(prepared):
    cfg, captioner = prepared
    ingest_all(cfg, captioner=captioner)
    freeze_dataset(cfg)
    return cfg, captioner
