from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from adapters import get_evaluator
from adapters.qvhighlights import QVHighlightsAdapter
from conftest import Captioner, ROOT
from harness.aggregate import aggregate
from harness.common import HarnessError, read_json, read_jsonl, write_json, write_jsonl
from harness.dataset import (load_dataset, load_query_inputs, load_videos,
                             select_split, validate_dataset)
from harness.evaluate import evaluate
from harness.freeze import freeze_dataset, verify_wiki
from harness.ingest_all import ingest_all
from harness.run_query import Experiment
from harness.validate import validate_prediction
from test_query_pipeline import ProcessFixtureRunner


@pytest.fixture
def split_dataset(prepared, tmp_path):
    cfg, _ = prepared
    original = Path(read_jsonl(Path(cfg["paths"]["datasets"]) / "qvhighlights/videos.jsonl")[0]["video_path"])
    # Distinct media identities permit checking that unselected videos aren't ingested.
    for vid in ("train-only", "val-only"):
        (original.parent / f"{vid}.mp4").write_bytes(original.read_bytes())
    sources = {}
    for split in ("train", "val", "test"):
        rows = [{"qid": "1", "vid": "video", "duration": 3, "query": f"{split.upper()}_SECRET"}]
        if split != "test":
            rows.append({"qid": f"{split}-only", "vid": f"{split}-only", "duration": 3,
                         "query": f"{split.upper()}_SECRET_SECOND"})
            for row in rows:
                row["relevant_windows"] = [[0, 1], [2, 3]]
        source = tmp_path / f"highlight_{split}_release.jsonl"
        write_jsonl(source, rows)
        sources[split] = source
    cfg["paths"]["datasets"] = str(tmp_path / "split-datasets")
    cfg["dataset"]["split"] = None
    directory = Path(cfg["paths"]["datasets"]) / "qvhighlights"
    QVHighlightsAdapter().prepare(sources, original.parent, directory, default_eval_split="val")
    return cfg, directory, Captioner()


@pytest.mark.parametrize("names", [
    ["train", "val", "test"], ["training", "validation", "testing"],
    ["train", "val_1", "val_2"], ["all", "custom / split", "验证集"],
])
def test_opaque_metadata_names_and_unknown_rejection(tmp_path, names):
    metadata = {"name": "example", "splits": {name: {"has_ground_truth": i != 2}
                                               for i, name in enumerate(names)}}
    write_json(tmp_path / "dataset.json", metadata)
    loaded = load_dataset(tmp_path)
    assert list(loaded["splits"]) == names
    for name in names:
        assert select_split(loaded, name) == name
    with pytest.raises(HarnessError) as exc:
        select_split(loaded, "VAL")
    assert "Unknown split 'VAL'" in str(exc.value)
    assert all(f"- {name}" in str(exc.value) for name in names)
    with pytest.raises(HarnessError, match="Specify --split"):
        select_split(loaded, None)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_has_ground_truth_is_a_real_boolean(value):
    with pytest.raises(HarnessError, match="boolean"):
        validate_dataset({"name": "example", "splits": {"s": {"has_ground_truth": value}}})


def test_default_eval_split_validation():
    metadata = {"name": "example", "splits": {"validation": {"has_ground_truth": True},
                                                 "testing": {"has_ground_truth": False}},
                "default_eval_split": "validation"}
    assert select_split(validate_dataset(metadata), None, evaluation=True) == "validation"
    for default in ("val", "testing", None):
        with pytest.raises(HarnessError):
            validate_dataset({**metadata, "default_eval_split": default})


def test_filtering_and_query_ids_scoped_to_split(split_dataset):
    cfg, directory, _ = split_dataset
    metadata = load_dataset(directory)
    videos, queries = load_query_inputs(directory, "val")
    assert set(videos) == {"video", "val-only"}
    assert {q["query_id"] for q in queries} == {"1", "val-only"}
    assert all(q["split"] == "val" for q in queries)
    assert set(load_videos(directory, metadata, "test")) == {"video"}
    assert metadata["splits"]["test"]["has_ground_truth"] is False
    with pytest.raises(HarnessError, match="Unknown split 'validation'"):
        ingest_all(cfg, split="validation")


def test_cross_split_video_identity_must_match(split_dataset):
    _, directory, _ = split_dataset
    rows = read_jsonl(directory / "videos.jsonl")
    next(row for row in rows if row["split"] == "val" and row["video_id"] == "video")["duration"] = 9
    write_jsonl(directory / "videos.jsonl", rows)
    with pytest.raises(HarnessError, match="Conflicting identity"):
        load_query_inputs(directory, "val")


def test_shared_wiki_ingested_once_with_incremental_split_freeze(split_dataset):
    cfg, directory, captioner = split_dataset
    root = Path(cfg["paths"]["wiki"]) / "qvhighlights" / "videos"
    ingest_all(cfg, split="val", captioner=captioner)
    first = freeze_dataset(cfg, split="val")
    assert set(p.name for p in root.iterdir()) == {"video", "val-only"}
    assert captioner.calls == 6
    shared_hash = verify_wiki(root / "video")["wiki_hash"]
    ingest_all(cfg, split="train", captioner=captioner)
    freeze_dataset(cfg, split="train")
    assert captioner.calls == 9  # shared video was already frozen by val
    ingest_all(cfg, split="test", captioner=captioner)
    assert captioner.calls == 9
    assert verify_wiki(root / "video")["wiki_hash"] == shared_hash
    assert freeze_dataset(cfg, split="val") == first
    assert (root / "video" / "wiki.md").stat().st_mode & 0o222 == 0
    assert not (root.parent / "val").exists()


def test_query_isolation_and_unlabeled_prediction_generation(split_dataset):
    cfg, directory, captioner = split_dataset
    ingest_all(cfg, split="val", captioner=captioner)
    freeze_dataset(cfg, split="val")
    # GT is deliberately unreadable JSON; neither selected Query flow may read it.
    (directory / "ground_truth.jsonl").write_text("GT_SECRET_INVALID_JSON")
    class Spy(ProcessFixtureRunner):
        def run(self, workspace, prompt, stdout, stderr):
            task = read_json(workspace / "task.json")
            assert task["query"].startswith("VAL_SECRET")
            # The real split and identifiers are lookup keys into public data,
            # so the workspace carries opaque per-experiment aliases instead.
            assert task["split"] != "val" and task["split"].startswith("s")
            assert task["query_id"].startswith("q") and task["video_id"].startswith("v")
            readable = [path for path in workspace.rglob("*")
                        if path.is_file() and path.suffix in (".md", ".json", ".jsonl")]
            text = "\n".join(path.read_text() for path in readable)
            names = "\n".join(str(path.relative_to(workspace)) for path in readable)
            assert all(secret not in text for secret in ("TRAIN_SECRET", "TEST_SECRET", "GT_SECRET"))
            assert "val-only" not in text and "val-only" not in names
            assert "val-only" not in str(workspace.name)
            return super().run(workspace, prompt, stdout, stderr)
    with Experiment(cfg, "val-run", split="val", runner=Spy()) as experiment:
        for query in experiment.queries:
            result = experiment.run(query)
            assert result["dataset"] == "qvhighlights" and result["split"] == "val"
            assert result["status"] == "success"
        with pytest.raises(HarnessError, match="selected split"):
            experiment.run({**experiment.queries[0], "split": "train"})
    with pytest.raises(HarnessError, match="changed"):
        with Experiment(cfg, "val-run", split="test", runner=ProcessFixtureRunner()):
            pass
    (directory / "ground_truth.jsonl").unlink()
    with Experiment(cfg, "test-run", split="test", runner=ProcessFixtureRunner()) as experiment:
        assert experiment.run(experiment.queries[0])["status"] == "success"
    root = Path(cfg["paths"]["results"]) / "test-run"
    rows = aggregate(root / "predictions", root / "predictions.jsonl", split="test")
    assert rows[0]["split"] == "test" and len(rows[0]["moments"]) == 2
    with pytest.raises(HarnessError, match="has_ground_truth=false"):
        evaluate(root / "predictions.jsonl", dataset_dir=directory, split="test")


def test_evaluation_and_aggregate_reject_mixed_provenance(split_dataset, tmp_path):
    cfg, directory, captioner = split_dataset
    ingest_all(cfg, split="val", captioner=captioner)
    freeze_dataset(cfg, split="val")
    with Experiment(cfg, "val-run", split="val", runner=ProcessFixtureRunner()) as experiment:
        for query in experiment.queries:
            experiment.run(query)
    root = Path(cfg["paths"]["results"]) / "val-run"
    bundle = root / "predictions.jsonl"
    aggregate(root / "predictions", bundle)
    metrics = evaluate(bundle, dataset_dir=directory)  # dataset default_eval_split
    assert metrics["split"] == "val" and metrics["num_queries"] == 2
    assert metrics["primary_score"] == 100.0
    assert metrics["average_predictions_per_query"] == 2.0
    sidecar = bundle.with_name(bundle.name + ".metadata.json")
    assert read_json(sidecar)["dataset"] == "qvhighlights"
    rows = read_jsonl(bundle)
    rows[0]["moments"] = []
    write_jsonl(bundle, rows)
    sidecar.unlink()
    with pytest.raises(HarnessError, match="Missing aggregate provenance sidecar"):
        evaluate(bundle, dataset_dir=directory)
    aggregate(root / "predictions", bundle)
    with pytest.raises(HarnessError):
        evaluate(bundle, dataset_dir=directory, split="train")
    meta_path = root / "run_metadata/1.json"
    saved = read_json(meta_path)
    write_json(meta_path, {**saved, "dataset": "another_dataset"})
    with pytest.raises(HarnessError, match="dataset/split mismatch"):
        aggregate(root / "predictions", bundle)
    write_json(meta_path, saved)
    rows = read_jsonl(bundle)
    rows[0]["split"] = "train"
    detached = tmp_path / "mixed.jsonl"
    write_jsonl(detached, rows)
    with pytest.raises(HarnessError, match="Mismatched split"):
        evaluate(detached, dataset_dir=directory, split="val")
    original = read_json(root / "predictions/1.json")
    write_json(root / "predictions/1.json", {**original, "split": "train"})
    with pytest.raises(HarnessError, match="Mismatched split"):
        aggregate(root / "predictions", bundle, split="val")


def test_adapter_discovers_original_names_and_unlabeled_split(tmp_path):
    raw = {"qid": 1, "vid": "v", "duration": 3, "query": "query"}
    for name in ("val_1", "val_2", "testing"):
        write_jsonl(tmp_path / f"highlight_{name}_release.jsonl", [raw])
    sources = QVHighlightsAdapter.discover_splits(tmp_path)
    assert set(sources) == {"val_1", "val_2", "testing"}
    output = tmp_path / "dataset"
    QVHighlightsAdapter().prepare(sources, tmp_path, output)
    metadata = load_dataset(output)
    assert all(not info["has_ground_truth"] for info in metadata["splits"].values())
    assert not (output / "ground_truth.jsonl").exists()
    with pytest.raises(HarnessError, match="has_ground_truth=false"):
        evaluate(tmp_path / "missing-prediction.jsonl", dataset_dir=output, split="testing")


def test_adapter_rejects_partially_labeled_split(tmp_path):
    raw = {"qid": 1, "vid": "v", "duration": 3, "query": "query"}
    source = tmp_path / "mixed.jsonl"
    write_jsonl(source, [raw, {**raw, "qid": 2, "relevant_windows": [[0, 1]]}])
    with pytest.raises(HarnessError, match="GT and query IDs"):
        QVHighlightsAdapter().prepare(source, tmp_path, tmp_path / "dataset", split="validation")


def test_schema_supports_requested_top_level_evidence_and_multiple_moments():
    row = {"query_id": "q", "video_id": "v", "split": "validation", "evidence": "description",
           "moments": [{"start_sec": 0, "end_sec": 1, "score": .9},
                       {"start_sec": 2, "end_sec": 3, "score": .5}]}
    assert validate_prediction(row, split="validation") == row
    with pytest.raises(HarnessError, match="Mismatched split"):
        validate_prediction(row, split="val")


def test_legacy_metadata_and_manifests_require_adapter_migration(tmp_path):
    with pytest.raises(HarnessError, match="Re-run the Dataset Adapter"):
        load_dataset(tmp_path)
    write_json(tmp_path / "dataset.json", {"name": "old", "splits": {"val": {"has_ground_truth": True}}})
    write_jsonl(tmp_path / "videos.jsonl", [{"video_id": "v", "video_path": "v.mp4", "duration": 3}])
    with pytest.raises(HarnessError, match="Re-run the Dataset Adapter"):
        load_query_inputs(tmp_path, "val")


def test_cli_split_help_and_prediction_only_rejection(split_dataset, tmp_path):
    cfg, directory, _ = split_dataset
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    for module in ("ingest_all", "freeze", "run_query", "run_all_queries", "aggregate", "evaluate", "validate"):
        proc = subprocess.run([sys.executable, f"harness/{module}.py", "--help"], cwd=ROOT,
                              capture_output=True, text=True)
        assert proc.returncode == 0 and "--split" in proc.stdout
    (directory / "ground_truth.jsonl").unlink()
    proc = subprocess.run([sys.executable, "harness/evaluate.py", "--config", str(config_path),
                           "--dataset", "qvhighlights", "--split", "test", "--pred", str(tmp_path / "missing.jsonl")],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0 and "has_ground_truth=false" in proc.stderr


def test_evaluator_resolution_is_adapter_driven():
    assert callable(get_evaluator("qvhighlights"))
    assert get_evaluator("generic")({}, {})["implementation"] == "generic"
    with pytest.raises(HarnessError, match="Unknown evaluator"):
        get_evaluator("does_not_exist")


def test_second_adapter_preserves_its_split_and_evaluator(tmp_path):
    from adapters.uca import UCAAdapter
    raw = {"query_id": "q", "video_id": "v", "path": "Videos/v.mp4", "duration": 20,
           "query": {"text": "A person enters"}, "gold_moments": [[1, 3], [10, 12]]}
    for split in ("training", "validation"):
        write_jsonl(tmp_path / f"{split}.jsonl", [raw])
    directory = tmp_path / "uca"
    UCAAdapter().prepare(UCAAdapter.discover_splits(tmp_path), tmp_path, directory,
                         default_eval_split="validation")
    metadata = load_dataset(directory)
    assert metadata["evaluator"] == "generic"
    assert set(metadata["splits"]) == {"training", "validation"}
    rows = read_jsonl(directory / "ground_truth.jsonl")
    assert len(rows) == 2 and all(len(row["moments"]) == 2 for row in rows)
    prediction = {"query_id": "q", "video_id": "v", "split": "validation",
                  "moments": [{"start_sec": 1, "end_sec": 3, "score": 1}]}
    pred = tmp_path / "prediction.jsonl"
    write_jsonl(pred, [prediction])
    with pytest.raises(HarnessError, match="Missing aggregate provenance sidecar"):
        evaluate(pred, dataset_dir=directory)
    metrics = evaluate(pred, dataset_dir=directory, allow_unverified_predictions=True)
    assert metrics["implementation"] == "generic"
    assert metrics["split"] == "validation" and metrics["num_queries"] == 1
    assert metrics["retrieval"]["R@1,IoU=0.5"] == 100


def test_cli_adapter_aggregate_and_evaluate_without_gt_argument(split_dataset, tmp_path):
    cfg, _, _ = split_dataset
    cfg = copy.deepcopy(cfg)
    cfg["paths"]["datasets"] = str(tmp_path / "cli-datasets")
    cfg_path = tmp_path / "cli-config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    directory = Path(cfg["paths"]["datasets"]) / "qvhighlights"
    commands = [["adapters/qvhighlights.py", "--annotation-dir", str(tmp_path),
                 "--video-root", str(tmp_path / "videos"), "--output", str(directory),
                 "--default-eval-split", "val"]]
    proc = subprocess.run([sys.executable, *commands[0]], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    _, queries = load_query_inputs(directory, "val")
    predictions = tmp_path / "cli-results" / "predictions"
    predictions.mkdir(parents=True)
    for query in queries:
        write_json(predictions / f"{query['query_id']}.json", {
            "query_id": query["query_id"], "video_id": query["video_id"], "split": "val",
            "moments": [{"start_sec": 0, "end_sec": 1, "score": .9},
                        {"start_sec": 2, "end_sec": 3, "score": .8}], "evidence": "fixture"})
    bundle = predictions.parent / "predictions.jsonl"
    for command in (
        ["harness/aggregate.py", "--config", str(cfg_path), "--dataset", "qvhighlights", "--split", "val",
         "--input", str(predictions), "--output", str(bundle)],
        ["harness/evaluate.py", "--config", str(cfg_path), "--dataset", "qvhighlights", "--pred", str(bundle)],
    ):
        proc = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    result = read_json(predictions.parent / "metrics.json")
    assert result["dataset"] == "qvhighlights" and result["split"] == "val"
    assert result["num_queries"] == 2 and result["primary_score"] == 100.0
