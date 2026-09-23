"""Shared dataset/split provenance checks for aggregation and evaluation."""

from pathlib import Path

from vmr.core.errors import HarnessError
from vmr.core.hashing import file_hash, object_hash
from vmr.core.jsonio import read_json
from vmr.datasets.manifest import load_dataset, load_query_inputs, select_split
from vmr.datasets.validation import validate_prediction


def check_identity(record: dict, dataset: str, split: str) -> None:
    if record.get("dataset") != dataset or record.get("split") != split:
        raise HarnessError(
            f"Result dataset/split mismatch: expected {dataset!r}/{split!r}, "
            f"got {record.get('dataset')!r}/{record.get('split')!r}"
        )


def experiment_context(root: Path, dataset_dir: Path | None, split: str | None):
    saved_path = root / "config.yaml"
    # Experiment snapshots use the normalized internal (v1-shaped) contract;
    # they are not user-authored legacy configs.
    snapshot_path = root / "query-config.json"
    snapshot_dataset = None
    if snapshot_path.exists():
        snapshot = read_json(snapshot_path)
        if snapshot.get("schema_version") != 1:
            raise HarnessError("Unsupported query config snapshot version")
        snapshot_dataset = Path(snapshot["dataset_path"])
        saved = {
            "query": snapshot["query"],
            "dataset": {"name": snapshot["dataset"], "split": snapshot["split"]},
            "paths": {"datasets": str(Path(snapshot["dataset_path"]).parent)},
        }
    else:
        from vmr.compat.config import load_config

        saved = (
            load_config(saved_path, warn_legacy=False) if saved_path.exists() else None
        )
    experiment_path = root / "experiment.json"
    experiment = read_json(experiment_path) if experiment_path.exists() else None
    if dataset_dir is None:
        if saved is None:
            raise HarnessError(
                "Specify a dataset context, or aggregate inside a saved experiment"
            )
        dataset_dir = (
            snapshot_dataset
            if snapshot_dataset is not None
            else Path(saved["paths"]["datasets"]) / saved["dataset"]["name"]
        )
    dataset = load_dataset(dataset_dir)
    if split is None:
        split = (
            experiment.get("split")
            if experiment
            else (saved["dataset"].get("split") if saved else None)
        )
    split = select_split(dataset, split)
    videos, queries = load_query_inputs(dataset_dir, split, dataset)
    if experiment:
        check_identity(experiment, dataset["name"], split)
        if (
            experiment.get("dataset_hash") != object_hash(dataset)
            or experiment.get("queries_hash") != object_hash(queries)
            or experiment.get("videos_hash") != object_hash(videos)
        ):
            raise HarnessError(
                "Dataset metadata or selected annotations changed since experiment creation"
            )
    if saved:
        check_identity(
            {
                "dataset": saved["dataset"]["name"],
                "split": saved["dataset"].get("split"),
            },
            dataset["name"],
            split,
        )
    return dataset, split, videos, queries, saved, experiment


def validate_result(
    prediction: dict, queries: dict, videos: dict, split: str, max_predictions: int
) -> dict:
    validate_prediction(prediction, split=split, max_predictions=max_predictions)
    qid = prediction["query_id"]
    if qid not in queries:
        raise HarnessError(
            f"Prediction query {qid!r} does not belong to selected dataset/split"
        )
    vid = queries[qid]["video_id"]
    return validate_prediction(
        prediction,
        query_id=qid,
        video_id=vid,
        split=split,
        duration=videos[vid]["duration"],
        max_predictions=max_predictions,
    )


def bundle_metadata_path(predictions: Path) -> Path:
    return predictions.with_name(predictions.name + ".metadata.json")


def verify_bundle(
    predictions: Path,
    dataset: dict,
    split: str,
    queries: list[dict],
    *,
    required: bool = False,
) -> None:
    sidecar = bundle_metadata_path(predictions)
    if not sidecar.exists():
        if required:
            raise HarnessError(f"Missing aggregate provenance sidecar: {sidecar}")
        return
    record = read_json(sidecar)
    check_identity(record, dataset["name"], split)
    if (
        record.get("prediction_sha256") != file_hash(predictions)
        or record.get("dataset_hash") != object_hash(dataset)
        or record.get("queries_hash") != object_hash(queries)
    ):
        raise HarnessError("Aggregate provenance/hash mismatch")
