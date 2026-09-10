"""Prepare, run and summarize matched caption ablations. Prepare never calls APIs."""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from copy import deepcopy
from pathlib import Path
import time

import yaml

from harness.common import (HarnessError, atomic_text, cli, file_hash, object_hash,
                            positive_int, read_json, read_jsonl, write_json, write_jsonl)
from harness.config import dataset_path, load_config
from harness.dataset import dataset_context, load_query_inputs, require_ground_truth


VARIANTS = (("simple_1_1", "simple", 1, 1), ("simple_5_1", "simple", 5, 1),
            ("dense_5_1", "dense", 5, 1), ("dense_9_4", "dense", 9, 4))
SIMPLE_PROMPT = (
    "Describe the visible content in the provided sampled video frame or frames. "
    "State the people, objects, actions, spatial relations, and visible state changes. "
    "Describe only evidence visible in the images. Return one concise factual paragraph."
)


def source_hash():
    root = Path(__file__).resolve().parents[1]
    return object_hash({str(p.relative_to(root)): file_hash(p)
                        for folder in ("harness", "adapters", "agents")
                        for p in sorted((root / folder).glob("*.py"))})


def prepare(cfg: dict, output: Path, *, split: str, limit_videos: int = 10, seed: int = 0):
    positive_int(limit_videos, "limit_videos")
    if output.exists():
        raise HarnessError("Ablation directory already exists; choose a new output")
    output = output.resolve()
    directory, metadata, split = dataset_context(cfg, split)
    require_ground_truth(metadata, split)
    videos, queries = load_query_inputs(directory, split, metadata)
    selected = sorted({q["video_id"] for q in queries},
                      key=lambda vid: object_hash([seed, vid]))[:limit_videos]
    selected = set(selected)
    subset_queries = [q for q in queries if q["video_id"] in selected]
    if not subset_queries:
        raise HarnessError("Selected videos have no queries; choose another seed or larger subset")
    subset_rows = []
    for vid in sorted(selected):
        video = deepcopy(videos[vid])
        path = Path(video["video_path"])
        video["video_path"] = str((directory / path).resolve())
        subset_rows.append(video)
    truth = [row for row in read_jsonl(directory / "ground_truth.jsonl")
             if row["split"] == split and row["video_id"] in selected]
    snapshot = output / "datasets" / metadata["name"]
    subset_meta = {**metadata, "splits": {split: metadata["splits"][split]},
                   "default_eval_split": split}
    write_json(snapshot / "dataset.json", subset_meta)
    write_jsonl(snapshot / "videos.jsonl", subset_rows)
    write_jsonl(snapshot / "queries.jsonl", subset_queries)
    write_jsonl(snapshot / "ground_truth.jsonl", truth)
    dense_prompt = cfg["ingest"]["vlm"]["prompt"]
    if "{{FRAME_TIMESTAMPS}}" not in dense_prompt:
        dense_prompt = load_config(Path(__file__).resolve().parents[1] / "config.yaml")["ingest"]["vlm"]["prompt"]
    configs = {}
    for name, mode, window, stride in VARIANTS:
        variant = deepcopy(cfg)
        variant["dataset"] = {"name": metadata["name"], "split": split}
        variant["paths"].update(datasets=str(output / "datasets"),
                                wiki=str(output / "wiki" / name),
                                results=str(output / "results"), runs=str(output / "runs" / name))
        variant["ingest"].update(caption_mode=mode, caption_window_frames=window,
                                  caption_stride_frames=stride, dense_timestamp_mode="absolute_seconds")
        variant["ingest"]["vlm"]["prompt"] = dense_prompt if mode == "dense" else SIMPLE_PROMPT
        path = output / "configs" / f"{name}.yaml"
        atomic_text(path, yaml.safe_dump(variant, sort_keys=False, allow_unicode=True))
        load_config(path)
        configs[name] = file_hash(path)
    manifest = {"version": 1, "dataset": metadata["name"], "split": split,
                "source_hash": source_hash(),
                "video_hashes": {row["video_id"]: file_hash(Path(row["video_path"])) for row in subset_rows},
                "seed": seed, "video_ids": sorted(selected),
                "num_queries": len(subset_queries), "configs": configs,
                "dataset_files": {p.name: file_hash(p) for p in snapshot.iterdir()},
                "templates": {p: file_hash(Path(cfg["paths"]["templates"]) / p)
                              for p in ("AGENTS.md", "query_prompt.md")}}
    write_json(output / "suite.json", manifest)
    return manifest


def load_suite(root: Path):
    root = root.resolve()
    manifest = read_json(root / "suite.json")
    if manifest["source_hash"] != source_hash():
        raise HarnessError("Ablation source code changed; prepare a new suite")
    configs = {}
    for name, expected in manifest["configs"].items():
        path = root / "configs" / f"{name}.yaml"
        if file_hash(path) != expected:
            raise HarnessError(f"Ablation config changed: {path}; prepare a new suite")
        cfg = load_config(path)
        for name_, digest in manifest["templates"].items():
            if file_hash(Path(cfg["paths"]["templates"]) / name_) != digest:
                raise HarnessError("Ablation templates changed; prepare a new suite")
        configs[name] = cfg
    directory = root / "datasets" / manifest["dataset"]
    for name, digest in manifest["dataset_files"].items():
        if file_hash(directory / name) != digest:
            raise HarnessError("Ablation dataset snapshot changed")
    for row in read_jsonl(directory / "videos.jsonl"):
        if file_hash(Path(row["video_path"])) != manifest["video_hashes"][row["video_id"]]:
            raise HarnessError("Ablation source video changed; prepare a new suite")
    return manifest, configs


def run(root: Path, stage: str, jobs: int):
    from harness.aggregate import aggregate
    from harness.evaluate import evaluate
    from harness.freeze import freeze_dataset
    from harness.ingest_all import ingest_all
    from harness.run_query import Experiment, run_status

    if stage not in ("all", "ingest", "query", "evaluate"):
        raise HarnessError(f"Invalid ablation stage: {stage}")
    positive_int(jobs, "jobs")
    manifest, configs = load_suite(root)
    for name, cfg in configs.items():
        measurement = root / "measurements" / f"{name}.json"
        durations = read_json(measurement) if measurement.exists() else {}
        for step in ("ingest", "query", "evaluate"):
            if stage not in ("all", step):
                continue
            started = time.monotonic()
            result_root = Path(cfg["paths"]["results"]) / name
            try:
                if step == "ingest":
                    ingest_all(cfg, split=manifest["split"], jobs=jobs)
                    freeze_dataset(cfg, split=manifest["split"])
                elif step == "query":
                    with Experiment(cfg, name) as experiment:
                        runtime_path = root / "runtime.json"
                        if runtime_path.exists() and read_json(runtime_path) != experiment.runner.provenance:
                            raise HarnessError("Ablation agent runtime changed; prepare a new suite")
                        write_json(runtime_path, experiment.runner.provenance)
                        for query in experiment.queries:
                            print(run_status(experiment.run(query), experiment.root))
                else:
                    prediction = result_root / "predictions.jsonl"
                    aggregate(result_root / "predictions", prediction)
                    metrics = evaluate(prediction, dataset_dir=dataset_path(cfg, "datasets"),
                                       split=manifest["split"],
                                       top_k=cfg["evaluation"]["top_k"],
                                       iou_thresholds=cfg["evaluation"]["iou_thresholds"],
                                       max_predictions=cfg["query"]["max_predictions"])
                    write_json(result_root / "metrics.json", metrics)
            finally:
                durations[step] = durations.get(step, 0.0) + time.monotonic() - started
                write_json(measurement, durations)


def summarize(root: Path) -> dict:
    manifest, configs = load_suite(root)
    rows = []
    for name, cfg in configs.items():
        video_root = dataset_path(cfg, "wiki") / "videos"
        telemetry = []
        for vid in manifest["video_ids"]:
            path = video_root / vid / "ingest.json"
            if path.exists():
                telemetry.append(read_json(path).get("telemetry", {}))
        result_root = Path(cfg["paths"]["results"]) / name
        path = result_root / "metrics.json"
        metrics = read_json(path) if path.exists() else None
        if metrics is not None:
            if (metrics["dataset"], metrics["split"], metrics["num_queries"]) != (
                    manifest["dataset"], manifest["split"], manifest["num_queries"]):
                raise HarnessError(f"Ablation metrics do not match suite: {name}")
            if metrics["inputs"]["prediction_sha256"] != file_hash(result_root / "predictions.jsonl"):
                raise HarnessError(f"Ablation predictions changed since evaluation: {name}")
            if metrics["inputs"]["ground_truth_sha256"] != manifest["dataset_files"]["ground_truth.jsonl"]:
                raise HarnessError(f"Ablation ground truth does not match metrics: {name}")
        statuses = [read_json(p).get("status") for p in (result_root / "run_metadata").glob("*.json")]
        finished_queries = sum(s in ("success", "failed") for s in statuses)
        tokens_complete = bool(telemetry) and all(
            t.get("api_requests", 0) > 0 and t.get("api_requests") == t.get("requests_with_total_tokens")
            for t in telemetry)
        measurement = root / "measurements" / f"{name}.json"
        rows.append({"variant": name, "ingested_videos": len(telemetry),
                     "finished_queries": finished_queries,
                     "complete": len(telemetry) == len(manifest["video_ids"])
                                 and finished_queries == manifest["num_queries"] and metrics is not None,
                     "metrics": metrics,
                     "stage_wall_sec": read_json(measurement) if measurement.exists() else None,
                     "caption_attempts": sum(t.get("caption_attempts", 0) for t in telemetry) if telemetry else None,
                     "rejected_attempts": sum(t.get("rejected_attempts", 0) for t in telemetry) if telemetry else None,
                     "extraction_sec": sum(t.get("extraction_sec", 0) for t in telemetry) if telemetry else None,
                     "caption_sec": sum(t.get("caption_sec", 0) for t in telemetry) if telemetry else None,
                     "tokens_complete": tokens_complete,
                     "caption_total_tokens": sum(t.get("usage", {}).get("total_tokens", 0) for t in telemetry)
                                     if tokens_complete else None})
    return {"dataset": manifest["dataset"], "split": manifest["split"],
            "num_videos": len(manifest["video_ids"]), "num_queries": manifest["num_queries"],
            "note": "Costs cover completed videos, including their retained failed attempts. "
                    "Unavailable token usage is null; no model pricing is assumed.", "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare", help="Create matched configs and dataset snapshot; no API calls")
    prep.add_argument("--config", default="config.yaml")
    prep.add_argument("--dataset")
    prep.add_argument("--split", required=True)
    prep.add_argument("--limit-videos", type=int, default=10)
    prep.add_argument("--seed", type=int, default=0)
    prep.add_argument("--output", type=Path, required=True)
    execute = sub.add_parser("run", help="Run real caption/query APIs for a prepared suite")
    execute.add_argument("--suite", type=Path, required=True)
    execute.add_argument("--stage", choices=("all", "ingest", "query", "evaluate"), default="all")
    execute.add_argument("--jobs", type=int, default=4)
    report = sub.add_parser("summarize")
    report.add_argument("--suite", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        cfg = load_config(args.config)
        if args.dataset:
            cfg["dataset"]["name"] = args.dataset
        result = prepare(cfg, args.output, split=args.split, limit_videos=args.limit_videos, seed=args.seed)
        print(f"Prepared {len(result['configs'])} variants, {result['num_queries']} queries at {args.output}")
    elif args.action == "run":
        run(args.suite.resolve(), args.stage, args.jobs)
    else:
        result = summarize(args.suite)
        path = args.suite / "comparison.json"
        write_json(path, result)
        print(path)


if __name__ == "__main__":
    cli(main)
