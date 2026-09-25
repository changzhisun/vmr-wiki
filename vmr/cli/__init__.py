"""CLI command imports are lazy so Query works without compiler packages."""

from pathlib import Path
import argparse
from vmr.core.envfile import load_env_file
from vmr.core.validation import cli


def main(argv=None):
    load_env_file()
    parser = argparse.ArgumentParser(
        prog="vmr", description="Compile → Immutable Artifact → Query → Evaluate"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser(
        "compile", help="Compile and atomically seal a WikiSet"
    )
    compile_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/compiler/base.yaml"),
        help="compile config (default: %(default)s)",
    )
    compile_parser.add_argument("--dataset", required=True, type=Path)
    compile_parser.add_argument(
        "--split", help="dataset split (default: dataset.split from the config)"
    )
    compile_parser.add_argument("--output-set", required=True, type=Path)
    compile_parser.add_argument("--artifact-store", type=Path)
    compile_parser.add_argument("-j", "--jobs", type=int, default=1)
    query = commands.add_parser(
        "query", help="Query sealed artifacts without compiler config/code"
    )
    query.add_argument(
        "--config",
        type=Path,
        default=Path("configs/query/base.yaml"),
        help="query config (default: %(default)s)",
    )
    query.add_argument("--dataset", required=True, type=Path)
    query.add_argument(
        "--split", help="dataset split (default: dataset.split from the config)"
    )
    query.add_argument("--wiki-set", type=Path)
    query.add_argument("--video-root", type=Path)
    query.add_argument("--experiment", required=True, type=Path)
    query.add_argument("--query-id")
    query.add_argument("-j", "--jobs", type=int, default=1)
    ev = commands.add_parser("evaluate")
    ev.add_argument("--dataset", required=True, type=Path)
    ev.add_argument(
        "--split",
        help="dataset split (default: the split recorded on the experiment)",
    )
    ev.add_argument("--experiment", required=True, type=Path)
    ev.add_argument("--official-root", type=Path)
    artifacts = commands.add_parser("artifact").add_subparsers(
        dest="operation", required=True
    )
    validate = artifacts.add_parser("validate")
    validate.add_argument("path", type=Path)
    migrate = artifacts.add_parser("migrate")
    migrate.add_argument("path", type=Path)
    migrate.add_argument("--artifact-store", required=True, type=Path)
    sets = commands.add_parser("wikiset").add_subparsers(
        dest="operation", required=True
    )
    validate = sets.add_parser("validate")
    validate.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "compile":
        from vmr.config.migrate import load_compile_config
        from vmr.compiler.batch import compile_dataset

        cfg = load_compile_config(args.config)
        result = compile_dataset(
            cfg,
            args.dataset,
            args.split or cfg.dataset.split,
            args.output_set,
            store=args.artifact_store,
            jobs=args.jobs,
        )
        print(
            f"Sealed WikiSet: {args.output_set} ({len(result.data['artifacts'])} artifacts)"
        )
    elif args.command == "query":
        from vmr.config.query import load_query_config
        from vmr.query.experiment import Experiment
        from vmr.query.batch import run_batch

        cfg = load_query_config(args.config)
        with Experiment(
            cfg.query,
            dataset=args.dataset,
            split=args.split or cfg.dataset.split,
            wiki_set=args.wiki_set,
            video_root=args.video_root,
            root=args.experiment,
            templates=cfg.storage.templates,
            runs=cfg.storage.runs,
        ) as experiment:
            failed = run_batch(experiment, jobs=args.jobs, query_id=args.query_id)
        if failed:
            raise SystemExit(1)
    elif args.command == "evaluate":
        from vmr.evaluation.aggregate import aggregate
        from vmr.evaluation.evaluate import evaluate
        from vmr.core.jsonio import write_json, read_json

        output = args.experiment / "predictions.jsonl"
        split = args.split
        experiment_path = args.experiment / "experiment.json"
        if split is None and experiment_path.is_file():
            split = read_json(experiment_path).get("split")
        aggregate(
            args.experiment / "predictions",
            output,
            dataset_dir=args.dataset,
            split=split,
        )
        snapshot = read_json(args.experiment / "query-config.json")
        result = evaluate(
            output,
            dataset_dir=args.dataset,
            split=split,
            max_predictions=snapshot["query"]["max_predictions"],
            official_root=args.official_root,
        )
        write_json(args.experiment / "metrics.json", result)
        print(result["retrieval"])
    elif args.command == "artifact":
        if args.operation == "validate":
            from vmr.artifact.artifact import WikiArtifact

            artifact = WikiArtifact.open(args.path)
        else:
            from vmr.compat.artifact_migrate import migrate_legacy

            artifact = migrate_legacy(args.path, args.artifact_store)
        artifact.verify()
        print(f"{artifact.artifact_id()} {artifact.root}")
    elif args.command == "wikiset":
        from vmr.artifact.wikiset import WikiSet

        result = WikiSet.open(args.path)
        result.verify()
        print(result.fingerprint())


def entrypoint():
    cli(main)
