"""Architectural rules complement the stripped-install integration test."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def imports(path):
    tree = ast.parse(path.read_text())
    return [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [
        name.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import)
        for name in n.names
    ]


def test_query_has_no_compiler_dependency():
    for path in (ROOT / "vmr/query").rglob("*.py"):
        assert not any(
            n.startswith(
                ("vmr.compiler", "harness.ingest", "harness.freeze", "harness.config")
            )
            for n in imports(path)
        ), path
        source = path.read_text()
        assert "ingest_content_hash" not in source and "ingest.json" not in source
        assert not any(
            n in source for n in ("nodes.jsonl", "observations.jsonl", "compiler.name")
        )


def test_core_and_artifact_have_no_method_dependency():
    for folder in ("vmr/core", "vmr/artifact"):
        for path in (ROOT / folder).rglob("*.py"):
            assert not any(
                n.startswith(("vmr.compiler", "harness.ingest", "harness.config"))
                for n in imports(path)
            ), path


def test_compiler_does_not_import_query_or_reverse_orchestrator():
    for path in (ROOT / "vmr/compiler").rglob("*.py"):
        assert not any(
            n.startswith(("vmr.query", "harness.run_query", "harness.workspace"))
            for n in imports(path)
        ), path
        if "methods" in path.parts:
            assert not any(
                n in ("vmr.compiler.pipeline", "vmr.compat.ingest", "harness.ingest")
                for n in imports(path)
            ), path


def test_graph_edit_domain_has_no_io_dependency():
    path = ROOT / "vmr/compiler/methods/bidirectional/edits.py"
    assert not any(n in ("sqlite3", "pathlib", "os") for n in imports(path))


def test_artifact_contains_no_legacy_file_policy():
    names = {
        "nodes.jsonl",
        "observations.jsonl",
        "bottomup_observations.jsonl",
        "coverage.jsonl",
        "ingest.json",
        "frozen.json",
    }
    for path in (ROOT / "vmr/artifact").rglob("*.py"):
        literals = {
            n.value
            for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert not names & literals, path
        assert not any(
            n.startswith(("vmr.compat", "harness")) for n in imports(path)
        ), path


def test_new_entry_points_use_moved_dataset_and_evaluation_services():
    for folder in (
        "vmr/cli",
        "vmr/query",
        "vmr/compiler",
        "vmr/datasets",
        "vmr/evaluation",
    ):
        for path in (ROOT / folder).rglob("*.py"):
            assert not any(
                n
                in (
                    "harness.dataset",
                    "harness.evaluate",
                    "harness.results",
                    "harness.metrics",
                    "harness.validate",
                )
                for n in imports(path)
            ), path


def test_new_execution_layers_do_not_import_legacy_shims():
    for folder in ("core", "media", "vlm", "compiler", "query", "cli"):
        for path in (ROOT / "vmr" / folder).rglob("*.py"):
            assert not any(
                n == "harness" or n.startswith("harness.") for n in imports(path)
            ), path
            if folder in ("core", "media", "vlm", "compiler", "query"):
                assert not any(
                    n == "vmr.compat" or n.startswith("vmr.compat.")
                    for n in imports(path)
                ), path
