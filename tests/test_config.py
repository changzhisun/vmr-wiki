from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from harness.common import HarnessError, ingest_content_hash
from harness.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_v2_normalizes_to_the_legacy_internal_contract(tmp_path):
    v2 = load_config(ROOT / "config.yaml")
    legacy_path = tmp_path / "legacy.yaml"
    legacy_path.write_text(yaml.safe_dump(v2, sort_keys=False), encoding="utf-8")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        v1 = load_config(legacy_path)
    assert v1 == v2
    assert ingest_content_hash(v1) == ingest_content_hash(v2)


def test_extends_deep_merges_and_resolves_storage_from_entry_config(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    query = raw.pop("query")
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text(yaml.safe_dump({
        "extends": "base.yaml",
        "query": {**query, "input_mode": "text", "timeout_sec": 42},
    }), encoding="utf-8")

    cfg = load_config(tmp_path / "experiment.yaml")
    assert cfg["query"]["text_only"] is True
    assert cfg["query"]["timeout_sec"] == 42
    assert cfg["paths"]["datasets"] == str(tmp_path / "datasets")


def test_extends_cycle_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(HarnessError, match="extends cycle"):
        load_config(tmp_path / "a.yaml")


def test_named_agent_profile_and_cli_kind_selection():
    path = ROOT / "config.yaml"
    by_name = load_config(path, query_agent_profile="claude_default",
                          query_model="fixture-claude")
    by_kind = load_config(path, query_agent="claude_code", query_model="fixture-claude")
    assert by_name == by_kind
    assert by_name["query"]["agent"] == "claude_code"
    assert by_name["query"]["api_key_env"] == {"claude_code": "ANTHROPIC_API_KEY"}


def test_v2_rejects_inactive_method_and_role_fields(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["wiki"]["method_config"]["legacy_window_frames"] = 5
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(HarnessError, match="Unknown or malformed bidirectional"):
        load_config(path)
