from pathlib import Path
import subprocess
import sys
import pytest
import yaml
from vmr.config.query import load_query_config
from vmr.config.migrate import load_compile_config, from_legacy

ROOT = Path(__file__).resolve().parents[2]


def test_query_config_ignores_compile_domain(tmp_path):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["wiki"] = {"method": "uninstalled-third-party", "bad": object.__name__}
    raw["profiles"].pop("vlms")
    path = tmp_path / "query.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_query_config(path)
    assert cfg.compile is None
    raw.pop("wiki")
    path.write_text(yaml.safe_dump(raw))
    assert load_query_config(path).query == cfg.query


def test_legacy_config_migrates_to_typed_domains(cfg):
    app = from_legacy(cfg)
    assert app.compile.compiler == "simple"
    assert app.compile.media.image_max_size == cfg["ingest"]["image_max_size"]
    assert app.query.runtime_config() == cfg["query"]


@pytest.mark.parametrize(
    "method", ["simple", "dense", "hierarchical", "bidirectional", "agentic"]
)
def test_compile_only_v3(tmp_path, method):
    raw = yaml.safe_load((ROOT / "config.yaml").read_text())
    raw["version"] = 3
    section = raw.pop("wiki")
    section["compiler"] = method
    section.pop("method")
    section["method_config"] = {}
    raw["compile"] = section
    raw.pop("query")
    raw["storage"]["templates"] = str(ROOT / "templates")
    if method == "agentic":
        section.pop("captioner_profile")
        section["kind"] = "codex"
        raw["profiles"]["agent"]["model"] = "fixture-agent"
        raw["profiles"].pop("vlms")
    elif method == "dense":
        raw["profiles"]["vlms"]["qwen_default"]["prompt"] = (
            "Describe {{FRAME_TIMESTAMPS}}"
        )
    path = tmp_path / "compile.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_compile_config(path)
    assert cfg.compile.compiler == method
    assert cfg.query is None


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["compile"],
        ["query"],
        ["evaluate"],
        ["artifact", "validate"],
        ["wikiset", "validate"],
    ],
)
def test_cli_help(command):
    result = subprocess.run(
        [sys.executable, "-m", "vmr", *command, "--help"], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 0
    if command == ["compile"]:
        assert b"configs/compiler/base.yaml" in result.stdout
    elif command == ["query"]:
        assert b"configs/query/base.yaml" in result.stdout


@pytest.mark.parametrize(
    "module",
    [
        "harness.ingest",
        "harness.ingest_all",
        "harness.run_query",
        "harness.run_all_queries",
        "harness.config",
        "harness.freeze",
    ],
)
def test_legacy_cli_help_and_deprecation(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "deprecated" in result.stderr


def test_shipped_compile_entry_extends_agentic(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "fixture-agent")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    raw = yaml.safe_load((ROOT / "configs/compiler/base.yaml").read_text())
    assert "profiles" not in raw
    cfg = load_compile_config(ROOT / "configs/compiler/base.yaml")
    assert cfg.compile.compiler == "agentic"
    assert cfg.dataset.split == "dev"
    assert cfg.query is None
    assert cfg.storage.root == ROOT
    agentic = cfg.compile.method.settings["agentic"]
    assert agentic["agent"] == "claude_code"
    assert agentic["model"] == "fixture-agent"
    assert agentic["templates"] == {
        "agents": "wiki_agents.md",
        "prompt": "wiki_prompt.md",
    }


def test_shipped_query_entry_loads():
    cfg = load_query_config(ROOT / "configs/query/base.yaml")
    assert cfg.query.agent == "claude_code"
    assert cfg.query.agents_template == "query_agents.md"
    assert cfg.query.prompt_template == "query_prompt.md"
    assert cfg.query.text_agents_template == "query_agents_text_only.md"
    assert cfg.query.text_prompt_template == "query_prompt_text_only.md"
    assert cfg.dataset.split == "dev"
    assert cfg.compile is None
    assert cfg.storage.root == ROOT


def test_template_filenames_in_yaml_select_the_markdown_files(tmp_path):
    templates = tmp_path / "templates"
    templates.mkdir()
    files = {
        "custom_agents.md": "custom agents",
        "custom_prompt.md": "custom prompt",
        "custom_text_agents.md": "custom text agents",
        "custom_text_prompt.md": "custom text prompt",
        "custom_wiki_agents.md": "wiki agents",
        "custom_wiki_prompt.md": "wiki prompt",
    }
    for name, body in files.items():
        (templates / name).write_text(body, encoding="utf-8")
    raw = yaml.safe_load((ROOT / "configs/query/base.yaml").read_text())
    raw["storage"]["root"] = str(tmp_path)
    raw["query"]["templates"] = {
        "agents": "custom_agents.md",
        "prompt": "custom_prompt.md",
        "text": {
            "agents": "custom_text_agents.md",
            "prompt": "custom_text_prompt.md",
        },
    }
    path = tmp_path / "query.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_query_config(path)
    assert cfg.query.agents_template == "custom_agents.md"
    assert cfg.query.text_prompt_template == "custom_text_prompt.md"
    from vmr.query.templates import load_query_templates

    loaded = load_query_templates(
        cfg.storage.templates,
        text_only=False,
        templates={
            "agents": cfg.query.agents_template,
            "prompt": cfg.query.prompt_template,
            "text": {
                "agents": cfg.query.text_agents_template,
                "prompt": cfg.query.text_prompt_template,
            },
        },
    )
    assert loaded == {
        "AGENTS.md": "custom agents",
        "query_prompt.md": "custom prompt",
    }
    from vmr.compiler.methods.agentic.config import settings

    resolved = settings(
        {
            "agentic": {
                "model": "m",
                "templates": {
                    "agents": "custom_wiki_agents.md",
                    "prompt": "custom_wiki_prompt.md",
                },
            }
        },
        templates,
    )
    assert resolved["templates"] == {
        "agents": "custom_wiki_agents.md",
        "prompt": "custom_wiki_prompt.md",
    }
    shipped = settings({"agentic": {"model": "m"}}, ROOT / "templates")
    assert resolved["templates_hash"] != shipped["templates_hash"]


@pytest.mark.parametrize(
    "method", ["simple", "dense", "hierarchical", "bidirectional", "agentic"]
)
def test_shipped_method_files_compose_with_compile_entry(tmp_path, method, monkeypatch):
    monkeypatch.setenv("VLM_MODEL", "Qwen3-VL-8B-Instruct")
    monkeypatch.setenv("VLM_BASE_URL", "https://vlm.example.com/v1")
    raw = yaml.safe_load((ROOT / "configs/compiler/base.yaml").read_text())
    method_path = ROOT / "configs/compiler/methods" / f"{method}.yaml"
    if method == "agentic":
        method_raw = yaml.safe_load(method_path.read_text())
        method_raw["profiles"]["agent"]["model"] = "fixture-agent"
        method_path = tmp_path / "agentic.yaml"
        method_path.write_text(yaml.safe_dump(method_raw))
    raw["extends"] = str(method_path)
    raw["storage"]["root"] = str(ROOT)
    path = tmp_path / "compile.yaml"
    path.write_text(yaml.safe_dump(raw))
    cfg = load_compile_config(path)
    assert cfg.compile.compiler == method
    assert "captioner_profile" not in yaml.safe_load(
        (ROOT / "configs/compiler/methods" / f"{method}.yaml").read_text()
    ).get("compile", {})
    if method == "dense":
        assert "{{FRAME_TIMESTAMPS}}" in cfg.compile.method.settings["vlm"]["prompt"]
        assert cfg.compile.method.settings["caption_window_frames"] == 8
    elif method == "simple":
        assert (
            "{{FRAME_TIMESTAMPS}}" not in cfg.compile.method.settings["vlm"]["prompt"]
        )
    elif method == "agentic":
        assert cfg.compile.method.settings["agentic"]["agent"] == "claude_code"
        assert cfg.compile.method.settings["agentic"]["templates"] == {
            "agents": "wiki_agents.md",
            "prompt": "wiki_prompt.md",
        }
    if method != "agentic":
        shipped = yaml.safe_load(
            (ROOT / "configs/compiler/methods" / f"{method}.yaml").read_text()
        )["profiles"]["vlms"]["qwen_default"]
        assert shipped["model"] == "VLM_MODEL"
        assert shipped["base_url"] == "VLM_BASE_URL"
        assert shipped["api_key_env"] == "VLM_API_KEY"
        assert cfg.compile.method.settings["vlm"]["model"] == "Qwen3-VL-8B-Instruct"
        assert (
            cfg.compile.method.settings["vlm"]["base_url"]
            == "https://vlm.example.com/v1"
        )
        assert cfg.compile.method.settings["vlm"]["api_key_env"] == "VLM_API_KEY"


def old_v2(tmp_path):
    raw = yaml.safe_load((ROOT / "tests/compatibility/fixtures/v2.yaml").read_text())
    raw["storage"]["templates"] = str(ROOT / "templates")
    return raw, tmp_path / "v2.yaml"


def test_original_named_v2_loads_in_all_domains(tmp_path):
    from vmr.compat.config import load_config

    raw, path = old_v2(tmp_path)
    path.write_text(yaml.safe_dump(raw))
    legacy = load_config(path)
    assert legacy["query"]["agent"] == "codex"
    assert load_compile_config(path).compile.compiler == "bidirectional"
    assert load_query_config(path).query.runtime_config() == legacy["query"]
    assert (
        load_config(path, query_agent="claude_code")["query"]["agent"] == "claude_code"
    )
    assert (
        load_config(path, query_agent_profile="claude_default")["query"]["agent"]
        == "claude_code"
    )


def test_named_v2_keeps_compile_and_query_deployments_independent(tmp_path):
    raw, path = old_v2(tmp_path)
    raw["profiles"]["agents"]["codex_default"]["model"] = "fixture-query"
    raw["profiles"]["agents"]["claude_default"]["model"] = "fixture-compile"
    raw["wiki"] = dict(
        method="agentic", media=raw["wiki"]["media"], agent_profile="claude_default"
    )
    path.write_text(yaml.safe_dump(raw))
    cfg = load_compile_config(path)
    assert cfg.compile.method.settings["agentic"]["agent"] == "claude_code"
    assert cfg.query.agent == "codex"
    assert cfg.query.model == raw["profiles"]["agents"]["codex_default"]["model"]
    assert (
        cfg.compile.method.settings["agentic"]["model"]
        == raw["profiles"]["agents"]["claude_default"]["model"]
    )
    del raw["wiki"]
    del raw["profiles"]["vlms"]
    path.write_text(yaml.safe_dump(raw))
    assert load_query_config(path).query == cfg.query


@pytest.mark.parametrize("loader", [load_query_config, load_compile_config])
@pytest.mark.parametrize("damage", ["mixed", "missing_key", "mixed_selector"])
def test_named_v2_rejects_ambiguous_or_incomplete_profiles(tmp_path, loader, damage):
    from vmr.core.errors import HarnessError

    raw, path = old_v2(tmp_path)
    if damage == "mixed":
        raw["profiles"]["agent"] = {}
    elif damage == "missing_key":
        del raw["profiles"]["agents"]["codex_default"]["api_key_env"]
    else:
        raw["query"]["kind"] = "claude_code"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError):
        loader(path)


def test_named_v2_override_requires_unambiguous_kind(tmp_path):
    from vmr.compat.config import load_config
    from vmr.core.errors import HarnessError

    raw, path = old_v2(tmp_path)
    raw["profiles"]["agents"]["another_codex"] = dict(
        raw["profiles"]["agents"]["codex_default"]
    )
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="matches 2 profiles"):
        load_config(path, query_agent="codex")
    with pytest.raises(HarnessError, match="different agent kinds"):
        load_config(path, query_agent="codex", query_agent_profile="claude_default")
    assert (
        load_config(path, query_agent="codex", query_agent_profile="another_codex")[
            "query"
        ]["agent"]
        == "codex"
    )


@pytest.mark.parametrize("mode", ["text-only", None, 123, False, "", [], {}])
def test_query_rejects_invalid_input_modes(tmp_path, mode):
    from vmr.core.errors import HarnessError

    raw = yaml.safe_load((ROOT / "configs/query/base.yaml").read_text())
    raw["query"]["input_mode"] = mode
    path = tmp_path / "query.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(HarnessError, match="input_mode"):
        load_query_config(path)


@pytest.mark.parametrize("mode", ["text", "multimodal"])
def test_query_preserves_valid_input_modes(tmp_path, mode):
    raw = yaml.safe_load((ROOT / "configs/query/base.yaml").read_text())
    raw["query"]["input_mode"] = mode
    path = tmp_path / "query.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert load_query_config(path).query.input_mode == mode
