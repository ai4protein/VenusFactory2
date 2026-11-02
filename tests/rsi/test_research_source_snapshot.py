"""Frozen Scientist runs must include prompt assets and reject later changes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
import run_harness_reproduction as reproduction
import run_reproduction_goal as goal_runner

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.contracts import file_hash
from agent.rsi.source_snapshot import research_implementation_paths

ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_imports_scientist_without_original_checkout(tmp_path):
    snapshot = tmp_path / "repository"
    paths = research_implementation_paths(ROOT)
    prompts = list((ROOT / "src/agent/prompts").glob("*.md"))
    assert prompts and set(prompts) <= set(paths)
    assert ROOT / "mcp_tools/tools/runtime_tool_policy.py" in paths
    assert ROOT / "backend/web_v2/__init__.py" in paths
    assert ROOT / "src/constant.json" in paths
    assert ROOT / "configs/rsi/model_access.json" in paths
    assert ROOT / "configs/harness/free_execution_catalog.json" in paths
    for path in paths:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LD_LIBRARY_PATH"}}
    env.update(HOME=str(tmp_path), PYTHONPATH=str(snapshot / "src"), PYTHONDONTWRITEBYTECODE="1")
    code = (
        "from agent.chat_agent import ProteinContextManager, _build_chains_for_llm; "
        "from agent.graph.compile import create_agent_graph; "
        "from pathlib import Path; import agent.prompts; "
        "assert Path(agent.prompts.__file__).resolve().is_relative_to(Path.cwd()); "
        "assert agent.prompts.PLANNER_SYSTEM_PROMPT_TEMPLATE; "
        "from agent.rsi.model_access import authorize_model, POLICY_PATH; "
        "assert POLICY_PATH.resolve().is_relative_to(Path.cwd()); "
        "assert authorize_model(model='glm-4-flash', base_url='https://www.dmxapi.cn/v1', "
        "protocol='chat_completions', output_tokens=4096)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=snapshot, env=env, capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("changed_asset", ["prompt", "free_catalog"])
def test_prompt_and_free_routes_are_bound_and_mutation_is_rejected(tmp_path, monkeypatch, changed_asset):
    root = tmp_path / "repository"
    prompt = root / "src/agent/prompts/principal_investigator.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Original bounded scientific role")
    graph = root / "src/agent/graph/compile.py"
    graph.parent.mkdir(parents=True)
    graph.write_text("# frozen graph fixture")
    catalog = root / "configs/harness/free_execution_catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({"models": []}))
    monkeypatch.setattr(goal_runner, "ROOT", root)
    monkeypatch.setattr(reproduction, "ROOT", root)
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("fixture", "fixture", "https://example.org/v1"),
        Budget(wall_seconds=30, model_calls=4, tool_calls=4),
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile.public()))
    store = goal_runner.prepare(
        SimpleNamespace(
            output=root / "research_runs/rsi/fixture_goal",
            profile=profile_path,
            case="catapro_full_training",
            objective="Fixture: source-bound reproduction",
            max_cycles=2,
            model_calls=8,
            tool_calls=8,
            wall_seconds=120,
        )
    )
    protocol = json.loads((store.root / "protocol.json").read_text())
    name = str(prompt.relative_to(root))
    assert protocol["implementation"][name] == file_hash(prompt)
    assert (store.root / "implementation" / name).read_text() == prompt.read_text()
    assert str(graph.relative_to(root)) in protocol["implementation"]
    goal_runner.ReproductionGoal(store, None)
    assert protocol["implementation"][catalog.relative_to(root).as_posix()] == file_hash(catalog)
    if changed_asset == "prompt":
        prompt.write_text("A different scientific role")
    else:
        catalog.write_text(json.dumps({"models": [{"enabled": False}]}))
    with pytest.raises(ValueError, match="implementation changed"):
        goal_runner.ReproductionGoal(store, None)


def test_snapshot_excludes_credentials_results_and_deployment_config(tmp_path):
    files = [
        "src/agent/prompts/role.md",
        "src/agent/runtime.py",
        "scripts/rsi/run.py",
        "backend/web_v2/queue.py",
        "mcp_tools/tools/policy.py",
        ".env",
        "configs/private.toml",
        "research_runs/rsi/history/model.json",
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    assert {str(p.relative_to(tmp_path)) for p in research_implementation_paths(tmp_path)} == set(files[:5])
