import json
from pathlib import Path

import pytest

from agent.harness.catalog import ExecutionCatalog
from agent.harness.contracts import Budget, ModelBinding, Profile


@pytest.mark.parametrize("mode,harness", [
    ("science_agent", "kimi-code"), ("science_agent", "deepseek-harness"),
    ("science_expert", "expert-kimi-code"), ("science_expert", "expert-deepseek-harness"),
])
def test_free_reasoning_models_preserve_mandatory_thinking(mode, harness, tmp_path):
    path = Path(__file__).resolve().parents[2] / "configs/harness/free_execution_catalog.json"
    base = Profile(mode, harness, ModelBinding("dmx", "unisound-u2-flash", "https://www.dmxapi.cn/v1"),
                   Budget(thinking="disabled"))
    catalog = ExecutionCatalog(tmp_path, base, path=path, free_only=True)
    for identity in ["or-recursive-flint-v1-0-free", "or-liquid-lfm-2-5-2-6b-free"]:
        profile, _ = catalog.select({"harness": harness, "model_id": identity})
        assert profile.budget.thinking == "enabled"
        assert profile.model.model.endswith(":free")
    for identity in ["or-thinkingmachines-inkling-free", "or-thinkingmachines-inkling-small-free"]:
        assert identity not in catalog.models
        with pytest.raises(ValueError):
            catalog.select({"harness": harness, "model_id": identity})


@pytest.mark.parametrize("identity,expected", [("dmx-u2-free", 12288), ("dmx-glm4-free", 4096)])
def test_scientist_output_budget_uses_model_ceiling(tmp_path, identity, expected):
    root = Path(__file__).resolve().parents[2]
    base = Profile.from_dict(json.loads((root / "configs/harness/free_science_goal.json").read_text()))
    catalog = ExecutionCatalog(tmp_path, base, path=root / "configs/harness/free_execution_catalog.json", free_only=True)
    profile, _ = catalog.select({"harness": "expert-deepseek-harness", "model_id": identity})
    assert profile.budget.output_tokens == expected
    assert profile.budget.model_calls == base.budget.model_calls
    assert profile.budget.tool_calls == base.budget.tool_calls
