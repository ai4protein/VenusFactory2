import json
from dataclasses import replace
from pathlib import Path

import pytest

from agent.harness.catalog import ExecutionCatalog, free_model
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.prepared_mcp import check_entry
from agent.rsi.engine.openrouter_policy import enforce_free_responses_request, verify_free_responses_response


def test_missing_key_visible_free_key_selectable_and_paid_key_not_enough(tmp_path, monkeypatch):
    binding = ModelBinding("dmx", "glm-4-flash", "https://www.dmxapi.cn/v1", credential_env="WEB_TEST_FREE_KEY")
    profile = Profile("science_agent", "langgraph", binding, Budget(thinking="disabled"))
    catalog_path = tmp_path / "models.json"
    official = replace(
        binding,
        provider="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        credential_env="WEB_TEST_MISSING_KEY",
    )
    catalog_path.write_text(
        json.dumps(
            {
                "models": [
                    {"id": "free", "label": "Free", "binding": binding.__dict__},
                    {"id": "official", "label": "Official", "binding": official.__dict__},
                ]
            }
        )
    )
    runtime = tmp_path / "python"
    runtime.touch()
    monkeypatch.setenv("WEB_TEST_FREE_KEY", "fixture")
    monkeypatch.delenv("WEB_TEST_MISSING_KEY", raising=False)
    cat = ExecutionCatalog(tmp_path, profile, path=catalog_path, free_only=True)

    def choice(data, mid):
        return next(c for c in data["combinations"] if c["harness"] == "langgraph" and c["model_id"] == mid)

    data = cat.public({"python_bin": str(runtime)})
    assert {m["id"] for m in data["models"]} == {"free", "official"}
    assert choice(data, "free")["ready"]
    assert choice(data, "official")["reasons"][0] == "credential_unavailable"
    monkeypatch.setenv("WEB_TEST_MISSING_KEY", "fixture")
    assert choice(cat.public({"python_bin": str(runtime)}), "official")["reasons"] == ["free_model_required"]
    assert "fixture" not in json.dumps(data)


def test_codex_selects_responses_before_frozen_profile(tmp_path, monkeypatch):
    binding = ModelBinding("openrouter", "vendor/model:free", "https://openrouter.ai/api/v1")
    cat = ExecutionCatalog(tmp_path, Profile("science_agent", "langgraph", binding, Budget(thinking="disabled")))
    codex, _ = cat.select({"harness": "codex", "model_id": cat.default_model})
    other, _ = cat.select({"harness": "deepseek-harness", "model_id": cat.default_model})
    assert codex.model.protocol == "responses" and codex.budget.thinking is None
    assert other.model.protocol == "chat_completions" and other.budget.thinking == "disabled"
    assert codex.model.model == other.model.model
    assert not free_model(codex.model)  # A :free suffix alone is not server admission.
    route = tmp_path / "routes.json"
    route.write_text(json.dumps({"models": [{"binding": binding.__dict__, "pricing": {"input_per_million": 0, "output_per_million": 0}, "protocols": ["chat_completions", "responses"], "max_output_tokens": 4096}]}))
    monkeypatch.setenv("VENUS_EXECUTION_CATALOG", str(route))
    assert free_model(codex.model)
    assert not free_model(replace(binding, base_url="https://example.test/v1"))


def test_responses_free_policy_requires_price_receipt_and_no_fallback():
    body = {"model": "vendor/model:free", "input": "hello", "store": False}
    sent, meta = enforce_free_responses_request(body, {})
    assert sent["provider"]["max_price"]["prompt"] == 0
    assert sent["provider"]["allow_fallbacks"] is False
    assert meta["free_route"]
    for change in [
        {"model": "vendor/paid"},
        {"models": ["vendor/paid"]},
        {"provider": {"allow_fallbacks": True}},
        {"service_tier": "priority"},
    ]:
        with pytest.raises(ValueError):
            enforce_free_responses_request({**body, **change}, {})
    response = {"status": "completed", "model": "vendor/model", "usage": {"cost": 0}}
    assert verify_free_responses_response(json.dumps(response).encode(), False, body["model"])["billed_cost"] == 0
    for usage in [{}, {"cost": 0.01}, {"cost": None}]:
        with pytest.raises(ValueError):
            verify_free_responses_response(json.dumps({**response, "usage": usage}).encode(), False, body["model"])


def test_web_proxy_is_prepared_without_eager_science_imports(tmp_path):
    from agent.harness import web_mcp

    path = tmp_path / "web_tools.py"
    path.write_bytes(Path(web_mcp.__file__).read_bytes())
    assert check_entry(path)
    path.write_text(path.read_text().replace("    main()", '    main("unexpected")'))
    with pytest.raises(ValueError):
        check_entry(path)


def test_openrouter_removes_native_display_controls_and_keeps_inference_budget():
    body = {
        "model": "vendor/model:free",
        "input": "hello",
        "store": False,
        "client_metadata": {"client": "codex"},
        "prompt_cache_key": "native-cache",
        "include": ["reasoning.encrypted_content"],
        "parallel_tool_calls": True,
        "reasoning": {"summary": "auto"},
        "max_output_tokens": 1024,
        "temperature": 0.2,
    }
    sent, _ = enforce_free_responses_request(body, {})
    assert not {"client_metadata", "prompt_cache_key", "include", "parallel_tool_calls", "reasoning"} & sent.keys()
    assert sent["max_output_tokens"] == 1024 and sent["temperature"] == 0.2
    assert sent["provider"]["max_price"]["completion"] == 0
    assert body["reasoning"] == {"summary": "auto"}
