"""Configuration failures stop review locally without consuming the remaining budget."""
import copy
import json

import httpx
import pytest

from backend.web_v2 import research_longform as longform
from backend.web_v2 import research_report as report
from backend.web_v2 import research_review_binding as binding
from backend.web_v2 import research_source_review as review
from backend.web_v2.usage_store import UsageStore
from mcp_tools.tools.capability_policy import CapabilityStore
from research_reviewer_fixture import install_catalog
from test_research_source_review_flow import SOURCE, FixtureModel, review_response


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["route", "disabled", "credential", "paid"])
async def test_configuration_change_stops_once_and_original_binding_resumes(tmp_path, monkeypatch, failure):
    catalog = install_catalog(monkeypatch, tmp_path)
    original_catalog = catalog.read_bytes()
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT", "12")
    request = binding.bind_payload({"model_id": "writer", "reviewer_model_id": "reviewer",
                                    "analyses": ["properties"], "sequence": "MKTAYIAKQRQISFVK"})
    original_request = copy.deepcopy(request)
    fixture = FixtureModel()
    real_complete = report.complete
    counts = {"writer": 0, "reviewer": 0, "http": 0, "collection": 0}
    restored = False
    switch_key = None
    switch_revision = None

    async def collect(*args, **kwargs):
        counts["collection"] += 1
        return {"sources": [copy.deepcopy(SOURCE)], "steps": [], "sections": [],
                "protein": {"name": "Fixture"}, "predictions": [{"task_id": "preserved-fixture", "status": "completed",
                 "task": "Fixture property", "model": "Fixture", "dataset": "Synthetic", "values": {"value": 1}}]}

    async def complete(identity, messages, tokens, **metadata):
        nonlocal switch_key, switch_revision
        if messages[0]["content"] != review.SYSTEM:
            counts["writer"] += 1
            return await fixture(identity, messages, tokens)
        counts["reviewer"] += 1
        assert identity == "reviewer"
        assert metadata["review_binding"] == original_request["reviewer_binding"]
        if restored:
            # Real admission still validates the original binding; only provider output is doubled.
            binding.resolve_reviewer(identity, metadata["review_binding"])
            return await fixture(identity, messages, tokens)
        if failure == "credential":
            monkeypatch.delenv("VENUS_TEST_REVIEW_KEY")
        elif failure == "disabled":
            from agent.harness.llm_policy import route_key
            switch_key = route_key("reviewer-model", "https://models.example.test/v1")
            switch_revision = CapabilityStore().set(switch_key, False, 0, {}, "fixture")
        else:
            data = json.loads(catalog.read_text())
            if failure == "route":
                data["models"][1]["binding"]["model"] = "changed-reviewer-model"
            else:
                data["models"][1]["pricing"]["output_per_million"] = 1
            catalog.write_text(json.dumps(data))
        return await real_complete(identity, messages, tokens, **metadata)

    def forbidden_http(*args, **kwargs):
        counts["http"] += 1
        pytest.fail("Configuration rejection must happen before HTTP")

    monkeypatch.setattr(report, "run_research", collect)
    monkeypatch.setattr(report, "complete", complete)
    monkeypatch.setattr(httpx, "AsyncClient", forbidden_http)
    first = await longform.run_long_report(request, lambda state: None)
    assert counts == {"writer": 8, "reviewer": 1, "http": 0, "collection": 1}
    assert first["model_calls"] == 9 and first["model_call_limit"] == 12
    assert first["phase"] == "incomplete" and first["stop_reason"] == "reviewer_configuration"
    blocker = first["execution_blocker"]
    assert blocker["source"] == "reviewer_configuration" and blocker["request_dispatched"] is False
    assert "http_status" not in blocker
    attempts = first["source_review"]["attempts"]
    assert len(attempts) == 1 and attempts[0]["status"] == "unavailable"
    assert attempts[0]["error"] == "ReviewerBindingUnavailable"
    assert "recovery" not in attempts[0] and "usage_event_id" not in attempts[0]
    assert not first["source_review"]["units"]
    assert not UsageStore().events()["items"]
    with pytest.raises(binding.ReviewerBindingUnavailable):
        binding.bind_payload(request)

    catalog.write_bytes(original_catalog)
    monkeypatch.setenv("VENUS_TEST_REVIEW_KEY", "offline-fixture-only")
    if switch_key:
        CapabilityStore().set(switch_key, True, switch_revision, {}, "fixture")
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "other")
    restored = True
    resumed = await longform.run_long_report(request, lambda state: None, checkpoint=first)
    assert resumed["phase"] == "completed" and resumed["model_calls"] == 12
    assert counts == {"writer": 9, "reviewer": 3, "http": 0, "collection": 1}
    assert resumed["model_call_limit"] == first["model_call_limit"]
    assert resumed["predictions"] == first["predictions"]
    assert resumed["reviewer_binding"] == first["reviewer_binding"]
    assert not resumed.get("execution_blocker")
    assert request == original_request


@pytest.mark.asyncio
async def test_direct_review_does_not_split_a_configuration_failure():
    state = {"sources": [{"id": "S1", "text": "Measured observation."}],
             "model_calls": 0, "model_call_limit": 10}
    document = {"id": "one", "title": "One", "text": "Measured observation. [S1]"}
    async def model(*args, **kwargs):
        state["model_calls"] += 1
        raise binding.ReviewerBindingUnavailable("route_changed")
    receipts = await review.review_documents(state, {"model_id": "fixture"}, [document], model, lambda: None)
    assert state["model_calls"] == 1
    assert receipts["one"]["status"] == "partial"
    assert len(state["source_review"]["attempts"]) == 1


@pytest.mark.asyncio
async def test_content_validation_error_remains_repairable():
    state = {"sources": [{"id": "S1", "text": "Measured observation."}],
             "model_calls": 0, "model_call_limit": 2}
    document = {"id": "one", "title": "One", "text": "Measured observation. [S1]"}
    async def model(identity, messages, tokens, **kwargs):
        state["model_calls"] += 1
        if state["model_calls"] == 1:
            raise ValueError("Invalid source-review JSON")
        return review_response(json.loads(messages[1]["content"])), False
    receipts = await review.review_documents(state, {"model_id": "fixture"}, [document], model, lambda: None)
    assert state["model_calls"] == 2 and receipts["one"]["status"] == "completed"


def test_local_diagnostic_does_not_reveal_underlying_configuration(monkeypatch):
    def unavailable(_):
        raise ValueError("synthetic-private-key-marker")
    monkeypatch.setattr(report, "model_option", unavailable)
    with pytest.raises(binding.ReviewerBindingUnavailable) as failure:
        binding.resolve_reviewer("reviewer")
    assert "synthetic-private-key-marker" not in str(failure.value)
    assert "synthetic-private-key-marker" not in json.dumps(failure.value.diagnostic)
    assert failure.value.diagnostic["request_dispatched"] is False


@pytest.mark.parametrize("url", ["https://name:canary@models.example.test/v1",
                                "https://models.example.test/v1?key=canary",
                                "https://models.example.test/v1#canary"])
def test_credential_bearing_urls_cannot_enter_public_binding(tmp_path, monkeypatch, url):
    catalog = install_catalog(monkeypatch, tmp_path)
    data = json.loads(catalog.read_text())
    data["models"][1]["binding"]["base_url"] = url
    catalog.write_text(json.dumps(data))
    with pytest.raises(binding.ReviewerBindingUnavailable) as failure:
        binding.resolve_reviewer("reviewer")
    assert "canary" not in str(failure.value)
