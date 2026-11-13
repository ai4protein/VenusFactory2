from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
from research_source_review_fixture import with_source_review
import asyncio
import copy
from contextlib import asynccontextmanager
from types import SimpleNamespace
import httpx
import pytest
from agent.harness.model_availability import ModelServiceUnavailable
from backend.web_v2 import research_longform as lf
from backend.web_v2 import research_report as report


def saved_report():
    return {
        "workflow": "specialist_research_v2", "phase": "incomplete",
        "collection_complete": True, "collection_partial": False, "partial": True,
        "model_calls": 4, "report": "Previous report", "steps": [], "sections": [],
        "sources": [{"id": "S1", "kind": "calculation", "text": "Measured evidence",
                     "url": "https://example.org", "title": "Evidence", "evidence_level": "calculation"}],
        "protein": {"name": "Protein"},
        "chapters": [
            {"id": identity, "title": title, "status": "partial", "attempts": 0,
             "draft": "Prior draft", "source_ids": ["S1"]}
            for identity, title, *_ in lf.CHAPTERS],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 402, 403, 404, 429])
async def test_service_rejection_stops_later_calls_and_resume_keeps_evidence(monkeypatch, status):
    checkpoint = saved_report()
    original = copy.deepcopy(checkpoint)
    requests = []
    async def rejected(*args, **kwargs):
        requests.append(args)
        await asyncio.sleep(0)  # Both admitted specialists may already be in flight.
        raise ModelServiceUnavailable(status)
    monkeypatch.setattr(lf.research, "complete", rejected)
    payload = {"model_id": "fixture", "analyses": ["properties"], "lang": "en"}
    stopped = await lf.run_long_report(payload, lambda state: None, checkpoint=checkpoint)
    assert checkpoint == original
    assert 1 <= len(requests) <= 2
    assert stopped["model_calls"] == 4 + len(requests)
    assert sum(c["attempts"] for c in stopped["chapters"]) == len(requests)
    assert stopped["stop_reason"] == "model_service"
    assert stopped["execution_blocker"]["http_status"] == status
    assert all(c["draft"] == "Prior draft" for c in stopped["chapters"])
    assert stopped["review_status"] == "unavailable"
    before_resume = copy.deepcopy(stopped)
    resumed_calls = []
    @with_source_review
    async def recovered(*args, **kwargs):
        resumed_calls.append(args)
        return "Measured evidence " * 260 + "[S1]", False
    monkeypatch.setattr(lf.research, "complete", recovered)
    resumed = await lf.run_long_report(payload, lambda state: None, checkpoint=stopped)
    assert stopped == before_resume
    assert len(resumed_calls) == 9 and resumed["completed_chapters"] == 8
    assert resumed["model_calls"] == stopped["model_calls"] + 12
    assert resumed["sources"] == stopped["sources"]
    assert "execution_blocker" not in resumed and "stop_reason" not in resumed
    assert resumed["partial"] is False


@pytest.mark.asyncio
async def test_one_transient_writer_failure_does_not_block_other_chapters(monkeypatch):
    requests = []
    @with_source_review
    async def transient(*args, **kwargs):
        requests.append(args)
        if len(requests) == 1:
            raise ValueError("Model returned HTTP 503")
        return "Measured evidence " * 260 + "[S1]", False
    monkeypatch.setattr(lf.research, "complete", transient)
    result = await lf.run_long_report(
        {"model_id": "fixture", "analyses": ["properties"], "lang": "en"},
        lambda state: None, checkpoint=saved_report())
    assert len(requests) == 9 and result["model_calls"] == 16
    assert result["completed_chapters"] == 7 and result["partial"] is True
    assert "execution_blocker" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("status,kind", [(403, ModelServiceUnavailable), (429, ModelServiceUnavailable), (503, ValueError)])
async def test_http_errors_keep_status_without_exposing_provider_payload(monkeypatch, status, kind):
    import agent.rsi.engine.model_access as access
    import agent.harness.provider_config as providers
    import backend.web_v2.job_queue as queue
    import backend.web_v2.usage_store as usage
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield
    class Observation:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): return False
    option = SimpleNamespace(
        binding=SimpleNamespace(model="fixture", base_url="https://model.invalid/v1", protocol="openai_chat"),
        max_output_tokens=4096, inference_controls={})
    monkeypatch.setattr(report, "model_option", lambda identity: (option, "fixture-credential"))
    monkeypatch.setattr(access, "authorize_model", lambda **kwargs: None)
    monkeypatch.setattr(providers, "adapt_request", lambda payload, *args: payload)
    monkeypatch.setattr(providers, "request_headers", lambda *args: {})
    monkeypatch.setattr(queue, "resource_slot", slot)
    monkeypatch.setattr(usage, "Observation", Observation)
    requests = []
    def handle(request):
        requests.append(request)
        return httpx.Response(status, json={"error": "private upstream payload"})
    real_client = httpx.AsyncClient
    monkeypatch.setattr(report.httpx, "AsyncClient",
                        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(kind) as caught:
        await report.complete("fixture", [{"role": "user", "content": "Private input"}])
    assert len(requests) == 1
    assert "private" not in str(caught.value).lower()
    if status in {403, 429}:
        assert caught.value.diagnostic["http_status"] == status
    else:
        assert not isinstance(caught.value, ModelServiceUnavailable)
