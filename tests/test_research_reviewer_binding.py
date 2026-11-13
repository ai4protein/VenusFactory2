"""Bound reviewer routes, cache generations and real HTTP/usage accounting (offline)."""
import copy
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from backend.web_v2 import research_report as report
from backend.web_v2 import research_review_binding as binding
from backend.web_v2 import research_source_review as review
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_jobs import ResearchJobs, write
from backend.web_v2.usage_store import UsageStore
from mcp_tools.tools.capability_policy import CapabilityStore
from research_reviewer_fixture import install_catalog


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    return install_catalog(monkeypatch, tmp_path)


def payload(**extra):
    return {"model_id": "writer", "analyses": ["properties"], "sequence": "MKTAYIAKQRQISFVK", **extra}


def change_catalog(path, change):
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_new_default_is_bound_and_resume_ignores_new_default(catalog, monkeypatch):
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "reviewer")
    saved = binding.bind_payload(payload(), new=True)
    assert saved["reviewer_model_id"] == "reviewer"
    assert "offline-fixture" not in json.dumps(saved)
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "other")
    assert binding.bind_payload(saved) == saved
    assert binding.bind_payload(payload(), new=False)["reviewer_model_id"] == "writer"
    assert binding.bind_payload(payload(reviewer_model_id="other"), new=True)["reviewer_model_id"] == "other"


def test_catalog_provenance_change_does_not_change_route(catalog):
    saved = binding.bind_payload(payload(reviewer_model_id="reviewer"))
    change_catalog(catalog, lambda data: data["models"][0].update(label="Writer label changed"))
    _, _, current = binding.resolve_reviewer("reviewer", saved["reviewer_binding"])
    assert current["sha256"] == saved["reviewer_binding"]["sha256"]
    assert current["catalog_sha256"] != saved["reviewer_binding"]["catalog_sha256"]
    change_catalog(catalog, lambda data: data["models"][1]["binding"].update(model="different-model"))
    with pytest.raises(ValueError, match="route changed"):
        binding.resolve_reviewer("reviewer", saved["reviewer_binding"])


@pytest.mark.parametrize("failure", ["disabled", "paid", "credential", "protocol", "unknown"])
def test_reviewer_must_be_currently_enabled_free_configured(catalog, monkeypatch, failure):
    saved = binding.bind_payload(payload(reviewer_model_id="reviewer"))
    if failure == "disabled":
        from agent.harness.llm_policy import route_key
        CapabilityStore().set(route_key("reviewer-model", "https://models.example.test/v1"), False, 0, {}, "test")
    elif failure == "paid":
        change_catalog(catalog, lambda data: data["models"][1]["pricing"].update(output_per_million=1))
    elif failure == "credential":
        monkeypatch.delenv("VENUS_TEST_REVIEW_KEY")
    elif failure == "protocol":
        change_catalog(catalog, lambda data: data["models"][1]["binding"].update(protocol="responses"))
    else:
        saved["reviewer_model_id"] = "unknown"
    with pytest.raises(ValueError):
        binding.bind_payload(saved)


def test_new_job_persists_binding_before_queue_and_retries_remain_idempotent(catalog, tmp_path, monkeypatch):
    jobs = ResearchJobs(tmp_path / "reports", JobQueue(tmp_path / "queue"))
    calls = []
    def launch(command, **kwargs):
        calls.append(command)
        return jobs.queue.submit(kwargs["owner"], kwargs["kind"], kwargs["resource"]), None
    monkeypatch.setattr("backend.web_v2.research_jobs.launch", launch)
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "reviewer")
    request = payload(request_id="same-request", reviewer_binding={"forged": True})
    result = jobs.start("alice", copy.deepcopy(request))
    saved = jobs.read(result["id"])
    assert saved["payload"]["reviewer_model_id"] == "reviewer"
    assert saved["payload"]["reviewer_binding"]["schema"] == binding.SCHEMA
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "not-available")
    assert jobs.start("alice", copy.deepcopy(request))["id"] == result["id"]
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError):
        jobs.get("bob", result["id"])


def test_legacy_resume_preserves_inputs_and_budget_but_binds_original_writer(catalog, tmp_path, monkeypatch):
    jobs = ResearchJobs(tmp_path / "reports", JobQueue(tmp_path / "queue"))
    identity = "a" * 32
    qid = jobs.queue.submit("alice", "protein-research", "research")
    jobs.queue.finish(qid, "failed")
    write(jobs.folder(identity) / "request.json", {"id": identity, "owner": "alice", "created": 1,
          "title": "Legacy", "queue_id": qid, "payload": payload()})
    old = {"model_calls": 4, "model_call_limit": 48, "report": "Preserved", "predictions": [{"id": "original"}]}
    write(jobs.folder(identity) / "state.json", old)
    jobs.records.register("report", identity, "alice", "research", "Legacy", 1)
    monkeypatch.setenv("VENUS_REPORT_REVIEWER_MODEL_ID", "reviewer")
    monkeypatch.setattr("backend.web_v2.research_jobs.launch", lambda command, **kw: (jobs.queue.submit(kw["owner"], kw["kind"], kw["resource"]), None))
    jobs.resume("alice", identity)
    assert jobs.read(identity)["payload"]["reviewer_model_id"] == "writer"
    assert json.loads((jobs.folder(identity) / "state.json").read_text()) == old
    assert list((jobs.folder(identity) / "attempts").glob("*.json"))


def test_legacy_review_generation_is_archived_without_recomputing_results(catalog):
    doc = {"id": "one", "title": "One", "text": "Measured observation. [S1]"}
    sources = [{"id": "S1", "text": "Measured observation."}]
    units = review.units(doc, sources)
    cached = {unit["id"]: {"unit_id": unit["id"], "input_sha256": unit["input_sha256"], "passed": True} for unit in units}
    receipt = review.document_receipt(doc, sources, cached)
    assert review.document_ready(doc, sources, receipt)  # Historical receipt remains readable.
    state = {"model_calls": 9, "predictions": [{"real": "receipt"}], "collection_complete": True,
             "sources": sources, "chapters": [{**doc, "draft": doc["text"], "status": "completed", "source_review": receipt}],
             "source_review": {"policy": review.VERSION, "units": cached, "attempts": [{"model_id": "writer"}]},
             "review": "Saved editorial draft", "review_source_review": receipt}
    before = copy.deepcopy(state)
    expected = binding.bind_payload(payload(reviewer_model_id="reviewer"))["reviewer_binding"]
    binding.align_review_state(state, expected)
    assert state["source_review_history"] == [before["source_review"]]
    assert "reviewer_binding_sha256" not in state["source_review_history"][0]
    assert state["predictions"] == before["predictions"] and state["model_calls"] == 9 and state["collection_complete"]
    assert state["chapters"][0]["draft"] == doc["text"] and state["chapters"][0]["status"] == "reviewing"
    assert not review.document_ready(doc, sources, receipt, expected["sha256"])


@pytest.mark.asyncio
async def test_complete_resolves_once_uses_that_binding_and_records_one_usage(catalog, tmp_path, monkeypatch):
    selected = binding.bind_payload(payload(reviewer_model_id="reviewer"))["reviewer_binding"]
    actual_option = report.model_option
    resolves = []
    def resolve_once(identity):
        resolves.append(identity)
        if len(resolves) > 1:
            raise AssertionError("The effective route was resolved twice")
        return actual_option(identity)
    monkeypatch.setattr(report, "model_option", resolve_once)
    requests = []
    def transport(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "reviewer-model" and request.url.host == "models.example.test"
        return httpx.Response(200, json={"choices": [{"message": {"content": "checked"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}})
    client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: client(transport=httpx.MockTransport(transport), **kw))
    @asynccontextmanager
    async def slot(*args, **kwargs):
        yield
    monkeypatch.setattr("backend.web_v2.job_queue.resource_slot", slot)
    q = JobQueue(tmp_path / "queue")
    qid = q.submit("alice", "protein-research", "research", metadata={"mode": "researcher", "report_id": "test-report"})
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID", qid)
    observed = []
    assert await report.complete("reviewer", [{"role": "user", "content": "Offline test"}], 10,
                                 review_binding=selected, on_observation=observed.append) == ("checked", False)
    assert len(resolves) == len(requests) == len(observed) == 1
    event = UsageStore().events()["items"][0]
    assert event["id"] == observed[0]["usage_event_id"]
    assert event["name"] == "reviewer-model" and event["mode"] == "researcher" and event["total_tokens"] == 7
    assert observed[0]["effective_binding"]["sha256"] == selected["sha256"]


@pytest.mark.asyncio
async def test_report_uses_separate_reviewer_and_reaudits_without_repeating_predictions(catalog, monkeypatch):
    from backend.web_v2 import research_longform as longform
    from test_research_source_review_flow import SOURCE, FixtureModel

    model = FixtureModel()
    calls, collections = [], []
    async def collect(*args, **kwargs):
        collections.append(True)
        return {"sources": [copy.deepcopy(SOURCE)], "steps": [], "sections": [],
                "protein": {"name": "Fixture protein"}, "predictions": [{"task_id": "original-prediction", "status": "completed", "task": "Fixture property", "model": "Fixture", "dataset": "Synthetic", "values": {"value": 1}}]}
    async def complete(identity, messages, tokens, **metadata):
        calls.append(identity)
        if messages[0]["content"] == review.SYSTEM:
            assert identity in {"reviewer", "other"}
            expected = metadata["review_binding"]
            assert expected["route"]["option_id"] == identity
            metadata["on_observation"]({"usage_event_id": "fixture-" + str(len(calls)), "effective_binding": expected})
        else:
            assert identity == "writer"
        return await model(identity, messages, tokens)
    monkeypatch.setattr(report, "run_research", collect)
    monkeypatch.setattr(report, "complete", complete)
    request = binding.bind_payload(payload(reviewer_model_id="reviewer"))
    first = await longform.run_long_report(request, lambda state: None)
    assert first["phase"] == "completed" and first["completed_chapters"] == 8
    assert calls.count("writer") == 9 and calls.count("reviewer") == 2
    assert all(a["usage_event_id"].startswith("fixture-") for a in first["source_review"]["attempts"])
    original_calls = len(calls)
    same = await longform.run_long_report(request, lambda state: None, checkpoint=first)
    assert len(calls) == original_calls and same["model_calls"] == first["model_calls"]
    other_request = binding.bind_payload(payload(reviewer_model_id="other"))
    changed = await longform.run_long_report(other_request, lambda state: None, checkpoint=first)
    assert changed["phase"] == "completed" and calls.count("writer") == 9
    assert calls[-2:] == ["other", "other"] and len(collections) == 1
    assert changed["predictions"] == first["predictions"]
    assert changed["source_review_history"] == [first["source_review"]]
    assert changed["model_calls"] == first["model_calls"] + 2
    assert all(c["source_review"]["reviewer_binding_sha256"] == other_request["reviewer_binding"]["sha256"] for c in changed["chapters"])


@pytest.mark.asyncio
async def test_source_review_exact_quotes_and_shared_budget_remain_strict(catalog, monkeypatch):
    from backend.web_v2 import research_longform as longform
    from test_research_source_review_flow import SOURCE, FixtureModel

    model = FixtureModel(bad_chapter=True)
    async def collect(*args, **kwargs):
        return {"sources": [copy.deepcopy(SOURCE)], "steps": [], "sections": [], "protein": {"name": "Fixture"}}
    async def complete(identity, messages, tokens, **metadata):
        return await model(identity, messages, tokens)
    monkeypatch.setattr(report, "run_research", collect)
    monkeypatch.setattr(report, "complete", complete)
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT", "9")
    request = payload(reviewer_model_id="reviewer")
    state = await longform.run_long_report(request, lambda state: None)
    assert state["phase"] == "incomplete" and state["model_calls"] == 9
    assert any(row["verdict"] == "unsupported" and not row["passed"] for row in state["source_review"]["units"].values())
    again = await longform.run_long_report(request, lambda state: None, checkpoint=state)
    assert again["model_calls"] == 9 and len(model.calls) == 9 and again["partial"]


@pytest.mark.asyncio
async def test_per_unit_identity_mismatch_is_reaudited_not_reused(catalog):
    from test_research_source_review_flow import review_response

    document = {"id": "one", "title": "One", "text": "Measured observation. [S1]"}
    sources = [{"id": "S1", "text": "Measured observation."}]
    selected = binding.bind_payload(payload(reviewer_model_id="reviewer"))["reviewer_binding"]
    unit = review.units(document, sources)[0]
    cached = {unit["id"]: {"unit_id": unit["id"], "input_sha256": unit["input_sha256"],
                           "reviewer_binding_sha256": "different-reviewer", "passed": True}}
    state = {"sources": sources, "reviewer_binding": selected, "model_calls": 0, "model_call_limit": 1,
             "source_review": {"policy": review.VERSION, "reviewer_binding_sha256": selected["sha256"],
                               "units": cached, "attempts": []}}
    calls = []
    async def model(identity, messages, tokens, **metadata):
        calls.append(identity)
        state["model_calls"] += 1
        return review_response(json.loads(messages[1]["content"])), False
    receipts = await review.review_documents(state, payload(), [document], model, lambda: None)
    assert calls == ["reviewer"] and receipts["one"]["status"] == "completed"
    assert state["source_review"]["units"][unit["id"]]["reviewer_binding_sha256"] == selected["sha256"]


def test_optional_reviewer_api_accepts_only_catalog_ids():
    from pydantic import ValidationError

    from backend.web_v2.report_api import ResearchRequest

    assert ResearchRequest(**payload()).reviewer_model_id is None
    assert ResearchRequest(**payload(reviewer_model_id="reviewer")).reviewer_model_id == "reviewer"
    for value in ("https://untrusted.test", "../reviewer", "x" * 81, ""):
        with pytest.raises(ValidationError):
            ResearchRequest(**payload(reviewer_model_id=value))
