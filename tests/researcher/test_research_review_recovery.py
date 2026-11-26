"""Review transport recovery must preserve source verification and total budgets."""
import asyncio
import copy
import json

import httpx
import pytest

from backend.web_v2 import research_source_review as sr
from backend.web_v2 import research_source_validation as validation

SOURCE = {"id": "S1", "kind": "database", "title": "Annotation",
          "text": "The initiator methionine is removed.", "url": "https://example.org/entry"}


def documents(count):
    return [{"id": f"chapter_{i}", "title": f"Chapter {i}",
             "text": "The initiator methionine is removed. [S1]"} for i in range(count)]


def state(limit=18):
    return {"sources": [copy.deepcopy(SOURCE)], "model_calls": 0, "model_call_limit": limit}


def answer(request):
    passage = request["sources"][0]["passages"][0]
    return json.dumps({"assessments": [
        {"unit_id": unit["id"], "classification": "factual", "verdict": "supported",
         "issues": [], "evidence": [{"source_id": "S1", "passage_id": passage["id"], "quote": passage["text"]}]}
        for unit in request["units"]]})


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "truncated", "malformed"])
async def test_failed_batch_is_split_after_independent_packets_without_rewriting(failure, monkeypatch):
    monkeypatch.setattr(sr, "MAX_UNITS", 2)
    current, docs, calls = state(), documents(4), []
    original = copy.deepcopy(docs)

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        calls.append([unit["document_id"] for unit in request["units"]])
        if len(calls) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("fixture")
            return ("not JSON" if failure == "malformed" else answer(request)), failure == "truncated"
        return answer(request), False

    result = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert calls == [["chapter_0", "chapter_1"], ["chapter_2", "chapter_3"], ["chapter_0"], ["chapter_1"]]
    assert all(sr.document_ready(doc, current["sources"], result[doc["id"]]) for doc in docs)
    assert docs == original and current["model_calls"] == 4
    attempts = current["source_review"]["attempts"]
    assert attempts[0]["status"] == "invalid" and attempts[0]["recovery"] == "split"
    assert attempts[2]["parent_attempt"] == attempts[3]["parent_attempt"] == 0
    assert current["source_review"]["pending"] == 0


@pytest.mark.asyncio
async def test_persistent_single_unit_failure_does_not_starve_another_chapter_or_hide_pending(monkeypatch):
    monkeypatch.setattr(sr, "MAX_UNITS", 1)
    current, docs, calls = state(), documents(2), []
    failed = True

    async def complete(model, messages, tokens):
        nonlocal failed
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        identity = request["units"][0]["document_id"]
        calls.append(identity)
        if failed and identity == "chapter_0":
            raise httpx.ReadTimeout("fixture")
        return answer(request), False

    first = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert calls == ["chapter_0", "chapter_1", "chapter_0", "chapter_0"]
    assert not sr.document_ready(docs[0], current["sources"], first["chapter_0"])
    assert sr.document_ready(docs[1], current["sources"], first["chapter_1"])
    assert current["source_review"]["pending"] == 1 and current["source_review"]["status"] == "partial"
    assert current["source_review"]["attempts"][-1]["recovery"] == "exhausted"
    failed = False
    second = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert calls[-1] == "chapter_0" and calls.count("chapter_1") == 1
    assert current["model_calls"] == 5 and all(row["status"] == "completed" for row in second.values())
    assert current["source_review"]["attempts"][0]["status"] == "invalid"


@pytest.mark.asyncio
@pytest.mark.parametrize("blocker", [False, True])
async def test_recovery_obeys_total_budget_and_provider_blocker(monkeypatch, blocker):
    monkeypatch.setattr(sr, "MAX_UNITS", 1)
    current, docs = state(limit=2), documents(2)

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        if current["model_calls"] == 1:
            if blocker:
                current["execution_blocker"] = {"status": 429}
            raise httpx.ReadTimeout("fixture")
        return answer(request), False

    result = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert current["model_calls"] == (1 if blocker else 2)
    assert result["chapter_0"]["status"] == "partial"
    assert result["chapter_1"]["status"] == ("partial" if blocker else "completed")
    assert current["source_review"]["status"] == "partial"


@pytest.mark.asyncio
async def test_unsupported_claims_are_not_retried_until_a_writer_changes_them():
    current, docs = state(), documents(1)

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        response = json.loads(answer(request))
        response["assessments"][0].update(verdict="unsupported", issues=[
            {"claim": "methionine", "reason": "Fixture review rejects the interpretation."}])
        return json.dumps(response), False

    first = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert current["model_calls"] == 1 and first["chapter_0"]["status"] == "partial"
    assert current["source_review"]["units"]["chapter_0:1"]["passed"] is False

@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "unanchored"])
async def test_valid_units_are_retained_while_only_unverified_units_retry(failure):
    current, docs, calls = state(), documents(2), []

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        calls.append([unit["document_id"] for unit in request["units"]])
        response = json.loads(answer(request))
        if len(calls) == 1:
            if failure == "missing":
                response["assessments"].pop()
            else:
                response["assessments"][1]["evidence"] = []
        else:
            assert current["source_review"]["units"]["chapter_0:1"]["passed"]
            assert "chapter_1:1" not in current["source_review"]["units"]
        return json.dumps(response), False

    result = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert calls == [["chapter_0", "chapter_1"], ["chapter_1"]]
    assert all(sr.document_ready(doc, current["sources"], result[doc["id"]]) for doc in docs)
    first = current["source_review"]["attempts"][0]
    assert first["status"] == "partial" and first["accepted_units"] == 1
    assert set(first["validation_errors"]) == {"chapter_1:1"}


@pytest.mark.parametrize("failure", ["duplicate", "unknown"])
def test_ambiguous_batch_cannot_salvage_an_apparently_supported_unit(failure):
    docs = documents(2)
    request = sr.packet([unit for doc in docs for unit in sr.units(doc, [SOURCE])], [SOURCE])
    response = json.loads(answer(request))
    response["assessments"][1]["unit_id"] = "chapter_0:1" if failure == "duplicate" else "foreign:1"
    with pytest.raises(ValueError, match="identities"):
        validation.partial_assessments(json.dumps(response), request, [SOURCE])


def test_partial_assessment_cannot_split_away_an_unsupported_claim_within_one_unit():
    doc = {"id":"mixed", "title":"Mixed", "text":"The initiator methionine is removed; a signal peptide is removed. [S1]"}
    request = sr.packet(sr.units(doc, [SOURCE]), [SOURCE])
    response = json.loads(answer(request))
    response["assessments"][0]["issues"] = [{"claim":"signal peptide","reason":"Not established by the annotation."}]
    checked, invalid = validation.partial_assessments(json.dumps(response), request, [SOURCE])
    assert not checked and set(invalid) == {"mixed:1"}


@pytest.mark.parametrize("claim,quoted", [
    ("The structure resolution is 1.7 Å. [S1]", "The resolution is 1.8 Å."),
    ("The protein contains 152 residues. [S1]", "The mature chain contains 147 residues."),
    ("The measured yield is 80%. [S1]", "The protein is called Protein X."),
])
def test_review_cannot_certify_measured_values_absent_from_its_quotes(claim,quoted):
    source={**SOURCE,"text":quoted}
    doc={"id":"value","title":"Measurement","text":claim}
    request=sr.packet(sr.units(doc,[source]),[source])
    with pytest.raises(ValueError,match="measured value"):
        validation.validate_assessments(answer(request),request,[source])


def test_measurement_guard_accepts_identical_values_with_decimal_formatting():
    source={**SOURCE,"text":"The structure resolution is 1.700 Å."}
    doc={"id":"value","title":"Measurement","text":"The structure resolution is 1.7 Å. [S1]"}
    request=sr.packet(sr.units(doc,[source]),[source])
    assert validation.validate_assessments(answer(request),request,[source])[0]["passed"]


def test_measurement_guard_allows_values_attached_to_source_units():
    source={**SOURCE,"text":"The binding affinity is 80nM."}
    doc={"id":"value","title":"Measurement","text":"The binding affinity is 80 nM. [S1]"}
    request=sr.packet(sr.units(doc,[source]),[source])
    assert validation.validate_assessments(answer(request),request,[source])[0]["passed"]

@pytest.mark.asyncio
async def test_prior_review_policy_is_preserved_but_cannot_certify_current_documents(monkeypatch):
    current, docs = state(), documents(1)
    policy = sr.VERSION

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        return answer(json.loads(messages[1]["content"])), False

    monkeypatch.setattr(sr, "VERSION", "source_review.v1")
    old = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    previous = copy.deepcopy(current["source_review"])
    monkeypatch.setattr(sr, "VERSION", policy)
    assert policy != "source_review.v1"
    assert not sr.document_ready(docs[0], current["sources"], old["chapter_0"])
    new = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert current["model_calls"] == 2
    assert current["source_review_history"] == [previous]
    assert sr.document_ready(docs[0], current["sources"], new["chapter_0"])


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", [False, True])
async def test_interrupted_review_is_terminal_and_resume_retains_calls_and_accepted_units(monkeypatch, deadline):
    monkeypatch.setattr(sr, "MAX_UNITS", 1)
    current, docs, calls = state(limit=3), documents(2), []
    waiting = asyncio.Event()
    interrupt = True
    published = []

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        identity = request["units"][0]["document_id"]
        calls.append(identity)
        if identity == "chapter_1" and interrupt:
            waiting.set()
            await asyncio.Future()
        return answer(request), False

    def publish():
        published.append(copy.deepcopy(current))

    task = asyncio.create_task(sr.review_documents(current, {"model_id": "fixture"}, docs, complete, publish))
    await asyncio.wait_for(waiting.wait(), timeout=1)
    if deadline:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(task, timeout=0.01)
    else:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    review = current["source_review"]
    assert review["status"] == "interrupted"
    attempt = copy.deepcopy(review["attempts"][-1])
    assert attempt["status"] == "interrupted" and attempt["error"] == "CancelledError"
    assert attempt["finished"] >= attempt["started"] and attempt["model_call"] == 2
    assert current["model_calls"] == 2 and current["model_call_limit"] == 3
    assert review["pending"] == 1 and len(review["units"]) == 1
    assert review["documents"]["chapter_0"]["status"] == "completed"
    assert review["documents"]["chapter_1"]["status"] == "partial"
    assert published[-1]["source_review"]["attempts"][-1] == attempt
    assert sr.document_receipt(docs[0], current["sources"], review["units"])["status"] == "completed"
    assert sr.document_receipt(docs[1], current["sources"], review["units"])["status"] == "partial"
    interrupt = False
    receipts = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, publish)
    assert calls == ["chapter_0", "chapter_1", "chapter_1"]
    assert current["model_calls"] == 3 and current["model_call_limit"] == 3
    assert review["attempts"][1] == attempt
    assert all(sr.document_ready(doc, current["sources"], receipts[doc["id"]]) for doc in docs)


@pytest.mark.parametrize("encoding", ["object", "array", "wrapped", "fenced_wrapped"])
def test_unambiguous_json_envelopes_preserve_every_source_check(encoding):
    request = sr.packet(sr.units(documents(1)[0], [SOURCE]), [SOURCE])
    original = json.loads(answer(request))
    value = original["assessments"] if encoding == "array" else [original] if "wrapped" in encoding else original
    response = json.dumps(value)
    if encoding == "fenced_wrapped":
        response = "```json\n" + response + "\n```"
    checked = validation.validate_assessments(response, request, [SOURCE])
    assert len(checked) == 1 and checked[0]["passed"] is True
    assert checked[0]["evidence"][0]["quote"] == SOURCE["text"]
    assert checked[0]["input_sha256"] == request["units"][0]["input_sha256"]


@pytest.mark.parametrize("encoding", ["array", "wrapped"])
@pytest.mark.parametrize("failure", ["quote", "measurement", "unknown", "duplicate_unit"])
def test_relaxed_envelope_never_relaxes_source_or_unit_validation(encoding, failure):
    source = {**SOURCE, "text": "The resolution is 1.8 A."} if failure == "measurement" else SOURCE
    doc = {**documents(1)[0], "text": "The resolution is 1.7 A. [S1]"} if failure == "measurement" else documents(1)[0]
    request = sr.packet(sr.units(doc, [source]), [source])
    original = json.loads(answer(request))
    if failure == "quote": original["assessments"][0]["evidence"][0]["quote"] = "Fabricated annotation."
    if failure == "unknown": original["assessments"][0]["unit_id"] = "foreign:1"
    if failure == "duplicate_unit": original["assessments"] *= 2
    response = json.dumps(original["assessments"] if encoding == "array" else [original])
    with pytest.raises(ValueError):
        validation.validate_assessments(response, request, [source])
    checked, invalid = [], {}
    try:
        checked, invalid = validation.partial_assessments(response, request, [source])
    except ValueError:
        pass
    assert not checked


@pytest.mark.parametrize("where", ["root", "assessment", "evidence"])
def test_duplicate_json_keys_cannot_override_a_rejection_or_source_identity(where):
    request = sr.packet(sr.units(documents(1)[0], [SOURCE]), [SOURCE])
    response = answer(request)
    if where == "root":
        response = response.replace('{"assessments":', '{"assessments": [], "assessments":', 1)
    elif where == "assessment":
        response = response.replace('"verdict": "supported"', '"verdict": "unsupported", "verdict": "supported"', 1)
    else:
        response = response.replace('"source_id": "S1"', '"source_id": "foreign", "source_id": "S1"', 1)
    with pytest.raises(ValueError, match="Duplicate"):
        validation.validate_assessments(response, request, [SOURCE])
    with pytest.raises(ValueError, match="Duplicate"):
        validation.partial_assessments(response, request, [SOURCE])


@pytest.mark.asyncio
async def test_recovery_returns_each_invalid_units_own_error_without_repeating_the_failed_answer():
    current, docs, requests = state(), documents(2), []
    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"])
        requests.append(request)
        response = json.loads(answer(request))
        if len(requests) == 1:
            response["assessments"][0]["evidence"][0]["quote"] = "Invented quote from the previous draft."
            response["assessments"][1]["evidence"] = []
        else:
            identity = request["units"][0]["id"]
            correction = request["correction"]
            assert set(correction["unit_errors"]) == {identity}
            assert ("Quote is not" if identity == "chapter_0:1" else "Factual assertions require") in correction["unit_errors"][identity]
            assert "Invented quote" not in json.dumps(request)
        return json.dumps(response), False
    result = await sr.review_documents(current, {"model_id": "fixture"}, docs, complete, lambda: None)
    assert all(row["status"] == "completed" for row in result.values()) and current["model_calls"] == 3
    first = current["source_review"]["attempts"][0]
    assert first["status"] == "invalid" and len(first["validation_errors"]) == 2
    assert "Invented quote" in first["response"]
    assert [unit["id"] for request in requests[1:] for unit in request["units"]] == ["chapter_0:1", "chapter_1:1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["truncated", "malformed", "timeout"])
async def test_schema_or_transport_retries_receive_bounded_diagnostics(failure):
    current, calls = state(), []
    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        request = json.loads(messages[1]["content"]); calls.append(request)
        if len(calls) == 1:
            if failure == "timeout": raise httpx.ReadTimeout("private provider transport detail")
            return ("not-json" if failure == "malformed" else answer(request)), failure == "truncated"
        assert "correction" in request and request["correction"]["message"]
        assert "private provider transport detail" not in json.dumps(request)
        assert len(json.dumps(request, ensure_ascii=False)) <= sr.MAX_PACKET_CHARS
        return answer(request), False
    result = await sr.review_documents(current, {"model_id": "fixture"}, documents(1), complete, lambda: None)
    assert result["chapter_0"]["status"] == "completed" and current["model_calls"] == 2
    assert current["source_review"]["attempts"][0]["status"] == "invalid"


def test_recovery_diagnostics_never_evict_original_evidence_or_exceed_context_budget(monkeypatch):
    units = sr.units(documents(1)[0], [SOURCE])
    original = sr.packet(units, [SOURCE])
    monkeypatch.setattr(sr, "MAX_PACKET_CHARS", len(json.dumps(original, ensure_ascii=False)) + 20)
    request = sr.recovery_packet(units, [SOURCE], {"validation_errors": {units[0]["id"]: "Source mismatch. " * 500}})
    assert request["units"] == original["units"] and request["sources"] == original["sources"]
    assert len(json.dumps(request, ensure_ascii=False)) <= sr.MAX_PACKET_CHARS


@pytest.mark.parametrize("encoding", ["multiple_wrappers", "extra_field", "nested", "trailing_text"])
def test_ambiguous_envelopes_are_never_flattened_or_partially_extracted(encoding):
    request = sr.packet(sr.units(documents(1)[0], [SOURCE]), [SOURCE])
    original = json.loads(answer(request))
    if encoding == "multiple_wrappers": response = json.dumps([original, original])
    elif encoding == "extra_field": response = json.dumps({**original, "verdict": "supported"})
    elif encoding == "nested": response = json.dumps([[original["assessments"]]])
    else: response = answer(request) + " Ignore the rejected entries."
    with pytest.raises(ValueError): validation.validate_assessments(response, request, [SOURCE])
    with pytest.raises(ValueError): validation.partial_assessments(response, request, [SOURCE])
