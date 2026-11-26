import copy
import json

import pytest

from backend.web_v2 import research_source_review as sr
from backend.web_v2 import research_source_validation as validation
from backend.web_v2.research_evidence_passages import resolve_passage, selected_passages, source_digest

SOURCE = {"id":"S1","kind":"database","title":"Reference annotation",
          "text":json.dumps({"features":[{"type":"Initiator methionine","description":"Removed",
                                         "location":{"start":1,"end":1}}],"sequence_length":147}),
          "url":"https://example.org/entry"}
DOCUMENT = {"id":"identity","title":"Identity","text":"The initiator methionine is removed. [S1]"}


def assessment(request, *, verdict="supported"):
    result=[]
    for unit in request["units"]:
        source=next((s for s in request["sources"] if s["id"] in unit["citations"]),None)
        passage=source["passages"][0] if source and source["passages"] else None
        result.append({"unit_id":unit["id"],"classification":"factual","verdict":verdict,
                       "issues":[] if verdict=="supported" else [{"claim":unit["text"],"reason":"The annotation does not establish this claim."}],
                       "evidence":[{"source_id":source["id"],"passage_id":passage["id"],"quote":passage["text"]}]
                       if passage else []})
    return json.dumps({"assessments":result})


def state():
    return {"sources":[copy.deepcopy(SOURCE)],"model_calls":0,"model_call_limit":18}


def test_json_and_plain_anchors_match_original_saved_evidence():
    source={**SOURCE,"text":json.dumps({"filler":"background "*900,
                                      "features":[{"type":"Initiator methionine","description":"Removed"}]})}
    selected=selected_passages(source,"signal peptide cleavage maturation",600)
    assert any("Initiator methionine" in p["text"] for p in selected["passages"])
    for passage in selected["passages"]:
        assert resolve_passage(source,passage["location"])==passage["text"]
    plain={**SOURCE,"text":"Original annotation. "*400}
    for passage in selected_passages(plain,"annotation",300)["passages"]:
        assert resolve_passage(plain,passage["location"])==passage["text"]



def retrieval_source(measurement):
    return {
        **SOURCE,
        "text": json.dumps({
            "references": [
                {"source": "Reference index", "description": "Background annotation. " * 15,
                 "id": str(100000 + index)}
                for index in range(12)
            ],
            "measurement": measurement,
        }),
    }


@pytest.mark.parametrize("query", ["pH", "pI", "Km", "Kd", "Ki"])
def test_short_scientific_queries_retrieve_measurements(query):
    source = retrieval_source({"assay": query + " was measured for this sample."})
    original = copy.deepcopy(source)
    selected = selected_passages(source, query, 600)
    assert selected["passages"][0]["location"]["json_pointer"] == "/measurement"
    assert sum(len(p["text"]) for p in selected["passages"]) <= 600
    assert all(resolve_passage(source, p["location"]) == p["text"]
               for p in selected["passages"])
    assert source == original


@pytest.mark.parametrize("field,query", [
    ("kineticParameters", "kinetic parameters"),
    ("michaelisConstants", "Michaelis constants"),
    ("RNAExpression", "RNA expression"),
    ("pHDependence", "pH dependence"),
    ("binding_affinity", "binding affinity"),
])
def test_structured_field_queries_retrieve_measurements(field, query):
    source = retrieval_source({field: {"value": 8.2, "unit": "mM"}})
    selected = selected_passages(source, query, 600)
    assert selected["passages"][0]["location"]["json_pointer"] == "/measurement"
    assert resolve_passage(source, selected["passages"][0]["location"]) == selected["passages"][0]["text"]


def test_writer_and_reviewer_share_ph_evidence_selection():
    from backend.web_v2.research_chapter_context import source_context
    source = retrieval_source({"phDependence": {"texts": [{"value": "Optimum pH is 6-8."}]}})
    original = copy.deepcopy(source)
    writer = source_context([source], query="pH")
    document = {**DOCUMENT, "text": "The optimum pH is 6-8. [S1]"}
    reviewer = sr.packet(sr.units(document, [source]), [source])
    assert "Optimum pH is 6-8." in writer[0]["text"]
    assert any("Optimum pH is 6-8." in p["text"]
               for p in reviewer["sources"][0]["passages"])
    assert source == original


def test_short_table_heading_and_long_paragraph_claims_are_not_dropped():
    doc={**DOCUMENT,"text":"## A strong experimental claim\n\n|Value|Result|\n|---|---|\n|Class|1 [S1]|\n\n"+
         "A factual assertion. "*500+"[S1]"}
    units=sr.units(doc,[SOURCE])
    assert "A strong experimental claim" in units[0]["text"]
    assert "|Class|1" in units[0]["text"]
    assert len(units)>2 and all(unit["citations"]==["S1"] for unit in units)

@pytest.mark.parametrize("mutation",["missing","duplicate","unknown_unit","wrong_quote","wrong_source","wrong_passage","empty_evidence",
                                    "unsupported_without_issue","supported_with_issue","unmarked_proposal","bad_issue","extra_field"])
def test_incomplete_or_unanchored_verdicts_are_rejected(mutation):
    request=sr.packet(sr.units(DOCUMENT,[SOURCE]),[SOURCE])
    response=json.loads(assessment(request));row=response["assessments"][0]
    if mutation=="missing":response["assessments"]=[]
    elif mutation=="duplicate":response["assessments"].append(copy.deepcopy(row))
    elif mutation=="unknown_unit":row["unit_id"]="unknown"
    elif mutation=="wrong_quote":row["evidence"][0]["quote"]="Signal peptide removed"
    elif mutation=="wrong_source":row["evidence"][0]["source_id"]="S9"
    elif mutation=="wrong_passage":row["evidence"][0]["passage_id"]="fabricated"
    elif mutation=="empty_evidence":row["evidence"]=[]
    elif mutation=="unsupported_without_issue":row["verdict"]="unsupported"
    elif mutation=="supported_with_issue":row["issues"]=[{"claim":"methionine","reason":"Unresolved"}]
    elif mutation=="unmarked_proposal":row.update(classification="proposal",evidence=[])
    elif mutation=="bad_issue":row.update(verdict="uncertain",issues=[{"claim":"not present","reason":"Missing evidence"}])
    else:row["authority"]="trust me"
    with pytest.raises((ValueError,KeyError,TypeError)):
        validation.validate_assessments(json.dumps(response),request,[SOURCE])


def test_fabricated_packet_cannot_replace_original_evidence():
    request=sr.packet(sr.units(DOCUMENT,[SOURCE]),[SOURCE])
    request["sources"][0]["passages"][0]["text"]="Invented signal peptide cleavage."
    with pytest.raises(ValueError,match="original source"):
        validation.validate_assessments(assessment(request),request,[SOURCE])


def test_explicit_proposals_need_no_fake_measurement_citations():
    document={**DOCUMENT,"text":"We propose testing whether expression conditions alter the yield."}
    request=sr.packet(sr.units(document,[]),[])
    response={"assessments":[{"unit_id":request["units"][0]["id"],"classification":"proposal",
                             "verdict":"supported","issues":[],"evidence":[]}]}
    assert validation.validate_assessments(json.dumps(response),request,[])[0]["passed"]


@pytest.mark.asyncio
async def test_review_cache_invalidated_by_source_or_draft_changes():
    current=state();calls=[]
    async def complete(model,messages,tokens):
        current["model_calls"]+=1;calls.append(messages)
        assert len(messages)==2 and "independently review" in messages[0]["content"]
        return assessment(json.loads(messages[1]["content"])),False
    receipts=await sr.review_documents(current,{"model_id":"fixture"},[DOCUMENT],complete,lambda:None)
    assert sr.document_ready(DOCUMENT,current["sources"],receipts["identity"])
    await sr.review_documents(current,{"model_id":"fixture"},[DOCUMENT],complete,lambda:None)
    assert len(calls)==1
    changed={**DOCUMENT,"text":DOCUMENT["text"]+" Additional assertion."}
    assert not sr.document_ready(changed,current["sources"],receipts["identity"])
    current["sources"][0]["query_arguments"]={"organism":"different"}
    assert not sr.document_ready(DOCUMENT,current["sources"],receipts["identity"])
    await sr.review_documents(current,{"model_id":"fixture"},[DOCUMENT],complete,lambda:None)
    assert len(calls)==2

@pytest.mark.asyncio
@pytest.mark.parametrize("failure",["truncated","missing","unsupported","budget"])
async def test_failed_review_retained_and_never_complete(failure):
    current=state()
    if failure=="budget":current["model_calls"]=current["model_call_limit"]
    async def complete(model,messages,tokens):
        current["model_calls"]+=1;request=json.loads(messages[1]["content"])
        return ('{"assessments":[]}' if failure=="missing" else
                assessment(request,verdict="unsupported" if failure=="unsupported" else "supported"),failure=="truncated")
    receipts=await sr.review_documents(current,{"model_id":"fixture"},[DOCUMENT],complete,lambda:None)
    assert not sr.document_ready(DOCUMENT,current["sources"],receipts["identity"])
    if failure!="budget":
        attempt=current["source_review"]["attempts"][0]
        assert attempt["response"] and attempt["request"]["sources"]
    assert current["model_calls"]<=current["model_call_limit"]


@pytest.mark.asyncio
async def test_source_changes_mid_response_do_not_validate_packet():
    current=state()
    async def complete(model,messages,tokens):
        current["model_calls"]+=1;request=json.loads(messages[1]["content"])
        current["sources"][0]["text"]="A different annotation."
        return assessment(request),False
    result=await sr.review_documents(current,{"model_id":"fixture"},[DOCUMENT],complete,lambda:None)
    assert result["identity"]["status"]=="partial"

def test_progress_events_do_not_duplicate_audit_prompts_or_old_drafts(monkeypatch):
    import backend.web_v2.task_progress as progress
    class Queue:
        frames=[]
        def append_frame(self,*args):self.frames.append(args)
    monkeypatch.setenv("VENUS_QUEUE_JOB_ID","test-job")
    monkeypatch.setenv("VENUS_QUEUE_LEASE_TOKEN","test-lease")
    monkeypatch.setattr(progress,"JobQueue",Queue)
    progress.publish_progress({"chapters":[{"id":"x","status":"reviewing","draft":"PRIVATE_DRAFT",
                                          "draft_history":[{"draft":"PRIVATE_OLD"}]}],
                               "source_review":{"attempts":[{"request":"PRIVATE_PACKET"}]}})
    assert len(Queue.frames)==1
    assert "reviewing" in Queue.frames[0][1] and "PRIVATE" not in Queue.frames[0][1]

def test_independent_reviewer_cannot_accept_a_wrong_concentration_unit():
    source = {**SOURCE, "text": json.dumps({"constant": 2.9, "unit": "mM"})}
    document = {**DOCUMENT, "text": "The measured constant is 2.9 nM. [S1]"}
    request = sr.packet(sr.units(document, [source]), [source])
    with pytest.raises(ValueError, match="measured value and unit"):
        validation.validate_assessments(assessment(request), request, [source])


@pytest.mark.asyncio
async def test_quantity_policy_upgrade_invalidates_old_receipts(monkeypatch):
    current = state()

    async def complete(model, messages, tokens):
        current["model_calls"] += 1
        return assessment(json.loads(messages[1]["content"])), False

    with monkeypatch.context() as old_policy:
        old_policy.setattr(sr, "VERSION", "source_review.v2")
        receipts = await sr.review_documents(
            current, {"model_id": "fixture"}, [DOCUMENT], complete, lambda: None
        )
        assert sr.document_ready(DOCUMENT, current["sources"], receipts["identity"])
    assert not sr.document_ready(DOCUMENT, current["sources"], receipts["identity"])


def compact_assessment(request):
    response = json.loads(assessment(request))
    for row in response["assessments"]:
        row["evidence"] = [{"passage_id": item["passage_id"]} for item in row["evidence"]]
    return json.dumps(response)


def test_compact_evidence_resolves_exact_original_text_and_keeps_receipt_shape():
    request = sr.packet(sr.units(DOCUMENT, [SOURCE]), [SOURCE])
    result = sr.validate_assessments(compact_assessment(request), request, [SOURCE])
    full = sr.validate_assessments(assessment(request), request, [SOURCE])
    assert result == full
    anchor = result[0]["evidence"][0]
    assert anchor["quote"] == resolve_passage(SOURCE, anchor["location"])
    assert anchor["source_sha256"] == source_digest(SOURCE)


@pytest.mark.parametrize("change", ["fabricated", "uncited", "altered_packet", "altered_original", "extra_field"])
def test_compact_evidence_cannot_expand_or_replace_its_cited_sources(change):
    extra = {**SOURCE, "id": "S2"}
    sources = [copy.deepcopy(SOURCE), extra]
    request = sr.packet(sr.units(DOCUMENT, sources), sources)
    response = json.loads(compact_assessment(request))
    anchor = response["assessments"][0]["evidence"][0]
    if change == "fabricated":
        anchor["passage_id"] = "S1:invented"
    elif change == "uncited":
        selected = selected_passages(extra, "methionine", 800)
        request["sources"].append(selected)
        anchor["passage_id"] = selected["passages"][0]["id"]
    elif change == "altered_packet":
        request["sources"][0]["passages"][0]["text"] = "Invented signal peptide cleavage."
    elif change == "altered_original":
        sources[0]["text"] = "Different annotation."
    else:
        anchor["quote"] = "Signal peptide removed"
    with pytest.raises((ValueError, KeyError, TypeError)):
        sr.validate_assessments(json.dumps(response), request, sources)


def test_compact_evidence_keeps_measured_value_and_unit_checks():
    source = {**SOURCE, "text": "The melting temperature was 45 C."}
    document = {**DOCUMENT, "text": "The melting temperature was 80 C. [S1]"}
    request = sr.packet(sr.units(document, [source]), [source])
    with pytest.raises(ValueError, match="measured value and unit"):
        sr.validate_assessments(compact_assessment(request), request, [source])


def test_old_policy_receipt_is_not_accepted_for_compact_review():
    receipt = {"policy": "source_review.v2.1", "status": "completed", "units": 1, "passed": 1,
               "input_sha256": sr.document_signature(DOCUMENT, [SOURCE])}
    assert not sr.document_ready(DOCUMENT, [SOURCE], receipt)


def test_compact_evidence_partial_recovery_uses_the_same_source_contract():
    request = sr.packet(sr.units(DOCUMENT, [SOURCE]), [SOURCE])
    response = compact_assessment(request)
    complete = validation.validate_assessments(response, request, [SOURCE])
    checked, invalid = validation.partial_assessments(response, request, [SOURCE])
    assert checked == complete and invalid == {}
    corrupted = json.loads(response)
    corrupted["assessments"][0]["evidence"][0]["passage_id"] = "S1:invented"
    checked, invalid = validation.partial_assessments(json.dumps(corrupted), request, [SOURCE])
    assert checked == []
    assert request["units"][0]["id"] in invalid


@pytest.mark.parametrize("identity", [None, 7, [], {}])
def test_compact_evidence_requires_an_explicit_string_identity(identity):
    request = sr.packet(sr.units(DOCUMENT, [SOURCE]), [SOURCE])
    response = json.loads(compact_assessment(request))
    response["assessments"][0]["evidence"][0]["passage_id"] = identity
    with pytest.raises(ValueError, match="source anchor"):
        validation.validate_assessments(json.dumps(response), request, [SOURCE])


def test_bare_passage_id_resolves_against_original_source():
    request = sr.packet(sr.units(DOCUMENT, [SOURCE]), [SOURCE])
    response = json.loads(assessment(request))
    row = response["assessments"][0]
    passage = row["evidence"][0]
    row["evidence"] = [passage["passage_id"]]
    resolved = validation.validate_assessments(json.dumps(response), request, [SOURCE])
    assert resolved[0]["evidence"][0]["quote"] == passage["quote"]
    row["evidence"] = ["S1:missing"]
    with pytest.raises(ValueError):
        validation.validate_assessments(json.dumps(response), request, [SOURCE])


@pytest.mark.asyncio
async def test_first_review_does_not_spend_revision_budget_on_format_retries():
    progress = state()
    progress["model_call_limit"] = 3

    async def invalid(*args, **kwargs):
        progress["model_calls"] += 1
        return "{}", False

    receipts = await sr.review_documents(progress, {"model_id": "fixture"}, [DOCUMENT], invalid,
                                         lambda: None, recovery_reserve=2)
    assert progress["model_calls"] == 1
    assert receipts[DOCUMENT["id"]]["status"] != "completed"


def test_targeted_author_repair_can_only_replace_unique_flagged_paragraphs():
    from backend.web_v2.research_source_repair import apply_unit_edits
    text = "Reviewed. [S1]\n\nNeeds revision. [S2]\n\nAlso reviewed. [S3]"
    units = [{"id": "topic:2", "text": "Needs revision. [S2]"}]
    change = {"unit_id": "topic:2", "text": "Revised with evidence. [S2]"}
    result = apply_unit_edits(text, units, json.dumps({"replacements": [change]}))
    assert result == text.replace(units[0]["text"], change["text"])
    for edits in ([], [change, change], [{**change, "unit_id": "topic:1"}]):
        with pytest.raises(ValueError):
            apply_unit_edits(text, units, json.dumps({"replacements": edits}))


def test_unverified_source_feedback_reaches_author_without_supplying_a_verdict():
    state_data = state()
    unit = sr.units(DOCUMENT, state_data["sources"])[0]
    request = sr.packet([unit], state_data["sources"])
    attempt = {"request": request, "reviewer_binding_sha256": "active",
               "validation_errors": {unit["id"]: "Quote is not in the cited original source passage"}}
    review = {"units": {}, "attempts": [attempt], "reviewer_binding_sha256": "active"}
    state_data["source_review"] = review
    revision = sr.revision_input(state_data, DOCUMENT)
    assert [row["unit_id"] for row in revision["issues"]] == [unit["id"]]
    assert sr.document_receipt(DOCUMENT, state_data["sources"], review["units"])["assessed"] == 0
    assert not sr.feedback(state_data, {**DOCUMENT, "text": "A different assertion. [S1]"})
    attempt["reviewer_binding_sha256"] = "old"
    assert not sr.feedback(state_data, DOCUMENT)
    attempt["reviewer_binding_sha256"] = "active"
    attempt["validation_errors"][unit["id"]] = "Missing assessment"
    assert not sr.feedback(state_data, DOCUMENT)
    attempt["validation_errors"][unit["id"]] = "Factual assertions require original-source evidence"
    review["units"][unit["id"]] = {"input_sha256": unit["input_sha256"], "passed": True,
                                   "reviewer_binding_sha256": "active"}
    assert not sr.feedback(state_data, DOCUMENT)


@pytest.mark.asyncio
async def test_a_blocked_chapter_cannot_spend_the_next_chapters_review_calls():
    state_data = state()
    failing = {**DOCUMENT, "id": "first"}
    next_doc = {**DOCUMENT, "id": "next"}
    async def call(model, messages, tokens, **kwargs):
        state_data["model_calls"] += 1
        request = json.loads(messages[-1]["content"])
        if request["units"][0]["document_id"] == "first":
            return "{}", False
        return assessment(request), False
    budget = state_data["model_call_limit"]
    receipts = await sr.review_documents(state_data, {"model_id":"free"}, [failing], call, lambda: None, max_calls=1)
    assert receipts["first"]["status"] == "partial" and state_data["model_calls"] == 1
    receipts = await sr.review_documents(state_data, {"model_id":"free"}, [next_doc], call, lambda: None, max_calls=1)
    assert receipts["next"]["status"] == "completed" and state_data["model_calls"] == 2
    assert state_data["model_call_limit"] == budget


def test_packet_alias_must_resolve_to_a_cited_original_passage():
    sources = [SOURCE, {**SOURCE, "id": "S2", "title": "Another entry"}]
    other = {**DOCUMENT, "id":"other", "text":"Information about another entry. [S2]"}
    request = sr.packet([*sr.units(DOCUMENT, sources), *sr.units(other, sources)], sources)
    result = json.loads(assessment(request))
    for row in result["assessments"]:
        row["evidence"] = [request["unit_refs"][row["unit_id"]][0]]
    checked = validation.validate_assessments(json.dumps(result), request, sources)
    assert checked[0]["evidence"][0]["passage_id"] == request["sources"][0]["passages"][0]["id"]
    for wrong in ("S2.1", "S1.99", "S1"):
        result["assessments"][0]["evidence"] = [wrong]
        with pytest.raises(ValueError):
            validation.validate_assessments(json.dumps(result), request, sources)


def test_markdown_heading_maps_back_to_original_draft_without_rewriting_other_units():
    from backend.web_v2.research_source_repair import apply_unit_edits
    draft = "## Context\r\n\r\nKeep this. [S1]\r\n\r\n## Finding\r\n\r\nRevise this. [S1]\r\n\r\nKeep the ending. [S1]"
    document = {**DOCUMENT, "text": draft}
    target = sr.units(document, [SOURCE])[1]
    edits = json.dumps({"replacements": [{"unit_id": target["id"], "text": "## Finding\r\n\r\nCorrected. [S1]"}]})
    assert apply_unit_edits(draft, [target], edits) == draft.replace("Revise this.", "Corrected.")
    with pytest.raises(ValueError):
        apply_unit_edits(draft, [target], "An unrelated replacement for the whole chapter.")
    assert apply_unit_edits(draft, [target], draft.replace("Revise this.", "Corrected.")) == draft.replace("Revise this.", "Corrected.")


def test_legacy_markdown_revision_preserves_intervening_reviewed_blocks():
    from backend.web_v2.research_source_repair import apply_unit_edits

    draft = "Introduction.\n\nRepair A.\n\nReviewed middle.\n\nRepair B.\n\nConclusion."
    units = sr.units({**DOCUMENT, "text": draft}, [])
    targets = [units[1], units[3]]
    corrected = draft.replace("Repair A.", "Fixed A.").replace("Repair B.", "Fixed B.")
    assert apply_unit_edits(draft, targets, corrected) == corrected
    for protected in ("Introduction.", "Reviewed middle.", "Conclusion."):
        with pytest.raises(ValueError, match="protected paragraph"):
            apply_unit_edits(draft, targets, corrected.replace(protected, "Altered."))


def test_table_divider_can_be_mapped_without_matching_other_equal_paragraphs():
    from backend.web_v2.research_source_repair import unit_locations

    draft = "| Item | Value |\n|:---|---:|\n| A | 1 |\n\nSame.\n\nSame."
    first, second, _ = sr.units({**DOCUMENT, "text": draft}, [])
    start, end = unit_locations(draft, [first])[first["id"]]
    assert draft[start:end] == draft.split("\n\n")[0]
    with pytest.raises(ValueError, match="Ambiguous"):
        unit_locations(draft, [second])


@pytest.mark.asyncio
async def test_long_chapter_sends_only_flagged_text_and_records_author_timeout():
    from backend.web_v2.research_source_repair import repair_chapters

    draft = "Unsupported sentence. [S1]\n\n" + "Previously reviewed text. " * 1800
    doc = {**DOCUMENT, "text": draft}
    state_data = state()
    unit = sr.units(doc, state_data["sources"])[0]
    state_data["source_review"] = {"units": {unit["id"]: {
        "unit_id": unit["id"], "input_sha256": unit["input_sha256"], "passed": False,
        "issues": [{"claim": "Unsupported sentence.", "reason": "Missing source."}],
    }}}
    chapter = {"id": doc["id"], "title": doc["title"], "draft": draft, "source_ids": ["S1"]}
    requests = []

    async def timed_out(model, messages, tokens, **kwargs):
        requests.append(json.loads(messages[1]["content"]))
        raise TimeoutError("private upstream detail")

    await repair_chapters(state_data, {"model_id": "fixture"}, [chapter], timed_out, lambda: None)
    assert len(requests) == 1
    assert "draft" not in requests[0]
    assert [row["id"] for row in requests[0]["units"]] == [unit["id"]]
    assert requests[0]["sources"] and requests[0]["issues"]
    assert chapter["draft"] == draft and chapter["status"] == "partial"
    assert chapter["revision_error"] == {"stage": "author", "type": "TimeoutError"}


def test_negative_whole_unit_feedback_preserves_review_failure():
    request = sr.packet(sr.units(DOCUMENT, [SOURCE]), [SOURCE])
    row = json.loads(assessment(request, verdict="uncertain"))["assessments"][0]
    row["issues"][0]["claim"] = row["unit_id"]
    result = validation.validate_assessments(json.dumps([row]), request, [SOURCE])[0]
    assert result["passed"] is False
    assert result["issues"][0]["claim"] == request["units"][0]["text"]
    row["verdict"] = "supported"
    with pytest.raises(ValueError):
        validation.validate_assessments(json.dumps([row]), request, [SOURCE])


def test_author_can_remove_flagged_unit_without_editing_protected_neighbors():
    from backend.web_v2.research_chapter_edits import apply_unit_edits
    draft = "Keep this first.\n\nRemove this assertion.\n\nKeep this last."
    target = {"id": "chapter:2", "text": "Remove this assertion."}
    edits = json.dumps([{"unit_id": target["id"], "text": ""}])
    assert apply_unit_edits(draft, [target], "```JSON\n" + edits + "\n```") == draft.replace(target["text"], "")


def test_scoped_revision_keeps_neighbor_review_signatures():
    from backend.web_v2.research_chapter_edits import apply_unit_edits, scoped_replacement
    doc = {**DOCUMENT, "text": "First fact. [S1]\n\nNeighboring fact. [S1]"}
    first, neighbor = sr.units(doc, [SOURCE])
    answer = "### Finding\n\nFirst revision. [S1]\n\nSecond sentence. [S1]"
    edit = {"unit_id": first["id"], "text": scoped_replacement(first, answer)}
    doc["text"] = apply_unit_edits(doc["text"], [first], json.dumps([edit]))
    assert sr.units(doc, [SOURCE])[1:] == [neighbor]
