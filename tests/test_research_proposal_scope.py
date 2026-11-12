"""A prospective label cannot exempt adjoining observations from source review."""
import json

import pytest

from backend.web_v2 import research_source_review as review
from backend.web_v2.research_source_validation import validate_assessments


def proposal_request(text):
    document = {"id": "scope", "title": "Research suggestions", "text": text}
    request = review.packet(review.units(document, []), [])
    response = json.dumps({"assessments": [
        {"unit_id": unit["id"], "classification": "proposal", "verdict": "supported",
         "issues": [], "evidence": []} for unit in request["units"]
    ]})
    return document, request, response


@pytest.mark.parametrize("text", [
    "We recommend further experiments. Protein X is stable at 80 C.",
    "Protein X is stable at 80 C; we recommend further experiments.",
    "We recommend measuring stability, because Protein X is stable at 80 C.",
    "We recommend measuring stability, Protein X is stable at 80 C.",
    "We recommend further experiments, Protein X: stable at 80 C.",
    "We recommend further experiments, measured yield: 5 g/L.",
    "We recommend purification and Protein X has a mass of 20 kDa.",
    "We recommend testing the confirmed activity at 80 C.",
    "We recommend measuring yield; it was 5 g/L.",
    "## Protein X is stable at 80 C\n\nWe recommend further experiments.",
    "We recommend expressing the protein in E. coli. Protein X is stable at 80 C.",
    "We propose testing binding, e.g. Protein X is stable at 80 C.",
    "We propose testing binding, e.g. with an orthogonal assay. Protein X is stable at 80 C.",
    "The binding hypothesis could be tested. Protein X is stable at 80 C.",
    "The effect should be tested at 37 C, Protein X: stable at 80 C.",
    "The binding hypothesis could be tested; activity was measured at 37 C.",
    "The binding hypothesis is established and could be tested at 37 C.",
    "该假设可通过正交实验检验，该蛋白在80 C下稳定。",
    "该效应应在37 C下测试。该蛋白的产量是5 g/L。",
    "建议开展进一步实验。该蛋白在80 C下稳定。",
    "建议测量稳定性，该蛋白在80 C下稳定。",
    "建议测量产量，因为该蛋白已有5 g/L的表达产量。",
])
def test_mixed_observations_cannot_be_accepted_as_uncited_proposals(text):
    _, request, response = proposal_request(text)
    with pytest.raises(ValueError, match="proposal|prospective|factual"):
        validate_assessments(response, request, [])


@pytest.mark.parametrize("text", [
    "We propose testing whether expression conditions alter the yield.",
    "We recommend measuring stability at 80 C.",
    "We propose testing whether Protein X is stable at 80 C.",
    "We propose testing whether Protein X has a mass of 20 kDa.",
    "We recommend measuring expression, solubility, and stability.",
    "We recommend measuring stability at 80 C and solubility at pH 7.4.",
    "We recommend comparing wild-type and mutant proteins.",
    "We propose measuring yield; we recommend testing activity.",
    "We recommend a stability assay at 80 C.",
    "## Future experiments\n\nWe propose testing activity at pH 7.4.",
    "We recommend expressing the protein in E. coli.",
    "We propose testing binding, e.g. with an orthogonal assay.",
    "We recommend expressing the protein in E. coli; we propose measuring yield.",
    "The binding hypothesis could be tested using an orthogonal assay.",
    "The effect should be tested at 37 C.",
    "Yield should be measured at 37 C.",
    "This hypothesis could be tested; the effect should be evaluated at 37 C.",
    "The effect should be tested at 37 C, we recommend comparing two conditions.",
    "该假设可检验结合机制。",
    "该假设可通过正交实验检验。",
    "该效应应在37 C下测试。",
    "建议在80 C下测量稳定性。",
    "建议检验该蛋白是否在80 C下稳定。",
    "下一步建议比较野生型与突变体的表达产量。",
])
def test_pure_experimental_proposals_need_no_invented_observations(text):
    document, request, response = proposal_request(text)
    checked = validate_assessments(response, request, [])
    assert all(row["passed"] and not row["evidence"] for row in checked)
    receipt = review.document_receipt(document, [], {row["unit_id"]: row for row in checked})
    assert review.document_ready(document, [], receipt)


@pytest.mark.asyncio
async def test_old_policy_receipts_are_reassessed_and_can_finish():
    source = {"id": "S1", "title": "Fixture observation", "kind": "database",
              "text": "Protein X is stable at 20 C."}
    document = {"id": "scope", "title": "Observation", "text": source["text"] + " [S1]"}
    old = {"policy": "source_review.v2.2", "status": "completed", "units": {}, "attempts": []}
    state = {"sources": [source], "source_review": old, "model_calls": 0, "model_call_limit": 4}
    calls = []

    async def model_call(model, messages, output_tokens):
        request = json.loads(messages[1]["content"])
        calls.append(request)
        rows = []
        for unit in request["units"]:
            passage = request["sources"][0]["passages"][0]
            rows.append({"unit_id": unit["id"], "classification": "factual", "verdict": "supported",
                         "issues": [], "evidence": [{"source_id": "S1", "passage_id": passage["id"],
                                                      "quote": source["text"]}]})
        return json.dumps({"assessments": rows}), False

    receipts = await review.review_documents(state, {"model_id": "fixture"}, [document], model_call, lambda: None)
    assert calls and review.VERSION != old["policy"]
    assert state["source_review_history"] == [old]
    assert review.document_ready(document, [source], receipts["scope"])


def test_cited_observation_and_suggestion_remain_factual_evidence_not_a_proposal_exemption():
    source = {"id": "S1", "title": "Fixture observation", "kind": "database",
              "text": "Protein X is stable at 20 C."}
    document = {"id": "scope", "title": "Mixed scope",
                "text": source["text"] + " [S1] We recommend comparing other conditions."}
    request = review.packet(review.units(document, [source]), [source])
    passage = request["sources"][0]["passages"][0]
    row = {"unit_id": request["units"][0]["id"], "classification": "proposal", "verdict": "supported",
           "issues": [], "evidence": [{"source_id": "S1", "passage_id": passage["id"], "quote": source["text"]}]}
    with pytest.raises(ValueError, match="Proposal-only"):
        validate_assessments(json.dumps({"assessments": [row]}), request, [source])
    row["classification"] = "factual"
    assert validate_assessments(json.dumps({"assessments": [row]}), request, [source])[0]["passed"]
