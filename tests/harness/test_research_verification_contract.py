"""Both output protocols disclose the same seven host verification requirements."""
import copy

import pytest
from jsonschema import Draft202012Validator, ValidationError

from agent.harness.research_verification_contract import (
    VERIFICATION_DESCRIPTION,
    VERIFICATION_DESCRIPTIONS,
    VERIFICATION_KINDS,
)
from agent.harness.scientist_role_output import request_fields, specification


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_cb_wire_schema_explains_execution_and_training_partition_evidence(protocol):
    spec = specification("CB", protocol)
    wire = request_fields(spec)["tools"][0]
    function = wire["function"] if protocol == "chat_completions" else wire
    verification = function["parameters"]["properties"]["steps"]["items"]["properties"]["verification"]
    assert verification["enum"] == ["artifact", "execution", "data", "model", "revision", "review", "delivery"]
    assert verification["description"] == VERIFICATION_DESCRIPTION
    assert all(VERIFICATION_DESCRIPTIONS[kind] in verification["description"] for kind in VERIFICATION_KINDS)
    assert "trusted queued computation record" in VERIFICATION_DESCRIPTIONS["execution"]
    assert "unchanged output files" in VERIFICATION_DESCRIPTIONS["execution"]
    assert "Reading FASTA or computing physicochemical properties is not" in VERIFICATION_DESCRIPTIONS["data"]
    assert "training-data partition validation" in VERIFICATION_DESCRIPTIONS["data"]
    assert "reusable model/tool candidate" in VERIFICATION_DESCRIPTIONS["review"]
    assert "SC assessment of an analysis report" in VERIFICATION_DESCRIPTIONS["review"]
    assert "Ordinary reports use artifact" in VERIFICATION_DESCRIPTIONS["review"]
    assert "completed computations use execution" in VERIFICATION_DESCRIPTIONS["review"]
    assert "reviewed model/tool deliverable package" in VERIFICATION_DESCRIPTIONS["delivery"]
    assert "ordinary analysis report uses artifact" in VERIFICATION_DESCRIPTIONS["delivery"]


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_cb_schema_keeps_existing_kinds_and_rejects_invented_analysis_kind(protocol):
    schema = specification("CB", protocol)["schema"]
    validator = Draft202012Validator(schema)
    original = {"summary": "Compute and document the requested property.", "needs_input": [], "steps": [
        {"id": "compute", "title": "Compute property", "acceptance": "Checkpoint the verified output.",
         "verification": "execution"}]}
    for kind in VERIFICATION_KINDS:
        value = copy.deepcopy(original)
        value["steps"][0]["verification"] = kind
        validator.validate(value)
    invalid = copy.deepcopy(original)
    invalid["steps"][0]["verification"] = "analysis"
    with pytest.raises(ValidationError):
        validator.validate(invalid)
    # This describes host checks; it does not let a planner certify its own pass.
    invalid = copy.deepcopy(original)
    invalid["steps"][0]["verification"] = {"kind": "execution", "passed": True}
    with pytest.raises(ValidationError):
        validator.validate(invalid)
