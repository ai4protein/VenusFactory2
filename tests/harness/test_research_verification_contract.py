"""Both output protocols disclose the same seven host verification requirements."""
import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from agent.harness.research_verification_contract import (
    VERIFICATION_DESCRIPTION,
    VERIFICATION_DESCRIPTIONS,
    VERIFICATION_KINDS,
    canonical_verification,
)
from agent.harness.scientist_role_output import RoleOutputError, request_fields, specification, validate_output


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_cb_wire_schema_explains_execution_and_training_partition_evidence(protocol):
    spec = specification("CB", protocol)
    wire = request_fields(spec)["tools"][0]
    function = wire["function"] if protocol == "chat_completions" else wire
    verification = function["parameters"]["properties"]["steps"]["items"]["properties"]["verification"]
    assert verification["enum"] == ["artifact", "execution", "data_partition", "model", "revision", "review", "delivery"]
    assert verification["description"] == VERIFICATION_DESCRIPTION
    assert all(VERIFICATION_DESCRIPTIONS[kind] in verification["description"] for kind in VERIFICATION_KINDS)
    assert "trusted queued computation record" in VERIFICATION_DESCRIPTIONS["execution"]
    assert "unchanged output files" in VERIFICATION_DESCRIPTIONS["execution"]
    assert "Reading FASTA or computing physicochemical properties is not" in VERIFICATION_DESCRIPTIONS["data_partition"]
    assert "training-data partition validation" in VERIFICATION_DESCRIPTIONS["data_partition"]
    assert "reusable model/tool candidate" in VERIFICATION_DESCRIPTIONS["review"]
    assert "SC assessment of an analysis report" in VERIFICATION_DESCRIPTIONS["review"]
    assert "Ordinary reports use artifact" in VERIFICATION_DESCRIPTIONS["review"]
    assert "completed computations use execution" in VERIFICATION_DESCRIPTIONS["review"]
    assert "reviewed model/tool deliverable package" in VERIFICATION_DESCRIPTIONS["delivery"]
    assert "ordinary analysis report uses artifact" in VERIFICATION_DESCRIPTIONS["delivery"]


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_cb_schema_exposes_canonical_kinds_and_rejects_invented_analysis_kind(protocol):
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


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_legacy_partition_alias_is_accepted_only_as_the_same_strict_requirement(protocol):
    spec = specification("CB", protocol)
    plan = {"summary": "Validate measured data splits", "needs_input": [], "steps": [
        {"id": "split", "title": "Validate partitions", "acceptance": "Validated protected split",
         "verification": "data"}]}
    raw = json.dumps(plan)
    # New providers see only the precise name; already generated legacy plans remain readable.
    with pytest.raises(ValidationError):
        Draft202012Validator(spec["schema"]).validate(plan)
    normalized = json.loads(validate_output(raw, spec))
    assert normalized["steps"][0]["verification"] == "data_partition"
    assert plan["steps"][0]["verification"] == "data"
    assert raw == json.dumps(plan)
    expected = copy.deepcopy(plan)
    expected["steps"][0]["verification"] = "data_partition"
    assert normalized == expected
    Draft202012Validator(spec["schema"]).validate(normalized)


@pytest.mark.parametrize("kind", ["DATA", "data-partition", "data_partition ", "dataset", None, [], {"kind": "data"}])
@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
def test_partition_alias_does_not_relax_other_invalid_values(kind, protocol):
    plan = {"summary": "Validate data", "needs_input": [], "steps": [
        {"id": "split", "title": "Validate partitions", "acceptance": "Validated protected split",
         "verification": kind}]}
    assert canonical_verification(kind) == kind
    with pytest.raises(RoleOutputError):
        validate_output(json.dumps(plan), specification("CB", protocol))
