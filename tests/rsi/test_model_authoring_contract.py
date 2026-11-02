"""Exercise actionable authoring guidance through the real bound MCP interface."""

import ast
import json

import pytest
from fastmcp import Client
from test_generated_recovery import GOOD, setup_bundle
from test_protein_references import wrappers

from agent.rsi.code_sandbox import guard_source
from agent.rsi.protein_references import audit_source_references, reference_catalog
from agent.rsi.tool_contracts import validate_generated_result


def reference_source(problem=None):
    catalog = reference_catalog()
    ref_id = "free_amino_acid.average_mass.v1"
    ref = catalog["references"][ref_id]
    metadata = {"reference_id": ref_id, "residue_order": catalog["canonical_order"], "unit": ref["unit"]}
    declared = "MASS"
    initialization = f"MASS = {ref['values']!r}\n"
    if problem == "nested_values":
        metadata["values"] = ref["values"]
        initialization = "MASS = PROTEIN_FEATURE_REFERENCES['MASS']['values']\n"
    elif problem == "residue_keys":
        metadata.update(ref["values"])
    elif problem == "missing_unit":
        metadata.pop("unit")
    elif problem == "symbol_mismatch":
        declared = "MOLECULAR_MASS"
    elif problem == "alias":
        initialization = f"TABLE = {ref['values']!r}\nMASS = TABLE\n"
    elif problem == "reassignment":
        initialization += f"MASS = {ref['values']!r}\n"
    return f"PROTEIN_FEATURE_REFERENCES = { {declared: metadata}!r}\n" + initialization + GOOD


@pytest.mark.asyncio
async def test_inspection_examples_are_valid_constants_not_model_implementations(tmp_path):
    cap, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        tools = await wrappers(client, cap, work, ["inspect_model_code_task"])
        result = json.loads(await tools["inspect_model_code_task"].ainvoke({}))
    assert result["residue_reference_catalog"] == reference_catalog()
    assert result["development_budget"] == {"training_feedback_attempts": 2, "submitted_candidates": 1}
    contract = result["authoring_contract"]
    assert contract["schema"] == 1
    examples = contract["reference_constants"]["standalone_examples"]
    assert len(examples) == 2
    for example in examples:
        source = example["source_code"]
        assert not any(isinstance(node, (ast.FunctionDef, ast.Call)) for node in ast.walk(ast.parse(source)))
        audit = audit_source_references(source)
        assert audit["declared_literals_verified"]
        assert not audit["biological_interpretation_verified"]
        assert audit["references"][0]["symbol"] == example["symbol"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "problem,detail",
    [
        ("nested_values", "unexpected metadata keys=['values']"),
        ("residue_keys", "unexpected metadata keys="),
        ("missing_unit", "missing keys=['unit']"),
        ("symbol_mismatch", "MOLECULAR_MASS: found 0 module-level initializations"),
        ("alias", "MASS: initializer is Name"),
        ("reassignment", "MASS: found 2 module-level initializations"),
    ],
)
async def test_actual_declaration_errors_remain_rejected_and_have_specific_repair_guidance(tmp_path, problem, detail):
    bad = reference_source(problem)
    with pytest.raises(ValueError):
        guard_source(bad)
    cap, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        tools = await wrappers(client, cap, work, ["evaluate_model_code", "submit_model_code"])
        result = json.loads(
            await tools["evaluate_model_code"].ainvoke({"source_code": bad, "rationale": "invalid declaration fixture"})
        )
        feedback = result["training_only_feedback"]
        assert set(feedback) == {"status", "stage", "error_type", "message"}
        assert feedback["stage"] == "source_validation"
        assert detail in feedback["message"]
        assert "EXACTLY reference_id, residue_order, unit" in feedback["message"]
        assert len(feedback["message"]) <= 3000
        assert result["submittable_sources"] == [] and result["attempts_remaining"] == 1
        forged = {**result, "training_only_feedback": {**feedback, "message": "forged repair guidance"}}
        with pytest.raises(ValueError):
            validate_generated_result("evaluate_model_code", forged, workspace=work, capability=cap)
        good = json.loads(
            await tools["evaluate_model_code"].ainvoke(
                {"source_code": reference_source(), "rationale": "correct literal declaration fixture"}
            )
        )
        assert good["status"] == "tested_not_published" and good["attempts_remaining"] == 0
        receipt = json.loads(
            await tools["submit_model_code"].ainvoke(
                {"source_sha256": good["source_sha256"], "rationale": "submit fixture"}
            )
        )
        assert receipt["registered"] is False
        with pytest.raises(Exception, match="budget exhausted"):
            await client.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "third attempt forbidden"})


@pytest.mark.asyncio
async def test_identical_failed_source_warns_and_still_exhausts_two_attempts(tmp_path):
    cap, work, config = setup_bundle(tmp_path)
    source = reference_source("nested_values")
    async with Client(config) as client:
        tools = await wrappers(client, cap, work, ["evaluate_model_code"])
        results = [
            json.loads(await tools["evaluate_model_code"].ainvoke({"source_code": source, "rationale": rationale}))
            for rationale in ["original", "claimed fix without source change"]
        ]
        assert "IDENTICAL SOURCE" not in results[0]["training_only_feedback"]["message"]
        assert "IDENTICAL SOURCE" in results[1]["training_only_feedback"]["message"]
        assert results[1]["attempts_remaining"] == 0 and results[1]["submittable_sources"] == []
        with pytest.raises(Exception, match="budget exhausted"):
            await client.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "third attempt forbidden"})


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["syntax", "name"])
async def test_other_failure_guidance_preserves_stage_and_error_type(tmp_path, kind):
    cap, work, config = setup_bundle(tmp_path)
    source = (
        GOOD.replace("return np.array", "return (np.array", 1)
        if kind == "syntax"
        else GOOD.replace("np.linalg.lstsq", "missing_solver")
    )
    async with Client(config) as client:
        tools = await wrappers(client, cap, work, ["evaluate_model_code"])
        result = json.loads(
            await tools["evaluate_model_code"].ainvoke({"source_code": source, "rationale": "test bounded diagnostics"})
        )
    feedback = result["training_only_feedback"]
    assert feedback["stage"] == ("source_validation" if kind == "syntax" else "execution")
    assert (
        "syntax/indentation" in feedback["message"]
        if kind == "syntax"
        else "Import every allowed module explicitly" in feedback["message"]
    )
    assert result["attempts_remaining"] == 1
