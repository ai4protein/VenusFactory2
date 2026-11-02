"""Reference provenance, candidate repair and independent registered reuse."""

import copy
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastmcp import Client

from agent.rsi.code_sandbox import guard_source
from agent.rsi.langgraph_tools import build_tools
from agent.rsi.model_lifecycle import ModelLifecycle
from agent.rsi.model_plugins import get_model_plugin
from agent.rsi.protein_model import digest
from agent.rsi.protein_references import audit_source_references, reference_catalog
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

ROOT = Path(__file__).resolve().parents[2]
AA = "ACDEFGHIKLMNPQRSTVWY"
REF = "free_amino_acid.average_mass.v1"
SCOPE = {
    "protein": "synthetic_reference_fixture",
    "assay": "synthetic",
    "target": "linear",
    "unit": "fixture",
    "sequence_length": 2,
}


def source(order=AA, *, reference=REF, mapping=False, mutation=None):
    catalog = reference_catalog()
    ref = catalog["references"][reference]
    values = {a: ref["values"][a] for a in order}
    if mutation:
        values.update(mutation)
    data = repr(values) if mapping else "np.array(" + repr([values[a] for a in order]) + ", dtype=float)"
    declaration = {"MW": {"reference_id": reference, "residue_order": order, "unit": ref["unit"]}}
    return (
        "import numpy as np\nAA="
        + repr(order)
        + "\nMW="
        + data
        + "\nPROTEIN_FEATURE_REFERENCES="
        + repr(declaration)
        + "\n"
        + "def features(sequences):\n    return np.array([sum(MW[AA.index(a)] for a in s)/len(s) for s in sequences])\n"
        + 'def fit(sequences,targets,seed):\n    x=features(sequences)\n    return {"intercept":float(np.mean(targets)-0.01*np.mean(x))}\n'
        + 'def predict(model,sequences):\n    return (0.01*features(sequences)+model["intercept"]).tolist()\n'
    )


def test_frozen_reference_values_match_the_pinned_primary_package():
    import Bio
    from Bio.Data import IUPACData
    from Bio.SeqUtils import ProtParamData

    assert Bio.__version__ == "1.87"
    catalog = reference_catalog()
    assert catalog["catalog_sha256"] == "d29135c8d338f850b4cdab6292e4e16608ebfeab17eb999a5279da395905daa3"
    for key, module, symbol in [
        ("kyte_doolittle.hydropathy.v1", ProtParamData, "kd"),
        (REF, IUPACData, "protein_weights"),
    ]:
        row = catalog["references"][key]
        assert row["values"] == {a: getattr(module, symbol)[a] for a in AA}
        assert row["source"]["sha256"] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    changed = reference_catalog()
    changed["references"][REF]["values"]["H"] = 181
    assert reference_catalog()["references"][REF]["values"]["H"] == 155.1546


@pytest.mark.parametrize("reference", [REF, "kyte_doolittle.hydropathy.v1"])
@pytest.mark.parametrize("order", [AA, AA[::-1]])
@pytest.mark.parametrize("mapping", [False, True])
def test_literal_checker_respects_declared_order_and_mapping(reference, order, mapping):
    report = audit_source_references(source(reference=reference, order=order, mapping=mapping))
    assert report["status"] == "matched_declared_literals"
    assert report["declared_literals_verified"]
    assert report["biological_interpretation_verified"] is False
    assert guard_source(source(reference=reference, order=order, mapping=mapping))["reference_audit"] == report


@pytest.mark.parametrize(
    "problem",
    [
        "wrong_value",
        "wrong_unit",
        "wrong_order",
        "unknown_reference",
        "dynamic_initializer",
        "short_vector",
        "bool_value",
        "reassignment",
        "duplicate_declaration",
        "invalid_declaration",
    ],
)
def test_contradictory_or_unverifiable_declarations_fail_before_execution(problem):
    code = source()
    if problem == "wrong_value":
        code = source(mutation={"H": 181})
    elif problem == "wrong_unit":
        code = code.replace("'unit': 'Da'", "'unit': 'kDa'")
    elif problem == "wrong_order":
        code = code.replace("'residue_order': " + repr(AA), "'residue_order': " + repr(AA[::-1]))
    elif problem == "unknown_reference":
        code = code.replace(REF, "unknown.reference.v1")
    elif problem == "dynamic_initializer":
        code = code.replace("MW=np.array(", "MW=custom_array(")
    elif problem == "short_vector":
        code = code.replace("89.0932, ", "")
    elif problem == "bool_value":
        code = code.replace("89.0932, ", "True, ")
    elif problem == "reassignment":
        code += "\nMW=[1.0]*20\n"
    elif problem == "duplicate_declaration":
        code += "\nPROTEIN_FEATURE_REFERENCES={}\n"
    else:
        code = code.replace("PROTEIN_FEATURE_REFERENCES={", "PROTEIN_FEATURE_REFERENCES=dict(")
    with pytest.raises((ValueError, SyntaxError)):
        guard_source(code)


def test_missing_declaration_and_correct_unused_constants_are_not_biological_proof():
    code = source(mutation={"H": 181})
    code = "\n".join(line for line in code.splitlines() if not line.startswith("PROTEIN_FEATURE_REFERENCES="))
    audit = guard_source(code)["reference_audit"]
    assert audit["status"] == "not_declared" and audit["unclaimed_property_named_assignments"] == ["MW"]
    assert not audit["declared_literals_verified"] and not audit["biological_interpretation_verified"]
    # Matching a declared literal cannot establish that the prediction uses it.
    code = source().replace('0.01*features(sequences)+model["intercept"]', "np.zeros(len(sequences))")
    audit = guard_source(code)["reference_audit"]
    assert audit["declared_literals_verified"] and not audit["biological_interpretation_verified"]


def bundle(tmp_path, phase, rows, packet=None):
    sys.path.insert(0, str(ROOT / "scripts/rsi"))
    from run_generated_lifecycle import FILES

    cap, work = tmp_path / "cap", tmp_path / "work"
    cap.mkdir(parents=True)
    work.mkdir()
    for name in FILES:
        shutil.copy2(ROOT / "src/agent/rsi" / name, cap / name)
    name = "training.json" if phase == "develop" else "inputs.json"
    task = {"phase": phase, "scope": SCOPE, "plugin_id": "generated_python.v1", "inputs": {name: digest(rows)}}
    if packet is not None:
        (cap / "registered").mkdir()
        atomic_json(cap / "registered/capability.json", packet)
        task["capability_sha256"] = digest(packet)
    atomic_json(cap / "task.json", task)
    atomic_json(work / name, rows)
    config = {
        "mcpServers": {
            "generated": {
                "command": sys.executable,
                "args": ["-u", str(cap / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work)},
            }
        }
    }
    return cap, work, config


async def wrappers(client, cap, work, names):
    result = {"tool_calls": 0, "tool_errors": 0}
    tools = await build_tools(
        client,
        allowed_tools=set(names),
        workspace=work,
        capability=cap,
        deadline=time.monotonic() + 120,
        max_tool_calls=8,
        result=result,
        event=lambda x: None,
    )
    return {x.name: x for x in tools}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["science_expert", "science_agent"])
async def test_real_reference_error_repair_registration_and_fresh_reuse(tmp_path, mode):
    mass = reference_catalog()["references"][REF]["values"]
    all_rows = [
        {"id": a + b, "sequence": a + b, "target": 0.7 + 0.01 * (mass[a] + mass[b]) / 2} for a in AA[:4] for b in AA
    ]
    train, valid, future = all_rows[:40], all_rows[40:60], all_rows[60:]
    cap, work, config = bundle(tmp_path / "develop", "develop", train)
    async with Client(config) as client:
        tools = await wrappers(
            client, cap, work, ["inspect_model_code_task", "evaluate_model_code", "submit_model_code"]
        )
        inspected = json.loads(await tools["inspect_model_code_task"].ainvoke({}))
        assert inspected["residue_reference_catalog"] == reference_catalog()
        forged = copy.deepcopy(inspected)
        forged["residue_reference_catalog"]["references"][REF]["values"]["H"] = 181
        with pytest.raises(ValueError):
            validate_generated_result("inspect_model_code_task", forged, workspace=work, capability=cap)
        bad = json.loads(
            await tools["evaluate_model_code"].ainvoke(
                {"source_code": source(mutation={"H": 181}), "rationale": "deliberate invalid reference fixture"}
            )
        )
        assert bad["status"] == "development_failed" and bad["training_only_feedback"]["stage"] == "source_validation"
        assert bad["attempts_remaining"] == 1 and bad["submittable_sources"] == []
        good = json.loads(
            await tools["evaluate_model_code"].ainvoke(
                {"source_code": source(), "rationale": "correct source-bound reference"}
            )
        )
        assert good["status"] == "tested_not_published" and good["attempts_remaining"] == 0
        submitted = json.loads(
            await tools["submit_model_code"].ainvoke(
                {"source_sha256": good["source_sha256"], "rationale": "submit synthetic repaired model"}
            )
        )
        assert submitted["registered"] is False
    provenance = json.loads((work / "source_provenance.json").read_text())
    assert provenance["source_reference_audit"]["declared_literals_verified"]
    provenance["source_reference_audit"]["biological_interpretation_verified"] = True
    atomic_json(work / "source_provenance.json", provenance)
    with pytest.raises(ValueError):
        validate_generated_result("submit_model_code", submitted, workspace=work, capability=cap)
    # Host assessment ignores caller claims and recomputes source evidence.
    plugin = get_model_plugin("generated_python.v1")
    lc = ModelLifecycle(
        tmp_path / "registry",
        {
            "schema": 1,
            "scope": SCOPE,
            "plugin_id": plugin.id,
            "plugin_sha256": plugin.source_sha256,
            "training_sha256": digest(train),
            "validation_sha256": digest(valid),
            "minimum_skill": 0.0,
            "max_candidates": 1,
        },
    )
    mid = lc.submit(work / "candidate.json", session_id="develop", mode=mode)
    assessment = lc.assess(mid, training=train, validation=valid)
    assert assessment["passed"] and assessment["source_reference_audit"]["declared_literals_verified"]
    assert assessment["source_reference_audit"]["biological_interpretation_verified"] is False
    packet = lc.export(mid, tmp_path / "export")
    inputs = [{"id": x["id"], "sequence": x["sequence"]} for x in future]
    reuse_cap, reuse_work, reuse_config = bundle(tmp_path / "reuse", "reuse", inputs, packet)
    async with Client(reuse_config) as client:
        tools = await wrappers(
            client, reuse_cap, reuse_work, ["discover_generated_models", "predict_with_generated_model"]
        )
        catalog = json.loads(await tools["discover_generated_models"].ainvoke({}))
        assert catalog["models"][0]["source_reference_audit"] == assessment["source_reference_audit"]
        output = json.loads(await tools["predict_with_generated_model"].ainvoke({"model_id": mid}))
        assert output["rows"] == len(inputs)
    predictions = json.loads((reuse_work / "predictions.json").read_text())
    assert lc.record_reuse(
        mid, session_id="fresh_reuse", mode=mode, inputs=inputs, predictions=predictions, scope=SCOPE
    )["rows"] == len(inputs)
