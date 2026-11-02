"""Candidate handoff stays replayable and cannot stand in for scientific acceptance."""

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from test_generated_context import CODE
from test_yeast_population_controls import example

from agent.rsi.development_handoff import FIELD, POLICY, description, receipt_fields
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
from run_paired_generated import make_bundle


@pytest.mark.parametrize("policy", [None, False, 1, {}, "unknown"])
def test_unknown_handoff_fails_closed(policy):
    with pytest.raises(ValueError, match="handoff"):
        description({"phase": "develop", FIELD: policy})


def test_phase_and_legacy_contract():
    assert description({"phase": "develop"}) is None
    assert receipt_fields({"phase": "develop"}, "a" * 64) == {}
    with pytest.raises(ValueError, match="handoff"):
        receipt_fields({"phase": "reuse", FIELD: POLICY}, "a" * 64)


def test_reuse_bundle_does_not_declare_development_handoff(tmp_path):
    scope, _, _ = example()
    versions, version, caps = make_bundle(
        tmp_path, "reuse", {}, scope, {"provider": "synthetic"}, plugin_id="generated_context.v1"
    )
    versions.verify(version)
    assert FIELD not in json.loads((caps / "task.json").read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_real_mcp_submission_and_historical_observations(tmp_path, legacy):
    scope, fit, score = example()
    rows = fit + score
    work = tmp_path / "workspace"
    work.mkdir()
    atomic_json(work / "training.json", rows)
    versions, version, caps = make_bundle(
        tmp_path / "capability", "develop", {"training.json": rows}, scope,
        {"provider": "synthetic"}, plugin_id="generated_context.v1",
    )
    task = json.loads((caps / "task.json").read_text())
    assert task[FIELD] == POLICY
    versions.verify(version)
    if legacy:
        # Emulate the old portable bundle: no policy declaration or new module.
        caps = Path(shutil.copytree(caps, tmp_path / "legacy_capability"))
        caps.chmod(0o700)
        task.pop(FIELD)
        atomic_json(caps / "task.json", task)
        (caps / "development_handoff.py").unlink()
    config = {"mcpServers": {"model": {"command": sys.executable,
              "args": ["-u", str(caps / "generated_tools.py")],
              "env": {"VENUS_RSI_WORKSPACE": str(work)}}}}

    def validate(name, value, workspace=work):
        validate_generated_result(name, value, workspace=workspace, capability=caps)

    async with Client(config) as client:
        async def call(name, args):
            result = await client.call_tool(name, args)
            validate(name, result.data)
            return result.data

        before = await call("inspect_model_code_task", {})
        assert before.get(FIELD) == description(task)
        trial = await call("evaluate_model_code", {"source_code": CODE, "rationale": "synthetic test"})
        args = {"source_sha256": trial["source_sha256"], "rationale": "tested candidate"}
        submitted = await call("submit_model_code", args)
        assert submitted["status"] == "candidate" and submitted["registered"] is False
        expected = {"artifact": "candidate.json", "source_sha256": trial["source_sha256"],
                    "training_rows": len(rows), "status": "candidate", "registered": False,
                    **receipt_fields(task, trial["source_sha256"])}
        assert submitted == expected
        if not legacy:
            assert submitted[FIELD]["state"] == "awaiting_host_validation"
            assert submitted[FIELD]["scientific_acceptance"] == "pending"
            assert submitted[FIELD]["next_action"] == "return_final_summary"
        assert await call("submit_model_code", args) == submitted
        after = await call("inspect_model_code_task", {})
        assert after["development_state"]["submission_status"] == "completed"
        assert await call("discover_generated_models", {}) == {"models": []}
        with pytest.raises(Exception, match="one submitted candidate permitted"):
            await client.call_tool("evaluate_model_code", {"source_code": CODE, "rationale": "repeat"})
    # Status observations bind immutable snapshots, not the final mutable state.
    validate("inspect_model_code_task", before)
    validate("evaluate_model_code", trial)
    validate("inspect_model_code_task", after)
    validate("submit_model_code", submitted)
    assert not (work / "registered").exists()
    assert not (work / "capability.json").exists()

    forged_info = copy.deepcopy(before)
    forged_info[FIELD] = {"schema": POLICY, "after_success": "register_without_review"}
    with pytest.raises(ValueError, match="disagrees"):
        validate("inspect_model_code_task", forged_info)

    forged = copy.deepcopy(submitted)
    forged["registered"] = True
    forged[FIELD] = {"schema": POLICY, "state": "accepted"}
    altered = Path(shutil.copytree(work, tmp_path / "forged_workspace"))
    ledger = json.loads((altered / ".model_development.json").read_text())
    ledger["submission"]["receipt"] = forged
    atomic_json(altered / ".model_development.json", ledger)
    atomic_json(altered / ("code_ledger_" + digest(ledger) + ".json"), ledger)
    with pytest.raises(ValueError, match="disagrees"):
        validate("submit_model_code", forged, workspace=altered)
