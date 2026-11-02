"""Actual MCP restarts and isolated numerical workers, without external LLM calls."""

import asyncio
import hashlib
import json

import pytest
from fastmcp import Client
from test_generated_recovery import GOOD, setup_bundle

from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json
from agent.rsi.tool_contracts import validate_generated_result

LEDGER = ".model_development.json"


async def checked(client, name, args, work, caps):
    result = (await client.call_tool(name, args)).data
    validate_generated_result(name, result, workspace=work, capability=caps)
    return result


@pytest.mark.asyncio
async def test_restarts_preserve_budget_successful_source_and_idempotent_submission(tmp_path):
    caps, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        inspected = await checked(client, "inspect_model_code_task", {}, work, caps)
        assert inspected["development_state"]["attempts_used"] == 0
        first = await checked(client, "evaluate_model_code", {"source_code": GOOD, "rationale": "first"}, work, caps)
    async with Client(config) as client:
        resumed = await checked(client, "inspect_model_code_task", {}, work, caps)
        assert resumed["development_state"]["attempts_remaining"] == 1
        assert resumed["development_state"]["submittable_sources"] == [first["source_sha256"]]
        bad = await checked(
            client, "evaluate_model_code", {"source_code": "def broken(!", "rationale": "second fails"}, work, caps
        )
        assert bad["attempts_remaining"] == 0 and bad["submittable_sources"] == [first["source_sha256"]]
        args = {"source_sha256": first["source_sha256"], "rationale": "submit proven source"}
        submitted = await checked(client, "submit_model_code", args, work, caps)
    recorded = {name: (work / name).read_bytes() for name in (LEDGER, "candidate.json", "source_provenance.json")}
    async with Client(config) as client:
        assert await checked(client, "submit_model_code", args, work, caps) == submitted
        with pytest.raises(Exception, match="budget exhausted"):
            await client.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "third forbidden"})
        with pytest.raises(Exception, match="arguments differ"):
            await client.call_tool("submit_model_code", {**args, "rationale": "different request"})
    assert all((work / name).read_bytes() == data for name, data in recorded.items())
    # Earlier inspection and tool events remain auditable after ledger progress.
    for name, value in (
        ("inspect_model_code_task", inspected),
        ("inspect_model_code_task", resumed),
        ("evaluate_model_code", first),
        ("evaluate_model_code", bad),
        ("submit_model_code", submitted),
    ):
        validate_generated_result(name, value, workspace=work, capability=caps)
    assert len(json.loads(recorded[LEDGER])["attempts"]) == 2
    assert not (work / "code_attempt_3.json").exists()


@pytest.mark.asyncio
async def test_two_actual_servers_share_one_budget(tmp_path):
    caps, work, config = setup_bundle(tmp_path)
    async with Client(config) as first, Client(config) as second:
        results = await asyncio.gather(
            *[
                checked(client, "evaluate_model_code", {"source_code": GOOD, "rationale": reason}, work, caps)
                for client, reason in ((first, "concurrent first"), (second, "concurrent second"))
            ]
        )
        assert sorted(value["attempts_remaining"] for value in results) == [0, 1]
        with pytest.raises(Exception, match="budget exhausted"):
            await first.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "third"})
    state = json.loads((work / LEDGER).read_text())
    assert [row["number"] for row in state["attempts"]] == [1, 2]
    assert all(row["status"] == "completed" for row in state["attempts"])


@pytest.mark.asyncio
async def test_real_server_exit_after_reservation_does_not_retry_unknown_execution(tmp_path):
    caps, work, config = setup_bundle(tmp_path)
    # Trusted test fault injection exits the MCP process before the numeric worker starts.
    path = caps / "generated_model.py"
    path.write_text(path.read_text() + "\ndef fit_candidate(*args, **kwargs):\n    import os\n    os._exit(77)\n")
    async with Client(config) as client:
        with pytest.raises(Exception, match="(?i)(closed|terminated|connection|transport|server|eof)"):
            await client.call_tool("evaluate_model_code", {"source_code": GOOD, "rationale": "process failure fixture"})
    original = (work / LEDGER).read_bytes()
    state = json.loads(original)
    assert len(state["attempts"]) == 1 and state["attempts"][0]["status"] == "reserved"
    assert not list(work.glob("code_attempt_*.json"))
    async with Client(config) as client:
        inspected = await checked(client, "inspect_model_code_task", {}, work, caps)
        assert inspected["development_state"]["pending_attempts"] == [1]
        assert inspected["development_state"]["attempts_remaining"] == 1
        for name, args in (
            ("evaluate_model_code", {"source_code": GOOD, "rationale": "must not retry"}),
            (
                "submit_model_code",
                {"source_sha256": hashlib.sha256(GOOD.encode()).hexdigest(), "rationale": "must not submit"},
            ),
        ):
            with pytest.raises(Exception, match="reconciliation required"):
                await client.call_tool(name, args)
    assert (work / LEDGER).read_bytes() == original


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["feedback", "task", "implementation", "ledger_deleted", "extra_attempt"])
async def test_restart_rejects_changed_or_unbound_evidence(tmp_path, mutation):
    caps, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        value = await checked(client, "evaluate_model_code", {"source_code": GOOD, "rationale": "first"}, work, caps)
    if mutation == "feedback":
        attempt = json.loads((work / "code_attempt_1.json").read_text())
        attempt["training_feedback"]["rmse"] = 7.0
        atomic_json(work / "code_attempt_1.json", attempt)
    elif mutation == "task":
        task = json.loads((caps / "task.json").read_text())
        task["scope"]["protein"] = "different task"
        atomic_json(caps / "task.json", task)
    elif mutation == "implementation":
        path = caps / "generated_model.py"
        path.write_text(path.read_text() + "\n# changed implementation\n")
    elif mutation == "ledger_deleted":
        (work / LEDGER).unlink()
    else:
        atomic_json(work / "code_attempt_3.json", {})
    async with Client(config) as client:
        with pytest.raises(Exception, match="changed|differs|coverage|unbound|reconciliation"):
            await client.call_tool("inspect_model_code_task", {})
    with pytest.raises((ValueError, FileNotFoundError)):
        validate_generated_result("evaluate_model_code", value, workspace=work, capability=caps)


@pytest.mark.asyncio
async def test_pending_submission_cannot_be_replayed_or_replace_partial_files(tmp_path):
    caps, work, config = setup_bundle(tmp_path)
    async with Client(config) as client:
        value = await checked(client, "evaluate_model_code", {"source_code": GOOD, "rationale": "first"}, work, caps)
    state = json.loads((work / LEDGER).read_text())
    state["submission"] = {
        "status": "reserved",
        "source_sha256": value["source_sha256"],
        "rationale": "interrupted submission",
    }
    atomic_json(work / ("code_ledger_" + digest(state) + ".json"), state)
    atomic_json(work / LEDGER, state)
    atomic_json(work / "candidate.json", {"partial_fixture": True})
    before = (work / "candidate.json").read_bytes()
    async with Client(config) as client:
        with pytest.raises(Exception, match="reconciliation required"):
            await client.call_tool(
                "submit_model_code", {"source_sha256": value["source_sha256"], "rationale": "interrupted submission"}
            )
    assert (work / "candidate.json").read_bytes() == before
