"""Real MCP rejection/correction stays within the exact evidence contract."""

import json
import shutil
import sys
import time
from pathlib import Path

import pytest
from fastmcp import Client

from agent.rsi.engine.langgraph_tools import build_tools

ROOT = Path(__file__).resolve().parents[2]
FACTS = {"job_status": "completed", "full_paper_reproduced": False, "wet_lab_validation": False}


def setup(tmp_path):
    cap, work = tmp_path / "cap", tmp_path / "work"
    cap.mkdir()
    work.mkdir()
    shutil.copy2(ROOT / "src/agent/rsi/engine/case_review_tools.py", cap / "generated_tools.py")
    (cap / "evidence.json").write_text(json.dumps({"job_id": "test", "facts": FACTS}))
    config = {
        "mcpServers": {
            "review": {
                "command": sys.executable,
                "args": ["-u", str(cap / "generated_tools.py")],
                "env": {"VENUS_RSI_WORKSPACE": str(work), "PYTHONPATH": str(ROOT / "src")},
            }
        }
    }
    return cap, work, config


async def wrapped(client, cap, work):
    counter, events = {"tool_calls": 0, "tool_errors": 0}, []
    tools = await build_tools(
        client,
        allowed_tools={"inspect_case_evidence", "submit_evidence_review"},
        workspace=work,
        capability=cap,
        deadline=time.monotonic() + 30,
        max_tool_calls=5,
        result=counter,
        event=events.append,
        validate_results=False,
    )
    return {t.name: t for t in tools}, counter, events


@pytest.mark.asyncio
async def test_wrapper_keeps_session_alive_for_exact_claim_correction(tmp_path):
    cap, work, config = setup(tmp_path)
    async with Client(config) as client:
        tools, counter, events = await wrapped(client, cap, work)
        original = json.loads(await tools["inspect_case_evidence"].ainvoke({}))
        rejected = json.loads(
            await tools["submit_evidence_review"].ainvoke(
                {"claims_json": json.dumps({**original["facts"], "job_id": "test"})}
            )
        )
        assert rejected["accepted"] is False and rejected["attempts_remaining"] == 1
        assert not (work / "review.json").exists()
        accepted = json.loads(
            await tools["submit_evidence_review"].ainvoke({"claims_json": json.dumps(original["facts"])})
        )
        assert accepted["accepted"] is True and accepted["attempts"] == 2
        assert json.loads((work / "review.json").read_text())["claims"] == FACTS
        assert counter == {"tool_calls": 3, "tool_errors": 0}
        assert len(events) == 6
        assert json.loads((work / "review_attempt_1.json").read_text())["feedback"]["accepted"] is False


@pytest.mark.asyncio
async def test_two_rejections_do_not_relax_scientific_contract(tmp_path):
    cap, work, config = setup(tmp_path)
    async with Client(config) as client:
        tools, counter, _ = await wrapped(client, cap, work)
        for value in ["invalid JSON", json.dumps({**FACTS, "wet_lab_validation": True})]:
            output = json.loads(await tools["submit_evidence_review"].ainvoke({"claims_json": value}))
            assert output["accepted"] is False
        assert output["attempts_remaining"] == 0
        with pytest.raises(Exception, match="budget exhausted"):
            await tools["submit_evidence_review"].ainvoke({"claims_json": json.dumps(FACTS)})
        assert not (work / "review.json").exists()
        assert counter == {"tool_calls": 3, "tool_errors": 1}
