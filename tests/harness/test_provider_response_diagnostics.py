"""Fixtures cover failed stream diagnostics and rejecting data before sandbox admission."""
import json
from pathlib import Path
import httpx
import pytest
from agent.harness.gateway import EvidenceGateway
from agent.harness.provider_response_diagnostics import response_evidence
from agent.rsi.engine.model_gateway import GatewayPolicy

def frame(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"

def test_diagnostics_never_keep_provider_content_or_identifiers():
    raw = frame({"error": {"message": "secret-prompt", "code": "secret-token"},
                 "model": "private-model", "choices": [{"finish_reason": "private-secret",
                 "delta": {"content": "private-content", "reasoning": "private-reasoning",
                           "tool_calls": [{"function": {"name": "private-tool", "arguments": "private-args"}}]}}],
                 "usage": {"prompt_tokens": 10}}) + b"data: [DONE]\n\n"
    evidence = response_evidence(raw, True)
    assert evidence["provider_error"] and evidence["done"] and evidence["usage_present"]
    assert evidence["tools_present"] and evidence["finish_reasons"] == ["unknown"]
    assert not any(word in json.dumps(evidence) for word in ("secret", "private"))

@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["response_completion", "tool_interface"])
async def test_gateway_keeps_failure_stage_and_provider_usage(tmp_path, stage):
    message = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "private-id", "type": "function",
         "function": {"name": "undeclared", "arguments": "{}"}}]}
    if stage == "response_completion":
        raw = frame({"error": {"message": "private-prompt"}, "choices": [
            {"index": 0, "delta": {}, "finish_reason": "error"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    else:
        raw = json.dumps({"model": "test", "choices": [{"index": 0, "message": message,
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}).encode()
    log = tmp_path / "model_requests.jsonl"
    async with EvidenceGateway(policy=GatewayPolicy(model="test", mode="science_agent", scientific_tools=set()),
        upstream_url="https://provider.example/v1", api_key="private-key", output=log,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw))) as relay:
        async with httpx.AsyncClient() as client:
            result = await client.post(relay.base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + relay.token},
                json={"model": "test", "messages": [{"role": "user", "content": "private-prompt"}],
                      "stream": stage == "response_completion"})
    assert result.status_code == 502
    records = [json.loads(line) for line in log.read_text().splitlines()]
    error = next(row for row in records if row["event"] == "transport_error")
    assert error["validation_stage"] == stage
    assert records[-1]["usage"] and not records[-1]["completed"]
    assert "private" not in log.read_text()

@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".pdb", ".pdf", ".ipynb"])
async def test_data_file_is_rejected_before_research_runner(tmp_path, suffix):
    from agent.harness.research_workspace import ResearchWorkspace
    called = []
    async def runner(*args, **kwargs):
        called.append(args)
        raise AssertionError("Data files must not start a runner")
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", {}, runner=runner)
    (service.workspace / ("data" + suffix)).write_text("observed,data\n1,2\n")
    with pytest.raises(ValueError, match="saved .py script"):
        await service.run_research_python("data" + suffix)
    assert not called and not (service.control / "python").exists()

@pytest.mark.asyncio
async def test_data_rejection_precedes_sandbox_resources(tmp_path, monkeypatch):
    from agent.harness import session_sandbox
    def forbidden():
        raise AssertionError("No sandbox or job allocation for a data file")
    monkeypatch.setattr(session_sandbox, "assert_active_job", forbidden)
    with pytest.raises(ValueError, match="saved .py script"):
        await session_sandbox.execute(tmp_path, tmp_path / "control", "python", "data.csv")

def test_valid_python_path_and_shell_are_preserved():
    from agent.harness.session_sandbox import validate_execution
    validate_execution("python", "src/analyze.PY", 30)
    validate_execution("shell", "cat data.csv", 30)
