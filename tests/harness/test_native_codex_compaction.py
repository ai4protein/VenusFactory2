"""Opt-in real Codex regression using a local scripted provider, never an API key.

VENUS_TEST_NATIVE_CODEX=1 enables this test when Codex and bubblewrap are installed.
VENUS_TEST_RUNTIME_ROOT optionally points to a separate managed runtime installation.
"""
import json
import os
import shutil
from pathlib import Path

import httpx
import pytest

from agent.harness import Budget, ModelBinding, Profile, Task
from agent.harness.host import execute
from agent.harness.installation import default_runtimes
from agent.harness.responses_bridge import ResponsesChatBridge

pytestmark = pytest.mark.skipif(
    os.environ.get("VENUS_TEST_NATIVE_CODEX") != "1", reason="Native Codex verification is opt-in"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["usage", "overflow"])
async def test_native_context_handling_preserves_sdk_behavior(tmp_path, monkeypatch, trigger):
    root = Path(os.environ.get("VENUS_TEST_RUNTIME_ROOT", Path(__file__).resolve().parents[2]))
    runtimes = default_runtimes(root)
    assert shutil.which("bwrap") and Path(runtimes["codex_bin"]).is_file()
    workspace, capability = tmp_path / "work", tmp_path / "caps"
    workspace.mkdir()
    capability.mkdir()
    (capability / "generated_tools.py").write_text(
        "from fastmcp import FastMCP\nfrom pathlib import Path\nimport json\n"
        "mcp = FastMCP('venusfactory')\n@mcp.tool()\n"
        "def probe(step: int) -> dict:\n"
        "    with Path('/workspace/receipts.jsonl').open('a') as f:\n"
        "        f.write(json.dumps({'step': step}) + '\\n')\n"
        "    return {'observed_step': step}\n"
        "if __name__ == '__main__': mcp.run()\n"
    )
    seen, normal, summaries = [], [], []
    rejected = False

    def provider(request):
        nonlocal rejected
        payload = json.loads(request.content)
        seen.append(payload)
        names = [item['function']['name'] for item in payload.get('tools', [])]
        if trigger == 'overflow' and names and len(normal) == 1 and not rejected:
            rejected = True
            return httpx.Response(400, json={'error': {
                'type': 'invalid_request_error', 'code': 'context_length_exceeded',
                'message': 'The maximum context length has been exceeded.'}})
        identity = 'fixture-' + str(len(seen))
        if not names:
            summaries.append(payload)
            message = {'role': 'assistant', 'content': 'The first probe completed. Run the second probe then report done.'}
            finish = 'stop'
        else:
            normal.append(payload)
            if len(normal) <= 2:
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': identity, 'type': 'function', 'function': {
                        'name': names[0], 'arguments': json.dumps({'step': len(normal)})}}]}
                finish = 'tool_calls'
            else:
                message = {'role': 'assistant', 'content': 'done'}
                finish = 'stop'
        prompt_tokens = 125000 if trigger == 'usage' and len(seen) == 1 else 500
        return httpx.Response(200, json={
            'id': identity, 'object': 'chat.completion', 'created': 1, 'model': payload['model'],
            'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': 20, 'total_tokens': prompt_tokens + 20}})

    class FixtureBridge(ResponsesChatBridge):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(provider), **kwargs)

    monkeypatch.setattr('agent.harness.responses_bridge.ResponsesChatBridge', FixtureBridge)
    profile = Profile('science_agent', 'codex',
        ModelBinding('offline', 'fixture-model', 'http://127.0.0.1:1/v1', context_tokens=131072),
        Budget(wall_seconds=90, request_seconds=15, model_calls=8, tool_calls=4, output_tokens=256, thinking=None))
    task = Task(workspace, capability, tmp_path / 'execution',
                'Call probe with step 1, then probe with step 2, then report done.',
                'Deterministic offline runtime test.', frozenset({'probe'}), phase='probe')
    result = await execute(profile, task, runtimes, credential='offline-fixture-only')
    if trigger == 'overflow':
        # The pinned SDK terminates on an unexpected provider rejection. Preserve
        # that native outcome; the platform must not fabricate a summary/retry.
        from agent.harness.failure_summary import public_failure

        assert result['status'] == 'runtime_error' and result['sandbox_stopped'], result
        assert result['tool_calls'] == 1 and result['native_compaction_notices'] == 0
        assert len(seen) == 2 and not summaries and rejected
        assert public_failure(task.output / 'model_requests.jsonl')['code'] == 'context_limit'
        assert not result['audit']['requests']['passed']
        return
    assert result['status'] == 'completed', result
    assert result['sandbox_stopped'] and result['audit']['passed']
    assert len(normal) == 3 and len(summaries) == 1
    assert result['tool_calls'] == 2 and result['native_compaction_notices'] == 1
    assert rejected == (trigger == 'overflow')
    assert 'another LLM that will resume' in str(summaries[0]['messages'][-1])
    rows = [json.loads(line) for line in (task.output / 'events.jsonl').read_text().splitlines()]
    notice = next(i for i, row in enumerate(rows) if row['kind'] == 'context_compaction')
    after = rows[notice + 1:]
    starts = {r['tool_call_id'] for r in after if r['kind'] == 'tool_call_start'}
    assert any(r['kind'] == 'tool_result' and not r['is_error'] and r['tool_call_id'] in starts for r in after)
    assert all(r['session_id'] == result['session_id'] for r in rows)
    assert rows[-1]['kind'] == 'native_terminal' and rows[-1]['gateway_calls'] == len(seen)
    assert [json.loads(line) for line in (workspace / 'receipts.jsonl').read_text().splitlines()] == [
        {'step': 1}, {'step': 2}]
