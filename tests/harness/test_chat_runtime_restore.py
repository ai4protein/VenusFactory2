import os
import subprocess
import sys
from pathlib import Path


def test_native_restore_never_rebuilds_legacy_model_or_toolbox(monkeypatch):
    from agent import chat_agent
    def unexpected(*args, **kwargs):
        raise AssertionError('Legacy runtime rebuilt for a native conversation')
    monkeypatch.setattr(chat_agent, 'make_llm', unexpected)
    monkeypatch.setattr(chat_agent, 'get_tools', unexpected)
    state = {'agent_execution': {'selection': {'harness': 'deepseek-harness'}},
             'agent_messages': [{'role': 'user', 'content': 'history'}], 'status': 'completed'}
    assert chat_agent.ensure_runtime_state(state) is state
    assert state['agent_messages'][0]['content'] == 'history'


def test_legacy_restore_still_builds_its_runtime(monkeypatch):
    from agent import chat_agent
    called = []
    def legacy(*args, **kwargs):
        called.append(True)
        raise RuntimeError('legacy constructor reached')
    monkeypatch.setattr(chat_agent, 'make_llm', legacy)
    import pytest
    with pytest.raises(RuntimeError, match='legacy constructor reached'):
        chat_agent.ensure_runtime_state({})
    assert called == [True]


def test_chat_boot_does_not_import_scientific_toolbox_or_torch():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c',
        "import backend.web_v2.chat_api._stream_agent; import sys; "
        "assert 'torch' not in sys.modules; assert 'mcp_tools.tools.tools_agent_hub' not in sys.modules"],
        cwd=root, env=dict(os.environ), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
