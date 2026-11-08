"""Shared pytest configuration and fixtures."""
import sys
import os
from pathlib import Path
import pytest

# Unit tests must not load a developer's provider keys during module collection.
os.environ.setdefault('PYTHON_DOTENV_DISABLED', '1')
os.environ.setdefault('VENUS_MODEL_CONFIG', str(Path(__file__).resolve().parents[1] / 'model-config.yaml.example'))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SRC_DIR = str(ROOT / 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


@pytest.fixture(autouse=True)
def isolate_cached_configuration(monkeypatch):
    """Do not retain one test's environment-derived mode or provider settings."""
    import config
    monkeypatch.setattr(config, "_config", None)


@pytest.fixture
def durable_chat_driver(monkeypatch, tmp_path):
    """Drive real admission/events with a deterministic worker in the test loop.

    Process independence is covered separately by test_durable_tasks; this bridge
    lets HTTP tests retain their mocked inference without spawning an unmocked LLM.
    """
    import asyncio
    from backend.web_v2 import chat_runs
    from backend.web_v2.chat_api import _hooks_runtime, messages
    running = {}
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART', '0')
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT', str(tmp_path / 'chat-queue'))
    original_response = chat_runs.response

    def install(store, stream=None):
        monkeypatch.setattr(_hooks_runtime, '_session_store', store)
        async def execute(q, identity, sid):
            row = q.get(identity)
            claimed = q.claim_next(row['namespace'], 'api')
            assert claimed and claimed['id'] == identity
            lease = claimed['lease']
            async def keep_lease():
                while True:
                    q.heartbeat(identity, lease)
                    await asyncio.sleep(.2)
            heartbeat = asyncio.create_task(keep_lease())
            lock = await _hooks_runtime._get_lock(sid)
            try:
                async with lock:
                    state = await store.get(sid)
                    data = row['metadata']['agent_input']
                    state['status'] = 'running'
                    await store.save(sid)
                    fn = stream or messages._stream_agent
                    async for frame in fn(state, data['text'], data['paths'], retry=data['retry'], **({'interaction_response':data['interaction_response']} if data.get('interaction_response') else {})):
                        q.heartbeat(identity, lease)
                        q.append_frame(identity, frame, lease)
                    if state.get('status') == 'running': state['status'] = 'completed'
                    await store.save(sid)
                q.finish(identity, 'completed', lease=lease)
            except BaseException:
                q.finish(identity, 'failed', lease=lease)
                raise
            finally:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)

        def response(q, identity, owner, sid, cursor=''):
            if identity not in running:
                running[identity] = asyncio.create_task(execute(q, identity, sid))
            return original_response(q, identity, owner, sid, cursor)
        monkeypatch.setattr(chat_runs, 'response', response)
    return install
