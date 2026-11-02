import asyncio
import json

import pytest

from agent.session_store import SessionStore, SqliteBackend, session_chat_mode
from backend.web_v2.chat_history import include_research_content
from backend.web_v2.chat_api._models import CreateSessionRequest


@pytest.mark.parametrize('state,expected', [
    ({}, 'science_agent'),
    ({'engine': 'graph', 'history': []}, 'science_agent'),
    ({'chat_mode': 'science_agent', 'history': [{'role': 'user', 'content': 'Explain proteins'}]}, 'science_agent'),
    ({'chat_mode': 'science_expert'}, 'science_expert'),
    ({'engine': 'graph', 'history': [{'role': 'user', 'content': 'Research proteins'}]}, 'science_expert'),
    ({'chat_mode': 'science_agent', 'history': [{'science_goal_id': 'goal-1'}]}, 'science_expert'),
])
def test_historical_mode_classification(state, expected):
    assert session_chat_mode(state) == expected


def test_scientist_goal_metadata_classifies_legacy_mixed_record(tmp_path):
    (tmp_path / 'goal.json').write_text(json.dumps({'spec': {'objective': 'Protein research'}}))
    summary = {'chat_mode': 'science_agent', 'has_content': True, 'history_size': 2}
    assert include_research_content(summary, tmp_path)['chat_mode'] == 'science_expert'
    assert summary['chat_mode'] == 'science_agent'  # Listing does not rewrite history.


def test_upload_only_scientist_draft_is_not_put_in_agent_history(tmp_path):
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs' / 'data.csv').write_text('sequence,label\nAA,1')
    assert include_research_content({'has_content': False}, tmp_path)['chat_mode'] == 'science_expert'


def test_mode_survives_persistence_and_owner_filter(tmp_path):
    backend = SqliteBackend(str(tmp_path / 'sessions.sqlite'))
    store = SessionStore(backend, lambda state: state)

    async def check():
        for sid,owner,mode in [('a','alice','science_agent'),('s','alice','science_expert'),('other','bob','science_expert')]:
            await store.create({'session_id': sid, 'owner_key': owner, 'chat_mode': mode})
        wanted = {'a': 'science_agent', 's': 'science_expert'}
        assert {x['session_id']:x['chat_mode'] for x in await store.list_summaries('alice')} == wanted
        store._mem.clear()
        assert {x['session_id']:x['chat_mode'] for x in await store.list_summaries('alice')} == wanted
    asyncio.run(check())


def test_creation_accepts_only_known_modes_and_defaults_agent():
    assert CreateSessionRequest().chat_mode == 'science_agent'
    assert CreateSessionRequest(chat_mode='science_expert').chat_mode == 'science_expert'
    with pytest.raises(ValueError):
        CreateSessionRequest(chat_mode='unknown')
