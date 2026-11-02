import asyncio
import json

import pytest

from agent.session_store import SessionStore, SqliteBackend, session_has_content
from backend.web_v2.chat_history import include_research_content


def test_unused_session_is_a_draft():
    assert not session_has_content({
        'session_id': 'draft', 'created_at': 'today', 'status': 'idle',
        'history': [], 'plan': [], 'temp_files': [], 'chat_mode': 'science_agent',
        'default_llm_model_name': 'free-model', 'agent_session_dir': '/tmp/draft',
    })


@pytest.mark.parametrize('content', [
    {'history': [{'role': 'assistant', 'content': 'Recovered result'}]},
    {'last_user_text': 'Find homologues'},
    {'temp_files': ['sequence.fasta']},
    {'last_attachment_paths': ['structure.pdb']},
    {'plan': ['Train a model']},
    {'tool_executions': [{'name': 'blast'}]},
    {'waiting_for': 'clarification'},
    {'status': 'running'},
])
def test_content_does_not_require_a_title(content):
    assert session_has_content(content)


def test_disk_and_live_summaries_keep_owner_and_content(tmp_path):
    backend = SqliteBackend(str(tmp_path / 'sessions.sqlite'))
    store = SessionStore(backend, lambda state: state)

    async def check():
        draft = {'session_id': 'draft', 'owner_key': 'alice', 'history': []}
        await store.create(draft)
        await store.create({'session_id': 'other', 'owner_key': 'bob',
                            'history': [{'role': 'user', 'content': 'Private study'}]})
        assert [x['session_id'] for x in await store.list_summaries('alice')] == ['draft']
        assert not backend.list_summaries('alice')[0]['has_content']
        assert not (await store.list_summaries('alice'))[0]['has_content']
        draft['temp_files'] = ['sequence.fasta']
        # Other API processes see changes after persistence, never an unsaved cache.
        assert not (await store.list_summaries('alice'))[0]['has_content']
        await store.save('draft')
        assert (await store.list_summaries('alice'))[0]['has_content']
        assert backend.list_summaries('alice')[0]['has_content']
        assert await store.list_ids('alice') == ['draft']  # Nothing is deleted.
    asyncio.run(check())


def test_unused_scientist_folder_does_not_make_a_conversation(tmp_path):
    (tmp_path / '.supervisor.lock').touch()
    assert not include_research_content({'has_content': False}, tmp_path)['has_content']


def test_scientist_goal_is_retained_even_before_history_saved(tmp_path):
    (tmp_path / 'goal-1.json').write_text(json.dumps({'spec': {'objective': 'Predict protein solubility'}}))
    item = include_research_content({'history_size': 0, 'has_content': False}, tmp_path)
    assert item['has_content'] and item['title'].startswith('Predict protein')


def test_scientist_upload_is_retained(tmp_path):
    (tmp_path / 'inputs').mkdir()
    (tmp_path / 'inputs' / 'data.csv').write_text('sequence,activity\nAAA,1')
    assert include_research_content({'has_content': False}, tmp_path)['has_content']


def test_corrupt_goal_metadata_is_not_silently_hidden(tmp_path):
    (tmp_path / 'goal.json').write_text('{broken')
    assert include_research_content({'has_content': False}, tmp_path)['has_content']
