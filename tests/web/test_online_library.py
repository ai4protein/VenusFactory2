"""Conversation-scoped Library permissions and non-destructive online deletion."""
import asyncio
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from backend.web_v2 import online_library_api as api, online_visibility as visibility
from backend.web_v2.chat_api import _shared as shared, sessions
from backend.web_v2.research_jobs import ResearchJobs


@pytest.fixture
def library(tmp_path, monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    hidden = visibility.OnlineVisibility(tmp_path / 'access.sqlite3')
    monkeypatch.setattr(visibility, 'OnlineVisibility', lambda: hidden)
    monkeypatch.setattr(shared, '_runtime_mode', lambda: 'online' if visibility.is_online() else 'local')
    monkeypatch.setattr(sessions, '_runtime_mode', shared._runtime_mode)
    monkeypatch.setattr(shared, '_SESSION_TOKEN_SECRET', 'test-secret')
    monkeypatch.setattr(shared, '_session_owner_key_for_request', lambda request: hashlib.sha256((request.client.host + '|' + request.headers.get('user-agent', '')).encode()).hexdigest())

    states = {}
    class Store:
        async def get(self, identity): return states.get(identity)
        async def delete(self, identity): states.pop(identity, None)
    store = Store()
    monkeypatch.setattr(shared, '_session_store', store)
    monkeypatch.setattr(sessions, '_session_store', store)
    service = SimpleNamespace(state_root=tmp_path/'science', session_folder=lambda sid: tmp_path/'science'/sid, has_unfinished_goal=lambda sid: False)
    monkeypatch.setattr(api, 'service', service)
    monkeypatch.setattr(sessions, 'goal_service', service)
    monkeypatch.setattr(sessions, '_record_access_event', lambda *a: None)
    locks = {}
    async def lock(sid): return locks.setdefault(sid, asyncio.Lock())
    monkeypatch.setattr(sessions, '_get_lock', lock)
    monkeypatch.setattr(sessions, '_SESSIONS_GUARD', asyncio.Lock())
    monkeypatch.setattr(sessions, '_SESSION_LOCKS', locks)
    monkeypatch.setattr(sessions, '_SESSION_CANCEL_FLAGS', {})

    # Use the actual ResearchJobs ownership/deletion methods with an in-memory job index.
    report_rows = {}
    class Records:
        def resolve(self, source, identity, owner):
            if report_rows.get(identity) != owner: raise FileNotFoundError()
            return identity
        def entries(self, source, owner): return [{'identity': key} for key, value in report_rows.items() if value == owner]
        def register(self, *args): return {'job_number':'R-test'}
    jobs = ResearchJobs.__new__(ResearchJobs); jobs.root = tmp_path/'reports'; jobs.records = Records(); jobs.queue = None
    monkeypatch.setattr(api, 'ResearchJobs', lambda: jobs)
    app = FastAPI(); app.include_router(api.router)
    # These legacy tests exercise conversation ACLs under online identity rules.
    # The production router's mode gate is tested separately below.
    def origin_only(request: Request):
        api.require_same_origin(request)
    app.dependency_overrides[api.library_origin] = origin_only
    app.add_api_route('/api/chat/sessions/{session_id}', sessions.delete_session, methods=['DELETE'])
    client = TestClient(app, headers={'user-agent':'biology-browser'})
    # Both browsers deliberately share the same IP and user agent.
    other = TestClient(app, headers={'user-agent':'biology-browser'})
    owner = hashlib.sha256(b'testclient|biology-browser').hexdigest()
    root = tmp_path/'agent'/'chat-a'; root.mkdir(parents=True)
    (root/'result.csv').write_text('sequence,value\nACDE,1\n')
    (root/'.env').write_text('test credential')
    (root/'request.json').write_text('test internal metadata')
    states['chat-a'] = {'session_id':'chat-a', 'agent_session_dir':str(root), 'owner_key':owner, 'session_token_hash':shared._hash_session_token('token-a'), 'chat_mode':'science_agent'}
    states['chat-b'] = {**states['chat-a'], 'session_id':'chat-b', 'session_token_hash':shared._hash_session_token('token-b')}
    ref = {'source':'chat', 'identity':'chat-a', 'token':'token-a', 'key':'agent/result.csv'}
    return SimpleNamespace(client=client, other=other, hidden=hidden, root=root, states=states, ref=ref, jobs=jobs, report_rows=report_rows, tmp=tmp_path)


def test_lists_only_authorized_files_and_downloads(library):
    c = library.client
    response = c.post('/api/library/files', json={'sessions':[library.ref]})
    assert response.status_code == 200
    assert [i['name'] for i in response.json()['items']] == ['result.csv']
    assert response.headers['cache-control'] == 'private, no-store'
    assert not any(secret in response.text for secret in ['token-a', '.env', 'request.json', str(library.tmp)])
    content = c.post('/api/library/content', json=library.ref)
    assert content.status_code == 200 and content.content == (library.root/'result.csv').read_bytes()


@pytest.mark.parametrize('token', ['', 'token-b', 'wrong', '非 ASCII'])
def test_shared_ip_does_not_grant_other_browser_access(library, token):
    ref = {**library.ref, 'token':token}
    listed = library.other.post('/api/library/files', json={'sessions':[ref]}).json()['items']
    assert listed == []
    assert library.other.post('/api/library/content', json=ref).status_code in {401,403}
    assert library.other.post('/api/library/archive', json={'files':[ref]}).status_code in {401,403}


@pytest.mark.parametrize('key', ['agent/../outside.csv', 'agent/.env', 'agent/request.json', 'agent/no-file.csv', 'other/result.csv', 'agent/C:/secret', 'agent/..\\secret'])
def test_paths_and_control_files_are_not_exposed(library, key):
    ref = {**library.ref, 'key':key}
    assert library.client.post('/api/library/content', json=ref).status_code == 404
    assert library.client.post('/api/library/archive', json={'files':[library.ref, ref]}).status_code == 404


def test_archive_verifies_every_entry_and_preserves_contents(library):
    response = library.client.post('/api/library/archive', json={'files':[library.ref, library.ref]})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.namelist() == ['chat/chat-a/agent/result.csv']
        assert archive.read(archive.namelist()[0]) == (library.root/'result.csv').read_bytes()
    bad = {**library.ref, 'identity':'chat-b'}
    assert library.client.post('/api/library/archive', json={'files':[library.ref,bad]}).status_code == 403


def test_online_delete_revokes_old_links_but_keeps_disk_and_state(library):
    before = (library.root/'result.csv').read_bytes()
    response = library.client.delete('/api/chat/sessions/chat-a', headers={'x-session-access-token':'token-a'})
    assert response.status_code == 200
    assert (library.root/'result.csv').read_bytes() == before
    assert 'chat-a' in library.states
    assert library.hidden.is_hidden('chat','chat-a')
    assert library.hidden.blocks_file(library.root/'result.csv')
    assert not library.hidden.blocks_file(library.root.with_name('chat-ab')/'result.csv')
    assert library.client.post('/api/library/files',json={'sessions':[library.ref]}).json()['items'] == []
    for path, data in [('content',library.ref), ('archive',{'files':[library.ref]})]:
        assert library.client.post('/api/library/'+path,json=data).status_code == 404
    assert library.client.delete('/api/chat/sessions/chat-a',headers={'x-session-access-token':'token-a'}).status_code == 404


def test_local_mode_can_still_inspect_retained_files(library, monkeypatch):
    library.hidden.hide('chat','chat-a',[library.root])
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    assert library.client.post('/api/library/content', json=library.ref).status_code == 200


def test_cross_origin_is_rejected(library):
    assert library.client.post('/api/library/files', json={}, headers={'origin':'https://unrelated.invalid'}).status_code == 403


def test_report_cookie_ownership_and_soft_delete(library, monkeypatch):
    import mcp_tools.common.common_utils as common
    monkeypatch.setattr(common,'get_web_v2_root_dir',lambda:library.tmp/'web')
    library.client.post('/api/library/files',json={})
    # Derive the real owner exactly as browser_owner does, using its cookie name.
    from backend.web_v2.job_access import COOKIES
    cookie = library.client.cookies.get(COOKIES['report'])
    who = hashlib.sha256(cookie.encode()).hexdigest()
    rid = 'a'*32; library.report_rows[rid] = who
    folder = library.jobs.folder(rid); folder.mkdir(parents=True)
    request = {'owner':who,'title':'Test protein','created':100,'payload':{'model_id':'free','analyses':[],'sequence':'ACDE'}}
    (folder/'request.json').write_text(json.dumps(request))
    (folder/'state.json').write_text(json.dumps({'status':'completed','report':'# Protein report'}))
    ref = {'source':'report','identity':rid,'key':'report.md'}
    assert library.client.post('/api/library/content',json=ref).text == '# Protein report'
    assert library.other.post('/api/library/content',json=ref).status_code == 404
    assert library.other.post('/api/library/files',json={}).json()['items'] == []
    library.jobs.delete_online(who,rid)
    assert (folder/'state.json').is_file() and (folder/'request.json').is_file()
    assert library.client.post('/api/library/content',json=ref).status_code == 404
    assert library.client.post('/api/library/files',json={}).json()['items'] == []
    with pytest.raises(FileNotFoundError): library.jobs.get(who,rid)


def test_hidden_marker_persists_and_does_not_touch_files(tmp_path):
    db = tmp_path/'access.db'; folder=tmp_path/'conversation'; folder.mkdir(); artifact=folder/'result.txt'; artifact.write_bytes(b'unchanged')
    first = visibility.OnlineVisibility(db); first.hide('chat','abc',[folder])
    second = visibility.OnlineVisibility(db)
    assert second.hidden_ids('chat') == {'abc'} and second.blocks_file(artifact)
    assert artifact.read_bytes() == b'unchanged'


def test_delete_requires_token_and_cannot_delete_active_science(library, monkeypatch):
    assert library.other.delete('/api/chat/sessions/chat-a').status_code == 401
    assert not library.hidden.is_hidden('chat','chat-a')
    monkeypatch.setattr(sessions.goal_service,'has_unfinished_goal',lambda sid: True)
    assert library.client.delete('/api/chat/sessions/chat-a',headers={'x-session-access-token':'token-a'}).status_code == 409
    assert not library.hidden.is_hidden('chat','chat-a') and (library.root/'result.csv').is_file()


def test_expired_session_token_revokes_library_access(library):
    library.states['chat-a']['token_expires_at'] = '2000-01-01T00:00:00+00:00'
    assert library.client.post('/api/library/files',json={'sessions':[library.ref]}).json()['items'] == []
    assert library.client.post('/api/library/content',json=library.ref).status_code == 401


def test_generic_urls_cannot_bypass_conversation_authentication(library, monkeypatch):
    import mcp_tools.common.common_utils as common
    import backend.web_v2.science_goals as goals
    monkeypatch.setattr(common,'get_temp_outputs_base_dir',lambda:library.tmp)
    monkeypatch.setattr(common,'get_web_v2_root_dir',lambda:library.tmp/'web')
    monkeypatch.setattr(goals,'service',SimpleNamespace(state_root=library.tmp/'science'))
    with pytest.raises(FileNotFoundError): visibility.guard_generic_file(library.root/'result.csv')
    # Sibling names must not be mistaken for an owned conversation root.
    visibility.guard_generic_file(library.tmp/'agent-other'/'result.csv')
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    visibility.guard_generic_file(library.root/'result.csv')


def test_symlink_cannot_escape_conversation(library):
    outside = library.tmp/'outside.txt'; outside.write_text('private')
    link = library.root/'link.txt'
    try: link.symlink_to(outside)
    except OSError: pytest.skip('Symlink creation is unavailable on this host')
    ref = {**library.ref, 'key':'agent/link.txt'}
    assert library.client.post('/api/library/content',json=ref).status_code == 404
    listed=library.client.post('/api/library/files',json={'sessions':[library.ref]}).json()['items']
    assert 'link.txt' not in [row['name'] for row in listed]


@pytest.mark.parametrize('path,payload', [('files', {}), ('content', {'source':'chat','identity':'chat-a','key':'agent/result.csv'}), ('archive', {'files':[{'source':'chat','identity':'chat-a','key':'agent/result.csv'}]})])
def test_online_library_routes_are_unavailable(monkeypatch, path, payload):
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    app=FastAPI(); app.include_router(api.router)
    with TestClient(app) as client:
        response=client.post('/api/library/'+path,json=payload)
        assert response.status_code==403
        assert response.json()['detail']=='Library is unavailable in online mode.'


def test_local_library_router_still_serves_owned_files(library, monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    library.client.app.dependency_overrides.clear()
    assert library.client.post('/api/library/content',json=library.ref).content == (library.root/'result.csv').read_bytes()
