import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest
from backend.web_v2.job_queue import JobQueue, launch, process_token, stop_process_tree
from backend.web_v2.task_store import SubmissionConflict, LeaseLost
from backend.web_v2.session_runtime import SharedSessionLock, persistent_secret


@pytest.fixture
def q(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    for family in ('CPU','GPU','API','RESEARCH'):monkeypatch.setenv('VENUS_WORKERS_'+family,'1')
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS','1')
    q=JobQueue()
    yield q
    for path in q.root.glob('pools/*/supervisor.json'):
        row=json.loads(path.read_text());stop_process_tree(row['pid'],row['token'])


def test_deleted_conversation_between_lookup_and_lock_is_not_submitted(q, monkeypatch):
    from types import SimpleNamespace
    from fastapi import HTTPException, Request
    from backend.web_v2.chat_api import _hooks_runtime
    from backend.web_v2.chat_api._models import ChatStreamRequest
    from backend.web_v2.chat_runs import submit
    async def missing(_): return None
    monkeypatch.setattr(_hooks_runtime, '_session_store', SimpleNamespace(get=missing))
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    request = Request({'type': 'http', 'headers': [], 'client': ('127.0.0.1', 1)})
    with pytest.raises(HTTPException) as error:
        asyncio.run(submit({'session_id': 'deleted-before-lock'}, request, ChatStreamRequest(text='test')))
    assert error.value.status_code == 404
    with q.connection() as connection:
        assert connection.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def enqueue(q,key='request-1',digest='input-1',session='session-1'):
    return q.enqueue('owner','test','cpu',{'command':[sys.executable,'-c','pass']},'test',
                     request_key=key,request_hash=digest,session_key=session)


def test_concurrent_submit_is_idempotent_and_conflicts_are_rejected(q):
    with ThreadPoolExecutor(8) as pool:results=list(pool.map(lambda _:enqueue(JobQueue(q.root)),range(16)))
    assert len({r[0] for r in results})==1
    assert sum(r[1] for r in results)==1
    with pytest.raises(SubmissionConflict):enqueue(q,digest='different-input')
    with pytest.raises(SubmissionConflict):enqueue(q,key='other-request')


def test_queued_commands_are_not_orphaned_or_bound_to_waiter_processes(q):
    identity,_=enqueue(q)
    with q.connection(True) as c:c.execute('UPDATE jobs SET created=created-900 WHERE id=?',(identity,))
    row=JobQueue(q.root).get(identity)
    assert row['status']=='queued' and row['pid'] is None
    assert len(list(q.root.glob('pools/*')))==0


def test_fencing_rejects_old_worker_events_heartbeats_and_completion(q):
    identity,_=enqueue(q);row=q.claim_next('test','cpu')
    assert row['id']==identity
    q.append_frame(identity,'event: token\ndata: {"text":"a"}\n\n',row['lease'])
    with q.connection(True) as c:c.execute('UPDATE jobs SET lease=? WHERE id=?',('new-lease',identity))
    with pytest.raises(LeaseLost):q.append_frame(identity,'data: stale\n\n',row['lease'])
    with pytest.raises(LeaseLost):q.heartbeat(identity,row['lease'])
    with pytest.raises(LeaseLost):q.finish(identity,'completed',lease=row['lease'])
    q.finish(identity,'interrupted',lease='new-lease')


def test_events_replay_from_cursor_and_enforce_owner(q):
    identity,_=enqueue(q);row=q.claim_next('test','cpu')
    start=q.append_frame(identity,'event: token\ndata: first\n\n',row['lease'])
    end=q.append_frame(identity,'event: token\ndata: second\n\n',row['lease'])
    q.finish(identity,'completed',lease=row['lease'])
    replay=JobQueue(q.root).events(identity,'owner',start)
    assert replay[0]['seq']==end and 'second' in replay[0]['frame']
    assert all(r['seq']>start for r in replay)
    with pytest.raises(FileNotFoundError):q.events(identity,'other')


def test_shared_lock_and_cancel_across_processes(q,tmp_path):
    async def scenario():
        lock=SharedSessionLock(q.root,'session')
        await lock.acquire()
        code='from backend.web_v2.session_runtime import SharedSessionLock;from backend.web_v2.job_queue import JobQueue;import sys;q=JobQueue();print(SharedSessionLock(q.root,"session").locked());q.set_cancel_flag("session",True)'
        output=subprocess.check_output([sys.executable,'-c',code],text=True)
        assert output.strip()=='True' and q.cancel_flag('session')
        lock.release();assert not SharedSessionLock(q.root,'session').locked()
    asyncio.run(scenario())


def test_signing_secret_survives_restart_and_concurrent_initialization(tmp_path):
    path=tmp_path/'signing-key'
    with ThreadPoolExecutor(6) as pool:values=list(pool.map(lambda _:persistent_secret(path),range(12)))
    assert len(set(values))==1
    child=subprocess.check_output([sys.executable,'-c',f'from backend.web_v2.session_runtime import persistent_secret;print(persistent_secret({str(path)!r}))'],text=True)
    assert child.strip()==values[0]
    assert path.stat().st_mode & 0o077 == 0


def test_session_cache_observes_another_process_store(tmp_path):
    from agent.session_store import SqliteBackend,SessionStore
    async def scenario():
        a=SessionStore(SqliteBackend(str(tmp_path/'sessions.sqlite3')),lambda x:x)
        b=SessionStore(SqliteBackend(str(tmp_path/'sessions.sqlite3')),lambda x:x)
        await a.create({'session_id':'s','owner_key':'a','history':[]})
        assert (await b.get('s'))['history']==[]
        state=await a.get('s');state['history']=[{'role':'user','content':'persisted'}];await a.save('s')
        assert (await b.get('s'))['history'][0]['content']=='persisted'
        assert (await b.list_summaries())[0]['history_size']==1
    asyncio.run(scenario())


def test_many_waiting_commands_share_one_fixed_pool(q):
    first,proc=launch([sys.executable,'-c','import time;time.sleep(20)'],owner='first',kind='test',queue=q)
    until=time.monotonic()+8
    while q.get(first)['status']!='running' and time.monotonic()<until:time.sleep(.03)
    ids=[launch([sys.executable,'-c','pass'],owner=str(i),kind='test',queue=q)[0] for i in range(12)]
    assert len(list(q.root.glob('pools/*/supervisor.json')))==1
    assert all(q.get(i)['pid'] is None and q.get(i)['status']=='queued' for i in ids)
    for identity in ids:q.cancel(identity)
    q.cancel(first);proc.wait(5)


def test_task_survives_submitter_exit_and_retry_replays_same_run(q,tmp_path):
    """A fresh submitter can recover an existing execution without running it twice."""
    gate=tmp_path/'continue'; output=tmp_path/'executions'
    command=tmp_path/'command.py'
    command.write_text('''import os,time
from pathlib import Path
from backend.web_v2.job_queue import JobQueue
q=JobQueue();i=os.environ['VENUS_QUEUE_JOB_ID'];lease=os.environ['VENUS_QUEUE_LEASE_TOKEN']
with Path(OUTPUT).open('a') as f:f.write('executed\\n')
q.append_frame(i,'event: token\\ndata: first\\n\\n',lease)
while not Path(GATE).exists():time.sleep(.05)
q.append_frame(i,'event: token\\ndata: second\\n\\n',lease)
'''.replace('OUTPUT',repr(str(output))).replace('GATE',repr(str(gate))))
    submit=tmp_path/'submit.py'
    submit.write_text('''import sys
from backend.web_v2.job_queue import launch
i,_=launch([sys.executable,COMMAND],owner='restart-test',kind='test',request_key='restart',request_hash='same')
print(i)
'''.replace('COMMAND',repr(str(command))))
    identity=subprocess.check_output([sys.executable,str(submit)],text=True).strip()
    until=time.monotonic()+10
    while not output.exists() and time.monotonic()<until:time.sleep(.05)
    assert output.read_text()=='executed\n'
    assert q.get(identity)['status']=='running'
    assert subprocess.check_output([sys.executable,str(submit)],text=True).strip()==identity
    gate.touch()
    until=time.monotonic()+10
    while q.get(identity)['status']=='running' and time.monotonic()<until:time.sleep(.05)
    assert q.get(identity)['status']=='completed'
    frames=q.events(identity,'restart-test')
    first=next(f['seq'] for f in frames if 'data: first' in f['frame'])
    replay=JobQueue(q.root).events(identity,'restart-test',first)
    assert any('data: second' in f['frame'] for f in replay)
    assert output.read_text()=='executed\n'
