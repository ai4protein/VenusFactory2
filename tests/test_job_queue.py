import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from backend.web_v2.job_queue import JobQueue, QueueCancelled, launch, process_token, resource_slot

@pytest.fixture
def q(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS','1')
    monkeypatch.setenv('VENUS_QUEUE_GPU_SLOTS','1')
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID',raising=False)
    return JobQueue()

def wait(q,jid,status,timeout=12):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        row=q.get(jid)
        if row['status'] in status:return row
        time.sleep(.04)
    raise AssertionError(q.get(jid))

def test_resources_are_independent_and_capacity_is_atomic(q):
    ids=[q.submit(str(i),'test','cpu') for i in range(8)]
    for i in ids:q.bind(i)
    with ThreadPoolExecutor(8) as pool:claims=list(pool.map(q.claim,ids))
    assert sum(claims)==1
    gpu=q.submit('another','test','gpu');q.bind(gpu);assert q.claim(gpu)
    assert q.get(ids[1])['position']==1
    q.finish(ids[0],'completed');assert q.claim(ids[1])

def test_priority_aging_and_owner_fairness(q,monkeypatch):
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS','2');monkeypatch.setenv('VENUS_QUEUE_RUNNING_PER_OWNER','1')
    a=q.submit('one','train');b=q.submit('one','train');c=q.submit('two','predict',priority=10)
    for i in [a,b,c]:q.bind(i)
    assert q.claim(c);assert q.claim(a);assert not q.claim(b)
    q.finish(a,'completed');q.finish(c,'completed')
    with q.connection(True) as db:db.execute('UPDATE jobs SET created=created-1500 WHERE id=?',(b,))
    d=q.submit('three','predict',priority=10);q.bind(d)
    assert not q.claim(d);assert q.claim(b)

def test_ownership_and_queued_cancellation(q):
    a=q.submit('one','train');q.bind(a)
    with pytest.raises(FileNotFoundError):q.cancel(a,'other')
    with pytest.raises(FileNotFoundError):q.public(a,'other')
    assert q.cancel(a,'one')['status']=='cancelled'
    with pytest.raises(QueueCancelled):q.claim(a)
    assert not q.list('other')

def test_waiting_job_starts_automatically_and_wait_does_not_consume_runtime(q,tmp_path):
    first,p1=launch([sys.executable,'-c','import time;time.sleep(1)'],owner='a',kind='test',resource='gpu',queue=q)
    wait(q,first,{'running'})
    output=tmp_path/'ran'
    second,p2=launch([sys.executable,'-c',f'from pathlib import Path;Path({str(output)!r}).write_text("done")'],owner='b',kind='test',resource='gpu',timeout=.5,queue=q)
    assert q.public(second)['status']=='queued'
    assert wait(q,second,{'completed','failed'})['status']=='completed'
    assert output.read_text()=='done';p1.wait(3);p2.wait(3)

def test_cancel_waiting_never_starts_and_running_releases_slot(q,tmp_path):
    a,p1=launch([sys.executable,'-c','import time;time.sleep(30)'],owner='a',kind='test',queue=q)
    wait(q,a,{'running'})
    output=tmp_path/'must-not-exist'
    b,p2=launch([sys.executable,'-c',f'from pathlib import Path;Path({str(output)!r}).touch()'],owner='b',kind='test',queue=q)
    q.cancel(b);p2.wait(5);assert not output.exists()
    q.cancel(a);assert wait(q,a,{'cancelled'})['status']=='cancelled';p1.wait(5)
    c,p3=launch([sys.executable,'-c','pass'],owner='c',kind='test',queue=q)
    assert wait(q,c,{'completed'})['status']=='completed';p3.wait(3)

def test_worker_crash_recovery_stops_live_child_before_freeing_slot(q):
    a,p=launch([sys.executable,'-c','import time;time.sleep(30)'],owner='a',kind='test',queue=q)
    wait(q,a,{'running'})
    for _ in range(100):
        child=q.get(a)['metadata'].get('_child')
        if child:break
        time.sleep(.02)
    assert child and process_token(child['pid'])==child['token']
    p.kill();p.wait(3)
    assert JobQueue(q.root).get(a)['status']=='interrupted'
    for _ in range(30):
        if process_token(child['pid'])!=child['token']:break
        time.sleep(.02)
    assert process_token(child['pid'])!=child['token']

def test_runtime_timeout_and_queue_persist_across_controller_instances(q):
    a,p=launch([sys.executable,'-c','import time;time.sleep(30)'],owner='a',kind='test',timeout=.25,queue=q)
    assert wait(JobQueue(q.root),a,{'failed'})['message']=='Execution time limit reached';p.wait(3)

def test_async_slots_release_on_cancellation(q):
    async def scenario():
        entered=asyncio.Event()
        async def holder():
            async with resource_slot('api:test.local'):
                entered.set();await asyncio.sleep(30)
        task=asyncio.create_task(holder());await entered.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert not [j for j in q.list(f'process:{os.getpid()}') if j['status']=='running']
    asyncio.run(scenario())

def test_credentials_not_written_to_queue(q):
    env=dict(os.environ,PRIVATE_TEST_CREDENTIAL='test-never-persist-this-value')
    a,p=launch([sys.executable,'-c','pass'],owner='a',kind='test',env=env,queue=q)
    wait(q,a,{'completed'});p.wait(3)
    for path in q.root.rglob('*'):
        if path.is_file():assert b'test-never-persist-this-value' not in path.read_bytes()

def test_detached_job_survives_the_submitting_process_exit(q,tmp_path):
    output=tmp_path/'output';record=tmp_path/'ticket'
    child_code=f'import time;from pathlib import Path;time.sleep(.5);Path({str(output)!r}).write_text("finished")'
    controller='from backend.web_v2.job_queue import launch;from pathlib import Path;import sys\n'+f'job,p=launch([sys.executable,"-c",{child_code!r}],owner="controller",kind="test")\nPath({str(record)!r}).write_text(job)'
    subprocess.run([sys.executable,'-c',controller],check=True,timeout=5)
    identity=record.read_text()
    assert wait(JobQueue(q.root),identity,{'completed','failed','interrupted'})['status']=='completed'
    assert output.read_text()=='finished'
