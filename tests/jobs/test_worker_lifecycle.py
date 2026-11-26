import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from backend.web_v2 import worker_pool as pool
from backend.web_v2.job_queue import JobQueue, launch, process_token, stop_process_tree


@pytest.fixture
def queue(tmp_path,monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE','local');monkeypatch.setenv('VENUS_WORKER_AUTOSTART','1')
    monkeypatch.setenv('VENUS_WORKER_IDLE_TIMEOUT','1')
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    for family in ['CPU','GPU','API','RESEARCH']:monkeypatch.setenv('VENUS_WORKERS_'+family,'1')
    for key in ['VENUS_QUEUE_JOB_ID','VENUS_QUEUE_OWNER','VENUS_QUEUE_LEASE_TOKEN']:monkeypatch.delenv(key,raising=False)
    q=JobQueue()
    yield q
    for path in q.root.glob('pools/*/supervisor.json'):
        try:record=json.loads(path.read_text())
        except FileNotFoundError:continue
        stop_process_tree(record['pid'],record['token'])


def until(check,timeout=25):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=check()
        if value:return value
        time.sleep(.05)
    raise AssertionError('Worker lifecycle did not reach the expected state')


def test_idle_pool_exits_and_new_work_starts_a_fresh_pool(queue,tmp_path):
    code='from pathlib import Path; Path('+repr(str(tmp_path/'calls'))+').open("a").write("done\\n")'
    first,proc=launch([sys.executable,'-c',code],owner='owner',kind='test',queue=queue)
    assert proc.wait(25)==0
    namespace=queue.get(first)['namespace'];record=queue.root/'pools'/namespace/'supervisor.json'
    previous=json.loads(record.read_text())
    until(lambda:not record.exists())
    until(lambda:process_token(previous['pid'])!=previous['token'])
    second,proc=launch([sys.executable,'-c',code],owner='owner',kind='test',queue=queue)
    assert proc.wait(25)==0
    assert first!=second and queue.get(second)['namespace']==namespace
    assert(tmp_path/'calls').read_text().splitlines()==['done','done']


def test_active_work_is_not_retired_and_pending_work_prevents_retirement(queue):
    identity,proc=launch([sys.executable,'-c','import time;time.sleep(3)'],owner='owner',kind='test',queue=queue)
    until(lambda:queue.get(identity)['status']=='running')
    assert not pool.retire_idle_pool(queue,queue.get(identity)['namespace'],{})
    assert proc.wait(25)==0


def test_retirement_and_submission_cannot_orphan_a_job(queue,monkeypatch):
    from threading import Event
    ensure_pool=pool._ensure_pool_locked
    monkeypatch.setattr(pool,'_ensure_pool_locked',lambda *args:None)
    first,_=launch([sys.executable,'-c','pass'],owner='owner',kind='test',queue=queue)
    namespace=queue.get(first)['namespace'];queue.cancel(first)
    monkeypatch.setattr(pool,'_ensure_pool_locked',ensure_pool)
    entered=Event();release=Event();original=pool.stop_workers
    def stop(workers):
        entered.set();assert release.wait(5);original(workers)
    monkeypatch.setattr(pool,'stop_workers',stop)
    with ThreadPoolExecutor(2)as threads:
        retirement=threads.submit(pool.retire_idle_pool,queue,namespace,{})
        assert entered.wait(5)
        submission=threads.submit(launch,[sys.executable,'-c','pass'],owner='owner',kind='test',queue=queue)
        assert not submission.done()
        release.set();assert retirement.result()
        identity,proc=submission.result()
    assert queue.get(identity)['namespace']==namespace
    assert proc.wait(25)==0


@pytest.mark.parametrize('value',['0','-1','nan','inf','invalid'])
def test_invalid_idle_setting_keeps_bounded_retirement(value,monkeypatch):
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART','1')
    monkeypatch.setenv('VENUS_WORKER_IDLE_TIMEOUT',value)
    assert pool.idle_timeout()==60


def test_manually_managed_pool_remains_available(monkeypatch):
    monkeypatch.setenv('VENUS_WORKER_AUTOSTART','0')
    assert pool.idle_timeout()==0


def test_source_update_starts_a_different_pool_without_stopping_old_one(queue, monkeypatch):
    monkeypatch.setenv('VENUS_WORKER_IDLE_TIMEOUT', '30')
    version = ['first-source']
    monkeypatch.setattr(pool, 'loaded_source_revision', lambda: version[0])
    first, proc = launch([sys.executable, '-c', 'pass'], owner='owner', kind='test', queue=queue)
    assert proc.wait(25) == 0
    first_namespace = queue.get(first)['namespace']
    first_record = json.loads((queue.root/'pools'/first_namespace/'supervisor.json').read_text())
    version[0] = 'second-source'
    second, proc = launch([sys.executable, '-c', 'pass'], owner='owner', kind='test', queue=queue)
    assert proc.wait(25) == 0
    second_namespace = queue.get(second)['namespace']
    second_record = json.loads((queue.root/'pools'/second_namespace/'supervisor.json').read_text())
    assert first_namespace != second_namespace
    assert first_record['pid'] != second_record['pid']
    assert process_token(first_record['pid']) == first_record['token']
