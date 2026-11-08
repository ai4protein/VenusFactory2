import hashlib
import json
from pathlib import Path
import sys
import time
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from backend.web_v2.job_queue import JobQueue, launch

def terminal(q,identity):
    for _ in range(200):
        value=q.get(identity)
        if value['status'] in {'completed','cancelled','failed','interrupted'}:return value
        time.sleep(.025)
    raise AssertionError(q.public(identity))

def test_model_platform_queues_instead_of_rejecting_and_cancel_is_owner_scoped(tmp_path,monkeypatch):
    from backend.web_v2.model_platform import Platform
    from backend.web_v2 import model_platform_api as api
    monkeypatch.setenv('VENUS_QUEUE_CPU_SLOTS','1')
    p=Platform(tmp_path/'platform');monkeypatch.setattr(api,'service',p)
    holder,worker=launch([sys.executable,'-c','import time;time.sleep(30)'],owner='holder',kind='test',queue=p.queue)
    for _ in range(100):
        if p.queue.get(holder)['status']=='running':break
        time.sleep(.02)
    app=FastAPI();app.include_router(api.router)
    try:
        with TestClient(app) as a,TestClient(app) as b:
            a.get('/api/model-platform/session');b.get('/api/model-platform/session')
            data=a.post('/api/model-platform/datasets',files={'file':('protein.fa',b'>a\nACDEFGHIK')}).json()
            model=next(m for m in p.official() if m['adapter']=='descriptors')
            jobs=[a.post('/api/model-platform/predict',json={'dataset_id':data['id'],'model_id':model['id']}) for _ in range(5)]
            assert all(j.status_code==200 for j in jobs)
            first=jobs[0].json();assert first['queue']['status']=='queued'
            assert b.post('/api/model-platform/jobs/'+first['id']+'/cancel').status_code==404
            assert a.post('/api/model-platform/jobs/'+first['id']+'/cancel').json()['status']=='cancelled'
            assert b.get('/api/model-platform/jobs').json()==[]
            for j in jobs[1:]:a.post('/api/model-platform/jobs/'+j.json()['id']+'/cancel')
            assert len(a.get('/api/model-platform/jobs').json())==5
    finally:p.queue.cancel(holder);worker.wait(5)

def test_custom_jobs_persist_stream_and_reject_cross_owner_control(tmp_path,monkeypatch):
    from backend.web_v2 import custom_model_api as api
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setattr(api,'_require_training_model',lambda args:None)
    monkeypatch.setattr(api,'_normalize_training_args',lambda args,constant:args)
    monkeypatch.setattr(api,'_safe_read_json',lambda *_:{})
    monkeypatch.setattr(api,'_build_exec_cmd',lambda *_:[sys.executable,'-u','-c','print("Epoch 0 Train Loss: 0.5",flush=True)'])
    app=FastAPI();app.include_router(api.router)
    with TestClient(app) as a,TestClient(app) as b:
        a.get('/api/custom-model/jobs');b.get('/api/custom-model/jobs')
        response=a.post('/api/custom-model/training/start',json={'args':{'output_dir':str(tmp_path),'num_epochs':1}})
        assert response.status_code==200,response.text
        assert 'event: queue' in response.text and 'event: done' in response.text
        jobs=a.get('/api/custom-model/jobs?task=training').json();assert len(jobs)==1
        identity=jobs[0]['id'];assert jobs[0]['status']=='completed'
        assert b.get('/api/custom-model/jobs').json()==[]
        assert b.post('/api/custom-model/jobs/'+identity+'/cancel').status_code==404
        assert b.post('/api/custom-model/jobs/'+identity+'/stream').status_code==404
        replay=a.post('/api/custom-model/jobs/'+identity+'/stream').text
        logs=[frame for frame in replay.split('\n\n') if frame.startswith('event: log\n')]
        assert sum('Epoch 0 Train Loss: 0.5' in frame for frame in logs)==1
        assert a.post('/api/custom-model/jobs/'+identity+'/cancel',headers={'Origin':'https://unrelated.example'}).status_code==403
        assert a.post('/api/custom-model/jobs/'+identity+'/retry').status_code==409
        first_output=JobQueue().get(identity)['metadata']['args']['output_dir']
        a.post('/api/custom-model/training/start',json={'args':{'output_dir':str(tmp_path),'num_epochs':1}})
        second=a.get('/api/custom-model/jobs?task=training').json()[0]
        assert JobQueue().get(second['id'])['metadata']['args']['output_dir']!=first_output

def test_scientist_uses_research_queue_and_retains_lease_while_waiting(tmp_path,monkeypatch):
    import fcntl
    from backend.web_v2.science_goals import ScienceGoalService
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('VENUS_QUEUE_RESEARCH_SLOTS','1')
    q=JobQueue();holder,worker=launch([sys.executable,'-c','import time;time.sleep(30)'],owner='other',kind='test',resource='research',queue=q)
    for _ in range(100):
        if q.get(holder)['status']=='running':break
        time.sleep(.02)
    class Driver:
        def run_command(self,*_):return [sys.executable,'-c','pass']
    class Service:
        root=tmp_path
        children={}
        def live(self,folder):return False
        def frozen_profile(self,folder):return __import__('types').SimpleNamespace(harness='expert-kimi-code')
        def environment(self,profile):return dict(__import__('os').environ)
    folder=tmp_path/'session'/'goal';folder.mkdir(parents=True)
    (folder.parent/'goal.json').write_text(json.dumps({'online_owner': 'test-research-owner', 'session_id': 'research-conversation'}))
    try:
        service=Service();ScienceGoalService.spawn(service,folder,Driver(),'run')
        ticket=json.loads((folder/'queue.json').read_text())['id']
        assert q.get(ticket)['resource']=='research'
        assert q.get(ticket)['owner']=='test-research-owner'
        assert q.get(ticket)['status']=='queued'
        # A queued task has a durable reservation, not a waiting process/FD.
        with (folder/'.worker.lock').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert q.get(ticket)['pid'] is None
        assert q.get(ticket)['session_key']=='chat:research-conversation'
        assert q.get(ticket)['metadata']['session_id']=='research-conversation'
        q.cancel(ticket);terminal(q,ticket);service.children[str(folder)].wait(5)
    finally:q.cancel(holder);worker.wait(5)
