"""Durable dispatch, fenced leases and replayable events for the shared queue."""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid

ACTIVE = ('queued', 'running')
DONE = {'completed', 'failed', 'cancelled', 'interrupted', 'paused', 'waiting_for_input'}

class SubmissionConflict(ValueError): pass
class LeaseLost(RuntimeError): pass


class DurableJobsMixin:
    @staticmethod
    def initialize_tasks(c):
        columns={r[1] for r in c.execute('PRAGMA table_info(jobs)')}
        for name,definition in {
            'dispatch':'TEXT', 'namespace':'TEXT', 'lease':'TEXT', 'heartbeat':'REAL',
            'lease_until':'REAL', 'attempt':'INTEGER NOT NULL DEFAULT 0',
            'session_key':'TEXT', 'request_key':'TEXT', 'request_hash':'TEXT',
        }.items():
            if name not in columns:c.execute(f'ALTER TABLE jobs ADD COLUMN {name} {definition}')
        c.execute('CREATE UNIQUE INDEX IF NOT EXISTS task_request ON jobs(owner,request_key) WHERE request_key IS NOT NULL')
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS task_session_active ON jobs(session_key) WHERE session_key IS NOT NULL AND status IN ('queued','running')")
        c.execute('CREATE INDEX IF NOT EXISTS task_dispatch ON jobs(namespace,status,resource)')
        c.execute('''CREATE TABLE IF NOT EXISTS task_events (
            run_id TEXT NOT NULL, seq INTEGER NOT NULL, created REAL NOT NULL,
            frame TEXT NOT NULL, PRIMARY KEY(run_id,seq))''')
        c.execute('''CREATE TABLE IF NOT EXISTS session_controls (
            session_id TEXT PRIMARY KEY, cancelled INTEGER NOT NULL DEFAULT 0)''')

    def request(self, owner, key, payload_hash=None):
        if not key:return None
        with self.connection() as c:
            r=c.execute('SELECT * FROM jobs WHERE owner=? AND request_key=?',(owner,key)).fetchone()
        if r and payload_hash and r['request_hash']!=payload_hash:
            raise SubmissionConflict('Request identity belongs to different input')
        return dict(r) if r else None

    def enqueue(self, owner, kind, resource, dispatch, namespace, *, metadata=None,
                priority=0, session_key=None, request_key=None, request_hash=None):
        import re

        from backend.compute.access import assert_resource_admitted
        assert_resource_admitted(resource)
        if not re.fullmatch(r'(cpu|gpu|research|api:[a-z0-9._-]+)',resource):raise ValueError('Invalid queue resource')
        identity=uuid.uuid4().hex
        with self.connection(True) as c:
            self._recover(c)
            if request_key:
                previous=c.execute('SELECT * FROM jobs WHERE owner=? AND request_key=?',(owner,request_key)).fetchone()
                if previous:
                    if previous['request_hash']!=request_hash:raise SubmissionConflict('Request identity belongs to different input')
                    return previous['id'],False
            if session_key and c.execute("SELECT 1 FROM jobs WHERE session_key=? AND status IN ('queued','running')",(session_key,)).fetchone():
                raise SubmissionConflict('This conversation already has an active run')
            total=c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            mine=c.execute("SELECT COUNT(*) FROM jobs WHERE owner=? AND status IN ('queued','running')",(owner,)).fetchone()[0]
            if total>=max(1,int(os.getenv('VENUS_QUEUE_MAX_PENDING','256'))):raise ValueError('Job queue is full. Try again later.')
            if mine>=max(1,int(os.getenv('VENUS_QUEUE_MAX_PENDING_PER_OWNER','32'))):raise ValueError('Your job queue is full. Wait for a task to finish.')
            c.execute('''INSERT INTO jobs(id,owner,kind,resource,status,created,priority,metadata,dispatch,namespace,session_key,request_key,request_hash)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(identity,owner,kind,resource,'queued',time.time(),max(0,min(20,priority)),json.dumps(metadata or {}),json.dumps(dispatch),namespace,session_key,request_key,request_hash))
            self._event(c,identity,'run',{'run_id':identity,'status':'queued'})
        return identity,True

    def claim_next(self, namespace, family):
        from .job_queue import process_token
        pid=os.getpid();token=process_token(pid)
        if not token:raise RuntimeError('Worker identity unavailable')
        with self.connection(True) as c:
            self._recover(c)
            resources=[r[0] for r in c.execute("SELECT DISTINCT resource FROM jobs WHERE namespace=? AND status='queued' AND dispatch IS NOT NULL",(namespace,)) if r[0].split(':')[0]==family]
            candidates=[]
            for resource in resources:
                identity=self._eligible(c,resource)
                if identity:
                    row=c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
                    if row['namespace']==namespace and row['dispatch']:candidates.append(row)
            if not candidates:return None
            r=max(candidates,key=lambda row:row['priority']+(time.time()-row['created'])/60)
            lease=uuid.uuid4().hex;now=time.time()
            c.execute("UPDATE jobs SET status='running',started=?,pid=?,token=?,lease=?,heartbeat=?,lease_until=?,attempt=attempt+1,message='Running' WHERE id=?",
                      (now,pid,token,lease,now,now+self.lease_seconds(),r['id']))
            self._event(c,r['id'],'run',{'run_id':r['id'],'status':'running','attempt':r['attempt']+1})
            row=dict(c.execute('SELECT * FROM jobs WHERE id=?',(r['id'],)).fetchone())
        row['metadata']=json.loads(row['metadata']);row['dispatch']=json.loads(row['dispatch'])
        return row

    @staticmethod
    def lease_seconds():return max(5,float(os.getenv('VENUS_QUEUE_LEASE_SECONDS','30')))

    def heartbeat(self, identity, lease):
        with self.connection(True) as c:
            now=time.time()
            updated=c.execute("UPDATE jobs SET heartbeat=?,lease_until=? WHERE id=? AND lease=? AND status='running'",(now,now+self.lease_seconds(),identity,lease)).rowcount
            if not updated:raise LeaseLost('Task lease is no longer current')

    def attach_child(self, identity, lease, pid):
        from .job_queue import process_token
        with self.connection(True) as c:
            r=c.execute("SELECT * FROM jobs WHERE id=? AND lease=? AND status='running'",(identity,lease)).fetchone()
            if not r or r['cancel_requested'] or process_token(r['pid'])!=r['token']:raise LeaseLost('Task stopped before execution')
            data=json.loads(r['metadata']);data['_child']={'pid':pid,'token':process_token(pid)}
            c.execute('UPDATE jobs SET metadata=? WHERE id=? AND lease=?',(json.dumps(data),identity,lease))

    @staticmethod
    def _event(c, identity, event, data):
        frame='event: '+event+'\ndata: '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n\n'
        return DurableJobsMixin._frame(c,identity,frame)

    @staticmethod
    def _frame(c, identity, frame):
        seq=c.execute('SELECT COALESCE(MAX(seq),0)+1 FROM task_events WHERE run_id=?',(identity,)).fetchone()[0]
        c.execute('INSERT INTO task_events VALUES(?,?,?,?)',(identity,seq,time.time(),frame))
        return seq

    def append_frame(self, identity, frame, lease=None):
        if not isinstance(frame,str) or len(frame.encode())>2_000_000:raise ValueError('Invalid task event')
        with self.connection(True) as c:
            row=c.execute('SELECT status,lease FROM jobs WHERE id=?',(identity,)).fetchone()
            if not row:raise FileNotFoundError(identity)
            if lease and (row['lease']!=lease or row['status']!='running'):raise LeaseLost('Task lease is no longer current')
            return self._frame(c,identity,frame)

    def events(self, identity, owner=None, after=0, limit=128):
        with self.connection() as c:
            row=c.execute('SELECT owner FROM jobs WHERE id=?',(identity,)).fetchone()
            if not row or owner is not None and row['owner']!=owner:raise FileNotFoundError(identity)
            return [dict(r) for r in c.execute('SELECT seq,created,frame FROM task_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?',(identity,max(0,int(after)),min(256,max(1,int(limit)))))]

    def latest_run(self, session_key, owner=None):
        with self.connection(True) as c:
            self._recover(c)
            row=c.execute('SELECT id,owner FROM jobs WHERE session_key=? ORDER BY created DESC,id DESC LIMIT 1',(session_key,)).fetchone()
        if not row or owner is not None and row['owner']!=owner:return None
        return self.public(row['id'],owner)

    def set_cancel_flag(self, session_id, value):
        with self.connection(True) as c:
            c.execute('INSERT INTO session_controls VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET cancelled=excluded.cancelled',(session_id,int(value)))

    def cancel_flag(self, session_id):
        with self.connection() as c:
            row=c.execute('SELECT cancelled FROM session_controls WHERE session_id=?',(session_id,)).fetchone()
        return bool(row and row[0])

    def metrics(self):
        with self.connection(True) as c:
            self._recover(c)
            counts=[dict(r) for r in c.execute('SELECT resource,status,COUNT(*) AS count FROM jobs GROUP BY resource,status')]
            workers=[dict(r) for r in c.execute("SELECT id,kind,resource,started,heartbeat,lease_until,attempt,cancel_requested FROM jobs WHERE status='running'")]
        return {'counts':counts,'active':workers,'at':time.time()}
