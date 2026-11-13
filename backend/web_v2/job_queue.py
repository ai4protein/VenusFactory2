"""Process-safe, persistent admission queue shared by the research application.

Fixed workers claim durable commands only when a resource slot is available.
Only metadata and non-secret command specifications are persisted. Credentials
remain in the worker environment. Lost executions require explicit resubmission.
"""
from __future__ import annotations
import asyncio
from contextvars import ContextVar

_held_slot = ContextVar("venus_held_resource_slot", default=None)
from contextlib import contextmanager, asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import signal
import sys
import time
import uuid

PROJECT = Path(__file__).resolve().parents[2]
from .task_store import DurableJobsMixin, LeaseLost
TERMINAL = {'completed', 'failed', 'cancelled', 'interrupted', 'paused', 'waiting_for_input'}


def process_token(pid):
    try:
        stat = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z': return None
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip() + ':' + stat[19]
    except (OSError, ValueError, IndexError): return None


class QueueCancelled(RuntimeError): pass


def stop_process_tree(pid, token):
    """Signal only processes whose recorded Linux start identities still match."""
    if not token or process_token(pid) != token: return
    descendants={int(pid):token}
    # Include descendants that made their own session/process group.
    parents={}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            stat=(path/'stat').read_text().rsplit(')',1)[1].split()
            parents[int(path.name)]=int(stat[1])
        except (OSError,ValueError,IndexError):pass
    for _ in range(32):
        found={p:process_token(p) for p,parent in parents.items() if parent in descendants and p not in descendants}
        if not found:break
        descendants.update({p:t for p,t in found.items() if t})
    for sig in [signal.SIGTERM,signal.SIGKILL]:
        for p,t in reversed(list(descendants.items())):
            if process_token(p)==t:
                try:os.kill(p,sig)
                except ProcessLookupError:pass
        if sig==signal.SIGTERM:time.sleep(.3)


class JobQueue(DurableJobsMixin):
    def __init__(self, root=None):
        default = Path.home()/'.local/share/venusfactory/job-queue'/hashlib.sha256(str(PROJECT).encode()).hexdigest()[:16]
        self.root = Path(root or os.getenv('VENUS_JOB_QUEUE_ROOT') or default).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = self.root/'jobs.sqlite3'
        with self.connection() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('BEGIN IMMEDIATE')
            c.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, kind TEXT NOT NULL, resource TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, started REAL, finished REAL,
                priority INTEGER NOT NULL, pid INTEGER, token TEXT, cancel_requested INTEGER DEFAULT 0,
                message TEXT DEFAULT '', exit_code INTEGER, metadata TEXT NOT NULL)''')
            c.execute('CREATE INDEX IF NOT EXISTS scheduling ON jobs(resource,status,created)')
            self.initialize_tasks(c)
            c.commit()
        self.db.chmod(0o600)

    @contextmanager
    def connection(self, write=False):
        c = sqlite3.connect(self.db, timeout=15, isolation_level=None)
        c.row_factory = sqlite3.Row
        try:
            c.execute('PRAGMA busy_timeout=15000')
            if write: c.execute('BEGIN IMMEDIATE')
            yield c
            if write: c.commit()
        except BaseException:
            if write: c.rollback()
            raise
        finally: c.close()

    @staticmethod
    def capacity(resource):
        family = resource.split(':')[0].upper()
        default = {'CPU': 2, 'GPU': 1, 'RESEARCH': 2, 'API': 2}[family]
        return max(1, int(os.getenv('VENUS_QUEUE_' + family + '_SLOTS', str(default))))

    def _recover(self, c):
        for r in c.execute("SELECT * FROM jobs WHERE status IN ('queued','running')").fetchall():
            managed = bool(r['dispatch'])
            if managed and r['status']=='queued':
                parent=json.loads(r['metadata']).get('parent_job')
                if parent:
                    p=c.execute('SELECT status,cancel_requested FROM jobs WHERE id=?',(parent,)).fetchone()
                    if not p or p['status'] in TERMINAL or p['cancel_requested']:
                        c.execute("UPDATE jobs SET status='cancelled',finished=?,cancel_requested=1,message='Parent stopped' WHERE id=?",(time.time(),r['id']))
                        self._event(c,r['id'],'run',{'run_id':r['id'],'status':'cancelled'})
                continue
            missing=(r['pid'] and process_token(r['pid']) != r['token']) or (not managed and not r['pid'] and time.time()-r['created'] > 120)
            expired=managed and r['lease_until'] and r['lease_until'] < time.time()
            if missing or expired:
                child=json.loads(r['metadata']).get('_child')
                if child:stop_process_tree(child['pid'],child['token'])
                if expired:stop_process_tree(r['pid'],r['token'])
                status='cancelled' if r['cancel_requested'] else 'interrupted'
                message='Cancelled' if status=='cancelled' else 'Worker stopped; saved progress is available for review and resume'
                c.execute("UPDATE jobs SET status=?,finished=?,message=?,lease=NULL,lease_until=NULL WHERE id=?",(status,time.time(),message,r['id']))
                self._event(c,r['id'],'run',{'run_id':r['id'],'status':status,'message':message})

    def child(self, identity, pid):
        with self.connection(True) as c:
            r=c.execute('SELECT metadata FROM jobs WHERE id=?',(identity,)).fetchone()
            metadata=json.loads(r['metadata']);metadata['_child']={'pid':pid,'token':process_token(pid)}
            c.execute('UPDATE jobs SET metadata=? WHERE id=?',(json.dumps(metadata),identity))

    def submit(self, owner, kind, resource='cpu', *, priority=0, metadata=None):
        from backend.compute.access import assert_resource_admitted
        assert_resource_admitted(resource)
        if not re.fullmatch(r'(cpu|gpu|research|api:[a-z0-9._-]+)', resource): raise ValueError('Invalid queue resource')
        identity = uuid.uuid4().hex
        with self.connection(True) as c:
            self._recover(c)
            count = c.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count >= max(1, int(os.getenv('VENUS_QUEUE_MAX_PENDING', '256'))): raise ValueError('Job queue is full. Try again later.')
            mine = c.execute("SELECT COUNT(*) FROM jobs WHERE owner=? AND status IN ('queued','running')", (owner,)).fetchone()[0]
            if mine >= max(1, int(os.getenv('VENUS_QUEUE_MAX_PENDING_PER_OWNER','32'))): raise ValueError('Your job queue is full. Wait for a task to finish.')
            c.execute('INSERT INTO jobs(id,owner,kind,resource,status,created,priority,metadata) VALUES(?,?,?,?,?,?,?,?)', (identity, owner, kind, resource, 'queued', time.time(), max(0,min(20,priority)), json.dumps(metadata or {})))
        return identity

    def bind(self, identity, pid=None):
        pid = pid or os.getpid(); token = process_token(pid)
        if not token: raise ValueError('Worker identity unavailable')
        with self.connection(True) as c:
            r = c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            if not r: raise FileNotFoundError(identity)
            if r['pid'] and (r['pid'] != pid or r['token'] != token): raise ValueError('Job already has a worker')
            if r['status'] not in TERMINAL: c.execute('UPDATE jobs SET pid=?,token=? WHERE id=?',(pid,token,identity))

    def _eligible(self, c, resource):
        from backend.compute.access import assert_resource_admitted
        assert_resource_admitted(resource)
        active = c.execute("SELECT owner FROM jobs WHERE resource=? AND status='running'",(resource,)).fetchall()
        if len(active) >= self.capacity(resource): return None
        per_owner = max(1,int(os.getenv('VENUS_QUEUE_RUNNING_PER_OWNER','2')))
        # Priority aging: short predictions go first, older batch work catches up.
        waiting = c.execute("SELECT * FROM jobs WHERE resource=? AND status='queued' AND cancel_requested=0 ORDER BY (priority + (? - created)/60.0) DESC,created,id",(resource,time.time())).fetchall()
        return next((r['id'] for r in waiting if sum(x['owner']==r['owner'] for x in active)<per_owner), None)

    def claim(self, identity):
        with self.connection(True) as c:
            self._recover(c)
            r = c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            if not r: raise FileNotFoundError(identity)
            if r['cancel_requested'] or r['status'] in TERMINAL: raise QueueCancelled(r['status'])
            from backend.compute.access import assert_resource_admitted
            assert_resource_admitted(r['resource'])
            if r['pid'] != os.getpid() or r['token'] != process_token(os.getpid()): raise ValueError('Only the bound worker may acquire a slot')
            if r['status']=='running': return True
            if self._eligible(c,r['resource']) != identity: return False
            c.execute("UPDATE jobs SET status='running',started=?,message='Running' WHERE id=?",(time.time(),identity))
            return True

    def get(self, identity, owner=None):
        with self.connection(True) as c:
            self._recover(c)
            r = c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            if not r or (owner is not None and r['owner']!=owner): raise FileNotFoundError(identity)
            result = dict(r)
            result['metadata'] = json.loads(result['metadata'])
            result['position'] = 0
            if r['status']=='queued':
                waiting = c.execute("SELECT id FROM jobs WHERE resource=? AND status='queued' ORDER BY (priority + (? - created)/60.0) DESC,created,id",(r['resource'],time.time())).fetchall()
                result['position'] = next(i+1 for i,row in enumerate(waiting) if row['id']==identity)
                result['message'] = 'Waiting for ' + r['resource'].split(':')[0].upper()
            return result

    def public(self, identity, owner=None):
        r=self.get(identity,owner)
        r['cancel_requested']=bool(r['cancel_requested'])
        return {k:r[k] for k in ['id','kind','resource','status','created','started','finished','position','message','cancel_requested','exit_code','heartbeat','attempt']}

    def list(self, owner, kind=None):
        with self.connection() as c:
            ids=[r[0] for r in c.execute('SELECT id FROM jobs WHERE owner=? AND (? IS NULL OR kind=?) ORDER BY created DESC LIMIT 100',(owner,kind,kind))]
        return [self.public(i,owner) for i in ids]

    def cancel(self, identity, owner=None):
        with self.connection(True) as c:
            r=c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            if not r or (owner is not None and r['owner']!=owner): raise FileNotFoundError(identity)
            if r['status'] not in TERMINAL:
                if r['status']=='queued': c.execute("UPDATE jobs SET status='cancelled',finished=?,cancel_requested=1,message='Cancelled' WHERE id=?",(time.time(),identity))
                else: c.execute("UPDATE jobs SET cancel_requested=1,message='Stopping' WHERE id=?",(identity,))
        return self.public(identity,owner)

    def finish(self, identity, status, message='', exit_code=None, *, lease=None):
        if status not in TERMINAL: raise ValueError('Invalid terminal state')
        with self.connection(True) as c:
            row=c.execute('SELECT * FROM jobs WHERE id=?',(identity,)).fetchone()
            if not row:raise FileNotFoundError(identity)
            if lease and (row['lease']!=lease or row['status']!='running'):raise LeaseLost('Task lease is no longer current')
            if row['status'] not in {'queued','running'}:return
            if row['dispatch'] and row['status']=='running' and not lease:raise LeaseLost('A running task requires its current lease')
            status='cancelled' if row['cancel_requested'] else status
            c.execute("UPDATE jobs SET status=?,finished=?,message=?,exit_code=?,lease=NULL,lease_until=NULL WHERE id=?",(status,time.time(),message or status.capitalize(),exit_code,identity))
            self._event(c,identity,'run',{'run_id':identity,'status':status,'message':message or status.capitalize()})

    def files(self, identity):
        if not re.fullmatch('[a-f0-9]{32}',identity): raise ValueError('Invalid job ID')
        p=self.root/'work'/identity;p.mkdir(parents=True,exist_ok=True,mode=0o700);return p


def launch(command, *, owner, kind, resource='cpu', env=None, cwd=PROJECT, stdout=None, pass_fds=(), timeout=3600, priority=0, metadata=None, queue=None,
           session_key=None, request_key=None, request_hash=None):
    from .worker_pool import submit_command
    return submit_command(command, owner=owner, kind=kind, resource=resource, env=env, cwd=cwd,
                          stdout=stdout, pass_fds=pass_fds, timeout=timeout, priority=priority,
                          metadata=metadata, queue=queue, session_key=session_key,
                          request_key=request_key, request_hash=request_hash)


@asynccontextmanager
async def resource_slot(resource, *, owner=None, kind='compute', on_wait=None):
    from backend.compute.access import assert_resource_admitted
    assert_resource_admitted(resource)
    q=JobQueue()
    inherited=_held_slot.get() or os.getenv('VENUS_QUEUE_JOB_ID')
    if inherited:
        try:
            held=q.get(inherited)
            if held['status']=='running' and held['resource']==resource:
                yield held['id'];return
        except FileNotFoundError:pass
    identity=q.submit(owner or os.getenv('VENUS_QUEUE_OWNER') or f'process:{os.getpid()}',kind,resource)
    q.bind(identity)
    try:
        while not q.claim(identity):
            if on_wait:on_wait(q.public(identity))
            await asyncio.sleep(.2)
        token = _held_slot.set(identity)
        try:
            yield identity
        finally:
            _held_slot.reset(token)
    except asyncio.CancelledError:
        q.cancel(identity);q.finish(identity,'cancelled');raise
    except BaseException:
        q.finish(identity,'failed');raise
    else:q.finish(identity,'completed')


def run_in_slot(resource, callback, *args, **kwargs):
    """For synchronous worker code; nested async adapters share the claimed slot."""
    async def run():
        async with resource_slot(resource):
            return await asyncio.to_thread(callback, *args, **kwargs)
    return asyncio.run(run())
