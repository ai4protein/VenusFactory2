"""Visitor-scoped online compute admission shared by web and MCP workers."""
from contextvars import ContextVar
import hashlib
import math
import os
import time
import uuid
import functools
import inspect

from fastapi import HTTPException
from backend.web_v2.online_access import AccessStore, online

actor_context = ContextVar('compute_actor', default=None)
request_key_context = ContextVar('compute_request_key', default=None)
inline_request_context = ContextVar('inline_compute_request', default=None)
LIMITS = {'workflow': ('WEBUI_V2_ONLINE_DAILY_WORKFLOW_LIMIT', 5),
          'prediction': ('WEBUI_V2_ONLINE_DAILY_PREDICTION_LIMIT', 20)}


def positive_setting(key, default):
    try:
        value = int(os.getenv(key, '').strip() or default)
        return value if value > 0 else default
    except ValueError:
        return default


class ComputeStore(AccessStore):
    def __init__(self, path=None):
        super().__init__(path)
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS compute_owners(owner TEXT PRIMARY KEY, visitor TEXT NOT NULL, ip TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS compute_admissions(scope TEXT NOT NULL, resource TEXT NOT NULL, day TEXT NOT NULL,
              visitor TEXT NOT NULL, ip TEXT NOT NULL, units INTEGER NOT NULL, created REAL NOT NULL,
              PRIMARY KEY(scope,resource));
            CREATE INDEX IF NOT EXISTS compute_daily ON compute_admissions(visitor,day,resource);
            CREATE INDEX IF NOT EXISTS compute_network ON compute_admissions(ip,created);
            ''')

    def bind(self, owner, actor):
        if not actor:
            return
        ip, visitor = actor
        with self.connect(True) as c:
            # Clearing only the quota cookie must not reset an existing workspace.
            c.execute('INSERT OR IGNORE INTO compute_owners VALUES(?,?,?)', (owner, visitor, ip))
            c.execute('UPDATE compute_owners SET ip=? WHERE owner=?', (ip, owner))

    @staticmethod
    def limit(resource):
        key, default = LIMITS[resource]
        return positive_setting(key, default)

    def _status(self, c, visitor, day, reset):
        from backend.web_v2.quota_feedback import since, bound_grant
        cutoff = since(c)
        grant = bound_grant(c, visitor)
        values = {}
        for resource in LIMITS:
            used = c.execute('SELECT COALESCE(SUM(units),0) FROM compute_admissions WHERE visitor=? AND day=? AND resource=? AND created>?',
                             (visitor, day, resource, cutoff)).fetchone()[0]
            limit = self.limit(resource)
            if resource == 'prediction' and grant:
                limit += grant['extra_predictions']
                shared = c.execute("""SELECT COALESCE(SUM(a.units),0) FROM compute_admissions a
                    JOIN compute_grants g ON g.scope=a.scope AND g.resource=a.resource
                    WHERE g.grant_id=? AND a.day=? AND a.resource=? AND a.created>?""",
                    (grant['id'],day,resource,cutoff)).fetchone()[0]
                used = max(used, shared)
            values[resource] = dict(limit=limit, used=used, remaining=max(0, limit-used))
        return dict(enforced=True, mode='online', **values, resets_at=reset,
                    timezone=os.getenv('VENUS_QUOTA_TIMEZONE', 'Asia/Shanghai'))

    def status(self, owner=None, actor=None):
        day, reset = self.day()
        with self.connect() as c:
            row = c.execute('SELECT visitor FROM compute_owners WHERE owner=?', (owner,)).fetchone()
            visitor = row['visitor'] if row else actor[1] if actor else ''
            return self._status(c, visitor, day, reset)

    def admit(self, scope, owner, demands):
        demands = {key: value for key, value in demands.items() if value}
        if any(key not in LIMITS or type(value) is not int or value < 0 for key, value in demands.items()):
            raise ValueError('Invalid compute allowance')
        if not demands:
            return False
        day, reset = self.day()
        with self.connect(True) as c:
            previous = c.execute('SELECT resource,units FROM compute_admissions WHERE scope=?', (scope,)).fetchall()
            if previous:
                if {row['resource']: row['units'] for row in previous} != demands:
                    raise ValueError('A submitted request cannot change its compute allowance')
                return False
            actor = c.execute('SELECT * FROM compute_owners WHERE owner=?', (owner,)).fetchone()
            if not actor:
                raise HTTPException(403, 'Start this computation from your browser workspace.')
            status = self._status(c, actor['visitor'], day, reset)
            for resource, units in demands.items():
                if units > status[resource]['remaining']:
                    raise HTTPException(429, {'code': 'COMPUTE_DAILY_LIMIT_REACHED', 'resource': resource,
                        'message': f'This job needs {units} {resource} runs; {status[resource]["remaining"]} remain today.',
                        'required': units, 'quota': status})
            now = time.time()
            maximum = positive_setting('VENUS_ONLINE_IP_COMPUTE_STARTS_PER_MINUTE', 30)
            recent = c.execute('SELECT MIN(created),COUNT(DISTINCT scope) FROM compute_admissions WHERE ip=? AND created>?',
                               (actor['ip'], now-60)).fetchone()
            if recent[1] >= maximum:
                retry = max(1, math.ceil(recent[0]+60-now))
                raise HTTPException(429, {'code':'COMPUTE_START_RATE_LIMIT', 'message':f'Too many computations from this network. Try again in {retry} seconds.', 'retry_after':retry}, headers={'Retry-After':str(retry)})
            c.executemany('INSERT INTO compute_admissions VALUES(?,?,?,?,?,?,?)',
                          [(scope, key, day, actor['visitor'], actor['ip'], units, now) for key, units in demands.items()])
            from backend.web_v2.quota_feedback import bound_grant
            grant = bound_grant(c, actor['visitor'])
            if grant:
                c.executemany('INSERT INTO compute_grants VALUES(?,?,?)',[(scope,k,grant['id']) for k in demands])
            return True

    def release(self, scope):
        # Only queue-admission failures refund a reservation; stopping an admitted
        # job does not grant more daily executions.
        with self.connect(True) as c:
            c.execute('DELETE FROM compute_admissions WHERE scope=?', (scope,))
            c.execute('DELETE FROM compute_grants WHERE scope=?', (scope,))


def allowance(owner=None):
    if not online():
        return dict(enforced=False, mode='local')
    return ComputeStore().status(owner, actor_context.get())


def prediction_tool(name, category):
    return category in {'model_prediction', 'molecular_docking', 'molecular_admet'} or name == 'predict_s4pred'


def demands_for(kind, metadata):
    if kind.startswith('platform:') and 'compute_demands' in metadata:
        return dict(metadata['compute_demands'])
    if kind in {'platform:predict', 'custom:predict', 'custom:evaluation'}:
        return {'prediction': 1}
    if kind == 'platform:workflow':
        return {'workflow': 1}
    if kind.startswith('mcp:') and prediction_tool(metadata.get('tool'), kind[4:]):
        return {'prediction': 1}
    return {}


def enqueue(q, owner, kind, resource, spec, namespace, *, metadata=None, priority=0,
            session_key=None, request_key=None, request_hash=None):
    """Reserve before enqueue, retaining queue idempotency and rollback semantics."""
    metadata = dict(metadata or {})
    args = dict(metadata=metadata, priority=priority, session_key=session_key,
                request_key=request_key, request_hash=request_hash)
    if not online():
        return q.enqueue(owner, kind, resource, spec, namespace, **args)
    store = ComputeStore()
    store.bind(owner, actor_context.get())
    demands = demands_for(kind, metadata)
    parent_id = metadata.get('parent_job')
    if parent_id and demands:
        parent = q.get(parent_id, owner)
        reserved = parent['metadata'].get('compute_reservation', {})
        if parent['kind'] in {'platform:predict', 'platform:workflow', 'custom:predict'} and reserved.get('demands', {}).get('prediction', 0):
            demands = {}  # A workflow's model nodes were reserved together at admission.
    from agent.rsi.engine.storage import file_lock
    request_id = hashlib.sha256((owner+'|'+request_key).encode()).hexdigest() if request_key else uuid.uuid4().hex
    lock = q.root/'compute-admission-locks'/request_id
    lock.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(lock):
        if request_key:
            existing = q.request(owner, request_key, request_hash)
            if existing:
                return existing['id'], False
        scope = 'compute:'+request_id
        charged = store.admit(scope, owner, demands)
        if demands:
            metadata['compute_reservation'] = {'scope': scope, 'demands': demands}
        try:
            identity, created = q.enqueue(owner, kind, resource, spec, namespace, **args)
        except BaseException:
            if charged:
                store.release(scope)
            raise
        if charged and not created:
            store.release(scope)
        return identity, created


def inline_prediction(owner, name, category):
    if online() and prediction_tool(name, category):
        store = ComputeStore()
        store.bind(owner, actor_context.get())
        store.admit('inline:'+uuid.uuid4().hex, owner, {'prediction': 1})


def metered_prediction(function):
    """Keep legacy HTTP prediction endpoints in the same browser allowance.

    Streaming adapters may call the non-streaming function repeatedly for one
    batch. Charge the HTTP batch once, and retain prepaid queue reservations.
    """
    def admit():
        if not online():
            return
        owner = os.getenv('VENUS_QUEUE_OWNER')
        job_id = os.getenv('VENUS_QUEUE_JOB_ID')
        if owner and job_id:
            from backend.web_v2.job_queue import JobQueue
            row = JobQueue().get(job_id, owner)
            if row['metadata'].get('compute_reservation', {}).get('demands', {}).get('prediction', 0):
                return
        request = inline_request_context.get()
        if request is not None and request.get('admitted'):
            return
        actor = actor_context.get()
        owner = owner or ('browser:'+actor[1] if actor else '')
        inline_prediction(owner, 'legacy_prediction', 'model_prediction')
        if request is not None:
            request['admitted'] = True

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def wrapped(*args, **kwargs):
            from starlette.concurrency import run_in_threadpool
            await run_in_threadpool(admit)
            return await function(*args, **kwargs)
    else:
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            admit()
            return function(*args, **kwargs)
    # FastAPI must resolve postponed annotations in the original module.
    wrapped.__signature__ = inspect.signature(function, eval_str=True)
    return wrapped
