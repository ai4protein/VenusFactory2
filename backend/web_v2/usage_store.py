"""Private operational telemetry: identifiers and counts, never prompts or secrets."""
from __future__ import annotations
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import sqlite3
import time
import uuid

LOG = logging.getLogger(__name__)
MODES = {'agent', 'scientist', 'researcher', 'tools'}


def normalize_usage(usage):
    if not isinstance(usage, dict): return None, None, None
    def number(*names):
        for name in names:
            value = usage.get(name)
            if type(value) is int and value >= 0: return value
        return None
    inp = number('prompt_tokens', 'input_tokens')
    out = number('completion_tokens', 'output_tokens')
    total = number('total_tokens')
    if total is None and inp is not None and out is not None: total = inp + out
    return inp, out, total


@contextmanager
def read_db(path):
    c = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=2)
    c.row_factory = sqlite3.Row
    try: yield c
    finally: c.close()


def execution_context():
    """Resolve attribution from trusted queue metadata, including MCP children."""
    run = os.getenv('VENUS_QUEUE_JOB_ID', '')
    result = {'run_id': run, 'session_id': '', 'mode': 'tools', 'harness': ''}
    db = Path(os.getenv('VENUS_JOB_QUEUE_ROOT', '')) / 'jobs.sqlite3'
    if not run or not db.is_file(): return result
    with read_db(db) as c:
        cursor = run
        for _ in range(8):
            row = c.execute('SELECT kind,metadata FROM jobs WHERE id=?', (cursor,)).fetchone()
            if not row: break
            meta = json.loads(row['metadata'])
            result['session_id'] = result['session_id'] or meta.get('session_id') or meta.get('report_id') or ''
            mode = meta.get('mode') or row['kind']
            if mode in MODES: result['mode'] = mode
            result['harness'] = result['harness'] or meta.get('harness', '')
            cursor = meta.get('parent_job')
            if not cursor: break
    sessions_db=Path(os.getenv('VENUSFACTORY_SESSION_DB_PATH',''))
    if result['mode']=='tools' and result['session_id'] and sessions_db.is_file():
        with read_db(sessions_db) as c:
            row=c.execute("SELECT json_extract(data,'$.chat_mode') FROM sessions WHERE session_id=?",(result['session_id'],)).fetchone()
            if row:result['mode']='scientist' if row[0]=='science_expert' else 'agent'
    return result


class UsageStore:
    def __init__(self, path=None):
        from .runtime_paths import runtime_root
        self.path = Path(path or os.getenv('VENUS_USAGE_DB') or runtime_root()/'state/usage.sqlite3')
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as c:
            c.execute('PRAGMA journal_mode=WAL')
            c.execute('''CREATE TABLE IF NOT EXISTS usage_events (
                id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL,
                kind TEXT NOT NULL, mode TEXT NOT NULL, session_id TEXT NOT NULL,
                run_id TEXT NOT NULL, name TEXT NOT NULL, provider TEXT NOT NULL,
                harness TEXT NOT NULL, status TEXT NOT NULL, latency_ms INTEGER,
                input_tokens INTEGER, output_tokens INTEGER, total_tokens INTEGER,
                error_type TEXT NOT NULL DEFAULT '')''')
            c.execute('CREATE INDEX IF NOT EXISTS usage_time ON usage_events(created)')
            c.execute('CREATE INDEX IF NOT EXISTS usage_session ON usage_events(session_id,created)')
            c.execute('CREATE INDEX IF NOT EXISTS usage_kind ON usage_events(kind,mode,created)')
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.path, timeout=2)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally: c.close()

    def record(self, *, identity, kind, name, status, context=None, provider='', usage=None, latency_ms=None, error_type=''):
        context = context or {}
        inp, out, total = normalize_usage(usage)
        now = time.time()
        mode = context.get('mode', 'tools')
        if mode not in MODES: mode = 'tools'
        values = (str(identity)[:200], now, now, kind, mode, str(context.get('session_id', ''))[:128],
                  str(context.get('run_id', ''))[:128], str(name)[:240], str(provider)[:120],
                  str(context.get('harness', ''))[:80], status, latency_ms, inp, out, total, str(error_type)[:100])
        with self.connection() as c:
            c.execute('''INSERT INTO usage_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,status=excluded.status,
                latency_ms=COALESCE(excluded.latency_ms,usage_events.latency_ms),
                input_tokens=COALESCE(excluded.input_tokens,usage_events.input_tokens),
                output_tokens=COALESCE(excluded.output_tokens,usage_events.output_tokens),
                total_tokens=COALESCE(excluded.total_tokens,usage_events.total_tokens),error_type=excluded.error_type''', values)

    @staticmethod
    def where(days, mode='', session_id='', kind=''):
        clauses = ['created>=?']; args = [time.time()-days*86400]
        for key, value in [('mode',mode), ('session_id',session_id), ('kind',kind)]:
            if value: clauses.append(key+'=?'); args.append(value)
        return ' AND '.join(clauses), args

    def overview(self, days=7, mode='', session_id=''):
        self.reconcile()
        where, args = self.where(days, mode, session_id)
        with self.connection() as c:
            summary = dict(c.execute(f'''SELECT COUNT(*) AS events,
                SUM(kind='model') AS model_calls,SUM(kind='tool') AS tool_calls,
                SUM(status='failed') AS failed,SUM(status='running') AS running,
                SUM(CASE WHEN kind='model' THEN input_tokens END) AS input_tokens,
                SUM(CASE WHEN kind='model' THEN output_tokens END) AS output_tokens,
                SUM(CASE WHEN kind='model' THEN total_tokens END) AS total_tokens,
                SUM(kind='model' AND total_tokens IS NOT NULL) AS reported_calls,
                SUM(kind='model' AND total_tokens IS NULL) AS unreported_calls
                FROM usage_events WHERE {where}''', args).fetchone())
            models = [dict(r) for r in c.execute(f'''SELECT name,provider,harness,COUNT(*) AS calls,
                SUM(status='failed') AS failed,COUNT(total_tokens) AS reported_calls,
                SUM(input_tokens) AS input_tokens,SUM(output_tokens) AS output_tokens,SUM(total_tokens) AS total_tokens,
                AVG(latency_ms) AS latency_ms FROM usage_events WHERE {where} AND kind='model'
                GROUP BY name,provider,harness ORDER BY calls DESC LIMIT 100''',args)]
            tools = [dict(r) for r in c.execute(f'''SELECT name,COUNT(*) AS calls,SUM(status='failed') AS failed,
                AVG(latency_ms) AS latency_ms FROM usage_events WHERE {where} AND kind='tool'
                GROUP BY name ORDER BY calls DESC LIMIT 100''',args)]
            daily = [dict(r) for r in c.execute(f'''SELECT date(created,'unixepoch') AS day,
                SUM(kind='model') AS model_calls,SUM(kind='tool') AS tool_calls,
                SUM(CASE WHEN kind='model' THEN total_tokens END) AS total_tokens
                FROM usage_events WHERE {where} GROUP BY day ORDER BY day''',args)]
            since = c.execute('SELECT MIN(created) FROM usage_events').fetchone()[0]
        return {'summary':summary,'models':models,'tools':tools,'daily':daily,'recording_since':since,
                'deployment':os.getenv('WEBUI_V2_MODE','local'),'at':time.time()}

    def events(self, days=7, mode='', session_id='', kind='', offset=0, limit=25):
        self.reconcile()
        where, args = self.where(days,mode,session_id,kind)
        with self.connection() as c:
            total = c.execute('SELECT COUNT(*) FROM usage_events WHERE '+where,args).fetchone()[0]
            rows = [dict(r) for r in c.execute('SELECT * FROM usage_events WHERE '+where+' ORDER BY created DESC,id DESC LIMIT ? OFFSET ?',[*args,limit,offset])]
        return {'items':rows,'total':total,'offset':offset,'limit':limit}

    def reconcile(self):
        db=Path(os.getenv('VENUS_JOB_QUEUE_ROOT',''))/'jobs.sqlite3'
        if not db.is_file():return
        with self.connection() as c, read_db(db) as q:
            for row in c.execute("SELECT id,run_id,created FROM usage_events WHERE status='running' AND run_id!=''").fetchall():
                job=q.execute('SELECT status,finished FROM jobs WHERE id=?',(row['run_id'],)).fetchone()
                if job and job['status'] not in {'running','queued'}:
                    c.execute("UPDATE usage_events SET status='interrupted',updated=?,error_type='WorkerEnded' WHERE id=?",(job['finished'] or time.time(),row['id']))

    def jobs(self, days=7, mode='', session_id='', offset=0, limit=25):
        """Read retained queue history separately from logical tool call counts."""
        db=Path(os.getenv('VENUS_JOB_QUEUE_ROOT',''))/'jobs.sqlite3'
        if not db.is_file():return {'items':[],'total':0,'offset':offset,'limit':limit}
        with read_db(db) as c:
            columns="""id,kind,resource,status,created,started,finished,
                json_extract(metadata,'$.mode') AS mode,
                COALESCE(json_extract(metadata,'$.session_id'),json_extract(metadata,'$.report_id'),'') AS session_id,
                json_extract(metadata,'$.parent_job') AS parent_id"""
            rows=[dict(r) for r in c.execute('SELECT '+columns+' FROM jobs WHERE created>=? ORDER BY created DESC,id DESC',(time.time()-days*86400,))]
            parents={r['id']:dict(r) for r in rows}
            for row in rows:
                cursor=row;seen=set()
                for _ in range(8):
                    if cursor['id'] in seen:break
                    seen.add(cursor['id'])
                    row['session_id']=row['session_id'] or cursor['session_id']
                    candidate=cursor['mode'] or cursor['kind']
                    if candidate in MODES:row['mode']=candidate
                    if candidate in {'protein-research','research'}:row['mode']='researcher'
                    parent=cursor['parent_id']
                    if not parent:break
                    if parent not in parents:
                        result=c.execute('SELECT '+columns+' FROM jobs WHERE id=?',(parent,)).fetchone()
                        parents[parent]=dict(result) if result else None
                    cursor=parents[parent]
                    if not cursor:break
                if row['mode'] not in MODES:row['mode']='tools'
                row['latency_ms']=round((row['finished']-row['started'])*1000) if row['finished'] and row['started'] else None
        rows=[r for r in rows if (not mode or r['mode']==mode) and (not session_id or r['session_id']==session_id)]
        return {'items':rows[offset:offset+limit],'total':len(rows),'offset':offset,'limit':limit}

    def sessions(self, days=7, mode='', session_id='', offset=0, limit=25):
        since = time.time()-days*86400; rows = {}
        sessions_db = Path(os.getenv('VENUSFACTORY_SESSION_DB_PATH',''))
        if sessions_db.is_file():
            with read_db(sessions_db) as c:
                for r in c.execute('''SELECT session_id,created_at,last_accessed,
                    json_extract(data,'$.chat_mode') AS mode,json_extract(data,'$.status') AS status,
                    json_array_length(data,'$.history') AS messages,
                    substr(COALESCE(json_extract(data,'$.history[0].content'),json_extract(data,'$.last_user_text'),''),1,100) AS title
                    FROM sessions WHERE last_accessed>=?''',(since,)):
                    d=dict(r);d['mode']='scientist' if d['mode']=='science_expert' else 'agent'
                    d.update(updated=d.pop('last_accessed'),created=d.pop('created_at'),runs=0)
                    rows[d['session_id']]=d
        queue_db=Path(os.getenv('VENUS_JOB_QUEUE_ROOT',''))/'jobs.sqlite3'
        if queue_db.is_file():
            with read_db(queue_db) as c:
                jobs=c.execute('''SELECT id,kind,status,created,started,finished,
                    json_extract(metadata,'$.session_id') AS session_id,json_extract(metadata,'$.report_id') AS report_id,
                    substr(json_extract(metadata,'$.title'),1,100) AS title
                    FROM jobs WHERE kind IN ('agent','scientist','research','protein-research') AND created>=? ORDER BY created''',(since,))
                for r in jobs:
                    sid=r['session_id'] or r['report_id']
                    if not sid:continue
                    d=rows.setdefault(sid,{'session_id':sid,'mode':'researcher' if r['kind'] in {'research','protein-research'} else r['kind'],
                        'title':r['title'] or '', 'messages':None,'created':r['created'],'updated':r['created'],'runs':0})
                    d.update(status=r['status'],run_id=r['id'],runs=d['runs']+1,updated=max(d['updated'] or 0,r['finished'] or r['started'] or r['created']))
        where,args=self.where(days,mode,session_id)
        with self.connection() as c:
            for r in c.execute(f'''SELECT session_id,SUM(kind='model') AS model_calls,SUM(kind='tool') AS tool_calls,
                SUM(CASE WHEN kind='model' THEN total_tokens END) AS total_tokens FROM usage_events
                WHERE {where} GROUP BY session_id''',args):
                if r['session_id'] in rows:rows[r['session_id']].update(dict(r))
        items=sorted([r for r in rows.values() if (not mode or r['mode']==mode) and (not session_id or r['session_id']==session_id)],key=lambda r:r['updated'] or 0,reverse=True)
        return {'items':items[offset:offset+limit],'total':len(items),'active':sum(r['status'] in {'queued','running','stopping'} for r in items),'offset':offset,'limit':limit}


class Observation:
    """A single request/attempt. Recording failure must not break execution."""
    def __init__(self, kind, name, *, identity=None, provider='', context=None):
        self.id=identity or uuid.uuid4().hex; self.kind=kind; self.name=name; self.provider=provider
        self.context=context; self.started=time.monotonic()
        self.usage=None
        self.write('running')

    def __enter__(self):return self

    def __exit__(self, kind, value, tb):
        self.finish(usage=self.usage,failed=kind is not None,error_type=kind.__name__ if kind else '')

    def write(self,status,usage=None,error_type=''):
        if not os.getenv('VENUS_USAGE_DB'):return
        try:
            if self.context is None:self.context=execution_context()
            UsageStore().record(identity=self.id,kind=self.kind,name=self.name,provider=self.provider,context=self.context,
                status=status,usage=usage,latency_ms=None if status=='running' else round((time.monotonic()-self.started)*1000),error_type=error_type)
        except Exception: LOG.warning('Usage recording unavailable',exc_info=False)

    def finish(self, *, usage=None, failed=False, error_type=''):
        self.write('failed' if failed else 'completed',usage,error_type)
