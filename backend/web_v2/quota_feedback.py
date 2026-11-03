"""Feedback requests, durable grant bindings and non-destructive quota resets."""
import json
import time
import secrets

def initialize(c):
    c.executescript("""
    CREATE TABLE IF NOT EXISTS grant_compute(grant_id TEXT PRIMARY KEY, extra_predictions INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS visitor_grants(visitor TEXT PRIMARY KEY, grant_id TEXT NOT NULL, key_hash TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS quota_reset(id INTEGER PRIMARY KEY CHECK(id=1), since REAL NOT NULL);
    CREATE TABLE IF NOT EXISTS visitor_days(visitor TEXT NOT NULL, day TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL, visits INTEGER NOT NULL, PRIMARY KEY(visitor,day));
    CREATE TABLE IF NOT EXISTS quota_feedback(id TEXT PRIMARY KEY, visitor TEXT NOT NULL, created REAL NOT NULL, payload TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS compute_grants(scope TEXT NOT NULL, resource TEXT NOT NULL, grant_id TEXT NOT NULL, PRIMARY KEY(scope,resource));
    CREATE INDEX IF NOT EXISTS compute_grants_by_key ON compute_grants(grant_id,resource);
    CREATE INDEX IF NOT EXISTS visitor_last_seen ON visitor_days(last_seen);
    CREATE INDEX IF NOT EXISTS feedback_created ON quota_feedback(created);
    """)

def since(c):
    row = c.execute("SELECT since FROM quota_reset WHERE id=1").fetchone()
    return row[0] if row else 0

def bound_grant(c, visitor):
    return c.execute("""SELECT g.*, COALESCE(x.extra_predictions,0) AS extra_predictions
        FROM visitor_grants b JOIN grants g ON g.id=b.grant_id AND g.key_hash=b.key_hash
        LEFT JOIN grant_compute x ON x.grant_id=g.id
        WHERE b.visitor=? AND g.active=1 AND (g.expires IS NULL OR g.expires>?)""", (visitor,time.time())).fetchone()

def bind(store, visitor, address, key):
    with store.connect(True) as c:
        grant=store.grant(c,address,key,required=True)
        c.execute("INSERT OR REPLACE INTO visitor_grants VALUES(?,?,?)",(visitor,grant['id'],grant['key_hash']))

def unbind(store, visitor):
    with store.connect(True) as c:
        c.execute("DELETE FROM visitor_grants WHERE visitor=?",(visitor,))

def visit(store, visitor, *, document=True):
    day,_=store.day(); now=time.time()
    with store.connect(True) as c:
        c.execute("""INSERT INTO visitor_days VALUES(?,?,?,?,?)
            ON CONFLICT(visitor,day) DO UPDATE SET last_seen=excluded.last_seen,
                visits=visits+excluded.visits""", (visitor,day,now,now,int(document)))

def feedback(store, visitor, payload):
    from backend.web_v2.online_access import email
    payload=dict(payload);payload['email']=email(payload['email'])
    if not payload['feedback'].strip(): raise ValueError("Please describe a bug or suggestion")
    store.throttle(['feedback:'+visitor],5,86400)
    identity=secrets.token_hex(16)
    with store.connect(True) as c:
        c.execute("INSERT INTO quota_feedback VALUES(?,?,?,?)",(identity,visitor,time.time(),json.dumps(payload)))
    return {'id':identity}

def requests(store):
    with store.connect() as c:
        return [{'id':r['id'],'created':r['created'],**json.loads(r['payload'])} for r in c.execute("SELECT * FROM quota_feedback ORDER BY created DESC LIMIT 200")]

def reset(store, actor):
    with store.connect(True) as c:
        now=time.time()
        c.execute("INSERT OR REPLACE INTO quota_reset VALUES(1,?)",(now,))
        store.log(c,actor['email'],'quota.reset_all',str(now))
    return {'ok':True,'reset_at':now}

def overview(store):
    day,_=store.day();now=time.time()
    with store.connect() as c:
        cutoff=since(c)
        visits=c.execute("SELECT COUNT(DISTINCT visitor),COALESCE(SUM(visits),0) FROM visitor_days WHERE day=?",(day,)).fetchone()
        total=c.execute("SELECT COUNT(*) FROM (SELECT visitor FROM visitor_days UNION SELECT visitor FROM admissions)").fetchone()[0]
        active=c.execute("SELECT COUNT(DISTINCT visitor) FROM visitor_days WHERE last_seen>?",(now-300,)).fetchone()[0]
        modes={k:0 for k in ('agent','scientist','researcher')}
        for row in c.execute("SELECT mode,COUNT(*) AS n FROM admissions WHERE day=? GROUP BY mode",(day,)):
            key={'science_agent':'agent','science_expert':'scientist','science':'scientist','report':'researcher'}.get(row['mode'],row['mode'])
            if key in modes: modes[key]+=row['n']
        used=c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND created>?",(day,cutoff)).fetchone()[0]
        compute={row['resource']:row['n'] for row in c.execute("SELECT resource,SUM(units) AS n FROM compute_admissions WHERE day=? AND created>? GROUP BY resource",(day,cutoff))}
        grants=c.execute("SELECT COUNT(*) FROM grants WHERE active=1 AND (expires IS NULL OR expires>?)",(now,)).fetchone()[0]
    from backend.web_v2.usage_store import UsageStore
    from datetime import datetime
    from zoneinfo import ZoneInfo
    import os
    start = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(os.getenv('VENUS_QUOTA_TIMEZONE','Asia/Shanghai'))).timestamp()
    with UsageStore().connection() as c:
        mode_calls = {row['mode']:dict(row) for row in c.execute("""SELECT mode,
            SUM(kind='model') AS model_calls,SUM(kind='tool') AS tool_calls,
            SUM(CASE WHEN kind='model' THEN total_tokens END) AS tokens
            FROM usage_events WHERE created>=? GROUP BY mode""",(start,))}
    return dict(mode_calls=mode_calls,day=day,visitors_today=visits[0],visits_today=visits[1],visitors_total=total,recent_visitors=active,
        modes=modes,conversations_used=used,predictions_used=compute.get('prediction',0),workflows_used=compute.get('workflow',0),
        active_grants=grants,last_reset=cutoff)
