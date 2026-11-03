"""Atomic online conversation admission and administrator credential storage.

No secrets are stored in the repository. The database is scoped to the checkout.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import hmac
import math
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from zoneinfo import ZoneInfo

from fastapi import HTTPException
from backend.web_v2.http_security import client_ip, secure_request

COOKIE = 'vf_online_visitor'
ADMIN_COOKIE = 'vf_admin_session'
CONTACT = 'tanyang.august@sjtu.edu.cn'
SCIENCE_LIMITS = dict(max_cycles=2, model_calls=48, tool_calls=24, wall_seconds=900)


def online():
    from config import get_config
    return get_config().server.is_online


def fingerprint(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def email(value):
    value = value.strip().lower()
    if len(value) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
        raise ValueError('Enter a valid email address')
    return value


def password_hash(password):
    if not isinstance(password, str) or not 8 <= len(password) <= 256:
        raise ValueError('Use a password of 8–256 characters')
    salt = secrets.token_hex(16)
    key = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=1, maxmem=64*1024**2)
    return f'scrypt${salt}${key.hex()}'


def password_matches(password, encoded):
    try:
        kind, salt, expected = encoded.split('$')
        if kind != 'scrypt' or len(password) > 256: return False
        actual = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=32768, r=8, p=1, maxmem=64*1024**2)
        return hmac.compare_digest(actual.hex(), expected)
    except (ValueError, TypeError):
        return False


class AccessStore:
    def __init__(self, path=None):
        root = Path(__file__).resolve().parents[2]
        default = Path.home()/'.local/share/venusfactory/access'/fingerprint(root)[:16]/'access.sqlite3'
        self.path = Path(path or os.getenv('VENUS_ACCESS_DB') or default)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connect() as c:
            c.executescript('''
            CREATE TABLE IF NOT EXISTS admissions(scope TEXT PRIMARY KEY, day TEXT NOT NULL, ip TEXT NOT NULL, visitor TEXT NOT NULL, grant_id TEXT, bucket TEXT NOT NULL, mode TEXT NOT NULL, created REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS admissions_day ON admissions(day,ip,visitor,grant_id);
            CREATE INDEX IF NOT EXISTS admissions_visitor_day ON admissions(visitor,day,bucket);
            CREATE INDEX IF NOT EXISTS admissions_ip_created ON admissions(ip,created);
            CREATE TABLE IF NOT EXISTS grants(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, key_hash TEXT NOT NULL, extra INTEGER NOT NULL, active INTEGER NOT NULL, expires REAL, name TEXT NOT NULL, institution TEXT NOT NULL, contact TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS admins(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, active INTEGER NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS admin_sessions(token_hash TEXT PRIMARY KEY, admin_id TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(subject TEXT NOT NULL, at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_at ON attempts(subject,at);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, at REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS science_slots(scope TEXT PRIMARY KEY, ip TEXT NOT NULL, visitor TEXT NOT NULL, grant_id TEXT, expires REAL NOT NULL);
            ''')
        from backend.web_v2.quota_feedback import initialize
        with self.connect() as c: initialize(c)
        self.path.chmod(0o600)

    @contextmanager
    def connect(self, write=False):
        c = sqlite3.connect(self.path, timeout=20)
        c.row_factory = sqlite3.Row
        try:
            c.execute('PRAGMA busy_timeout=20000')
            if write: c.execute('BEGIN IMMEDIATE')
            yield c
            c.commit()
        except BaseException:
            c.rollback(); raise
        finally: c.close()

    @staticmethod
    def day():
        zone = ZoneInfo(os.getenv('VENUS_QUOTA_TIMEZONE', 'Asia/Shanghai'))
        now = datetime.now(zone)
        reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return now.date().isoformat(), reset.isoformat()

    @staticmethod
    def limit():
        # Keep the existing deployment setting, defaulting to three.
        return max(1, int(os.getenv('WEBUI_V2_ONLINE_DAILY_CHAT_LIMIT', '3')))

    def grant(self, c, address='', key='', *, required=False):
        if not address and not key and not required: return None
        try: address = email(address)
        except ValueError: address = ''
        row = c.execute('SELECT * FROM grants WHERE email=?', (address,)).fetchone()
        valid = row and hmac.compare_digest(row['key_hash'], fingerprint(key)) and row['active'] and (row['expires'] is None or row['expires'] > time.time())
        if not valid:
            raise HTTPException(401, {'code':'QUOTA_KEY_INVALID', 'message':'This email or quota key is invalid, expired or disabled.'})
        return row

    def _status(self, c, ip, visitor, grant, day, reset):
        from backend.web_v2.quota_feedback import since
        cutoff = since(c)
        base = self.limit()
        # The persistent visitor cookie owns the daily allowance. A shared NAT
        # or reverse proxy must never merge different visitors' daily balances.
        base_used = c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND visitor=? AND bucket='base' AND created>?", (day,visitor,cutoff)).fetchone()[0]
        extra_used = grant_used = 0
        if grant:
            grant_used = c.execute('SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=? AND created>?', (day,grant['id'],cutoff)).fetchone()[0]
            extra_used = c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=? AND bucket='extra' AND created>?", (day,grant['id'],cutoff)).fetchone()[0]
        extra = grant['extra'] if grant else 0
        remaining = max(0,base-base_used) + max(0,extra-extra_used)
        if grant: remaining = min(remaining, max(0,base+extra-grant_used))
        return dict(mode='online', enforced=True, limit=base+extra, used=base+extra-remaining,
                    remaining=remaining, base_limit=base, extra_limit=extra, resets_at=reset,
                    timezone=os.getenv('VENUS_QUOTA_TIMEZONE','Asia/Shanghai'), email=grant['email'] if grant else None,
                    application_email=CONTACT, unit='new_conversation', identity_scope='visitor', base_remaining=max(0,base-base_used))

    def owned_chat_ids(self, visitor):
        with self.connect() as c:
            return {row[0][5:] for row in c.execute(
                "SELECT scope FROM admissions WHERE visitor=? AND scope LIKE 'chat:%'", (visitor,))}

    def status(self, ip, visitor, address='', key=''):
        day, reset = self.day()
        with self.connect() as c:
            return self._status(c,ip,visitor,self.visitor_grant(c,visitor,address,key),day,reset)

    def visitor_grant(self, c, visitor, address='', key=''):
        from backend.web_v2.quota_feedback import bound_grant
        return self.grant(c,address,key) if address or key else bound_grant(c,visitor)

    @staticmethod
    def _check_network_rate(c, ip, now):
        """A short burst guard, separate from every visitor's daily balance."""
        maximum = max(1, int(os.getenv('VENUS_ONLINE_IP_STARTS_PER_MINUTE', '30')))
        row = c.execute(
            'SELECT created FROM admissions WHERE ip=? AND created>? ORDER BY created DESC LIMIT 1 OFFSET ?',
            (ip, now - 60, maximum - 1),
        ).fetchone()
        if row:
            retry = max(1, math.ceil(row['created'] + 60 - now))
            raise HTTPException(429, {
                'code': 'CHAT_START_RATE_LIMIT',
                'message': f'Too many new conversations from this network. Try again in {retry} seconds.',
                'retry_after': retry,
            }, headers={'Retry-After': str(retry)})

    def admit(self, scope, mode, ip, visitor, address='', key=''):
        day, reset = self.day()
        with self.connect(True) as c:
            # Access to the conversation itself is checked by each HTTP adapter.
            if c.execute('SELECT 1 FROM admissions WHERE scope=?',(scope,)).fetchone(): return False
            grant = self.visitor_grant(c,visitor,address,key)
            status = self._status(c,ip,visitor,grant,day,reset)
            if status['remaining'] <= 0:
                raise HTTPException(429, {'code':'CHAT_DAILY_LIMIT_REACHED','message':'Today’s new conversation allowance has been used. Continue an existing conversation or add an extra quota key.', 'quota':status})
            now = time.time()
            self._check_network_rate(c, ip, now)
            bucket = 'base' if status['base_remaining'] > 0 else 'extra'
            c.execute('INSERT INTO admissions VALUES(?,?,?,?,?,?,?,?)',(scope,day,ip,visitor,grant['id'] if grant else None,bucket,mode,now))
            return True

    def science_slot(self, scope, ip, visitor, address='', key=''):
        with self.connect(True) as c:
            c.execute('DELETE FROM science_slots WHERE expires<?',(time.time(),))
            grant = self.visitor_grant(c,visitor,address,key)
            gid = grant['id'] if grant else ''
            existing = c.execute('SELECT * FROM science_slots WHERE scope=?',(scope,)).fetchone()
            if existing: return
            if c.execute('SELECT 1 FROM science_slots WHERE visitor=? OR grant_id=?',(visitor,gid)).fetchone():
                raise HTTPException(429, {'code':'SCIENCE_ALREADY_RUNNING','message':'One Scientist study can run at a time. Finish or cancel the current study first.'})
            # Includes at most an hour waiting in the queue and a 15-minute run.
            c.execute('INSERT INTO science_slots VALUES(?,?,?,?,?)',(scope,ip,visitor,gid or None,time.time()+4500))

    def release_science(self, scope):
        with self.connect(True) as c: c.execute('DELETE FROM science_slots WHERE scope=?',(scope,))

    def throttle(self, subjects, maximum=8, seconds=900):
        with self.connect(True) as c:
            c.execute('DELETE FROM attempts WHERE at<?',(time.time()-86400,))
            for subject in subjects:
                count = c.execute('SELECT COUNT(*) FROM attempts WHERE subject=? AND at>?',(subject,time.time()-seconds)).fetchone()[0]
                if count >= maximum: raise HTTPException(429,'Too many attempts. Please try again later.', headers={'Retry-After':str(seconds)})
            c.executemany('INSERT INTO attempts VALUES(?,?)',[(s,time.time()) for s in subjects])

    def bootstrap(self, address, password):
        address = email(address); encoded = password_hash(password)
        with self.connect(True) as c:
            if c.execute('SELECT 1 FROM admins LIMIT 1').fetchone(): return False
            c.execute('INSERT INTO admins VALUES(?,?,?,?,?)',(secrets.token_hex(16),address,encoded,1,time.time()))
            self.log(c,'bootstrap','admin.create',address)
            return True

    @staticmethod
    def log(c, actor, action, target):
        c.execute('INSERT INTO audit(at,actor,action,target) VALUES(?,?,?,?)',(time.time(),actor,action,target))

    def login(self, address, password, ip):
        address = email(address)
        self.throttle(['login:ip:'+ip,'login:email:'+fingerprint(address)])
        with self.connect(True) as c:
            row = c.execute('SELECT * FROM admins WHERE email=?',(address,)).fetchone()
            # Spend the same KDF work for unknown accounts.
            encoded = row['password_hash'] if row else 'scrypt$'+'00'*16+'$'+'00'*64
            if not password_matches(password,encoded) or not row or not row['active']:
                raise HTTPException(401,'Email or password is incorrect')
            token = secrets.token_urlsafe(48)
            c.execute('DELETE FROM admin_sessions WHERE expires<?',(time.time(),))
            c.execute('INSERT INTO admin_sessions VALUES(?,?,?)',(fingerprint(token),row['id'],time.time()+8*3600))
            self.log(c,row['email'],'admin.login',row['id'])
            return token, {'id':row['id'],'email':row['email']}

    def authenticate(self, token):
        with self.connect() as c:
            row = c.execute('SELECT a.id,a.email FROM admins a JOIN admin_sessions s ON s.admin_id=a.id WHERE s.token_hash=? AND s.expires>? AND a.active=1',(fingerprint(token),time.time())).fetchone()
            if not row: raise HTTPException(401,'Sign in to administration')
            return dict(row)

    def admins(self):
        with self.connect() as c: return [dict(r) for r in c.execute('SELECT id,email,active,created FROM admins ORDER BY created')]

    def save_admin(self, actor, data, ident=None):
        address = email(data['email']); encoded = password_hash(data['password']) if data.get('password') else None
        with self.connect(True) as c:
            row = c.execute('SELECT * FROM admins WHERE id=?',(ident,)).fetchone() if ident else None
            if ident and not row: raise HTTPException(404,'Administrator not found')
            if not row and not encoded: raise ValueError('A password is required')
            if row and not data['active']:
                self.protect_admin(c,actor,row)
            ident = ident or secrets.token_hex(16)
            if row:
                c.execute('UPDATE admins SET email=?,password_hash=?,active=? WHERE id=?',(address,encoded or row['password_hash'],int(data['active']),ident))
                c.execute('DELETE FROM admin_sessions WHERE admin_id=?',(ident,))
            else:
                c.execute('INSERT INTO admins VALUES(?,?,?,?,?)',(ident,address,encoded,int(data['active']),time.time()))
            self.log(c,actor['email'],'admin.update' if row else 'admin.create',ident)
            return {'id':ident,'email':address,'active':data['active']}

    @staticmethod
    def protect_admin(c, actor, row):
        if actor['id']==row['id']: raise ValueError('You cannot remove or disable your own account')
        if row['active'] and c.execute('SELECT COUNT(*) FROM admins WHERE active=1').fetchone()[0] <= 1:
            raise ValueError('Keep at least one active administrator')

    def delete_admin(self, actor, ident):
        with self.connect(True) as c:
            row = c.execute('SELECT * FROM admins WHERE id=?',(ident,)).fetchone()
            if not row: raise HTTPException(404,'Administrator not found')
            self.protect_admin(c,actor,row)
            c.execute('DELETE FROM admin_sessions WHERE admin_id=?',(ident,))
            c.execute('DELETE FROM admins WHERE id=?',(ident,))
            self.log(c,actor['email'],'admin.delete',ident)

    def grants(self, query=''):
        day,_ = self.day()
        with self.connect() as c:
            rows = c.execute('SELECT id,email,extra,active,expires,name,institution,contact,note,created FROM grants WHERE instr(lower(email||name||institution),?)>0 ORDER BY created DESC LIMIT 500',(query.lower(),)).fetchall()
            from backend.web_v2.quota_feedback import since
            return [{**dict(r),'extra_predictions':(c.execute('SELECT extra_predictions FROM grant_compute WHERE grant_id=?',(r['id'],)).fetchone() or [0])[0],
                     'used_today':c.execute('SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=? AND created>?',(day,r['id'],since(c))).fetchone()[0]} for r in rows]

    def save_grant(self, actor, data, ident=None, rotate=False):
        address = email(data['email']); extra = data['extra']
        if type(extra) is not int or not 0 <= extra <= 10000: raise ValueError('Extra daily quota must be 0–10000')
        key = 'vfq_'+secrets.token_urlsafe(32) if not ident or rotate else None
        with self.connect(True) as c:
            row = c.execute('SELECT * FROM grants WHERE id=?',(ident,)).fetchone() if ident else None
            if ident and not row: raise HTTPException(404,'Quota key not found')
            ident = ident or secrets.token_hex(16)
            values = (address,fingerprint(key) if key else row['key_hash'],extra,int(data['active']),data.get('expires'),data.get('name',''),data.get('institution',''),data.get('contact',''),data.get('note',''))
            if row: c.execute('UPDATE grants SET email=?,key_hash=?,extra=?,active=?,expires=?,name=?,institution=?,contact=?,note=? WHERE id=?',(*values,ident))
            else: c.execute('INSERT INTO grants VALUES(?,?,?,?,?,?,?,?,?,?,?)',(ident,*values,time.time()))
            predictions = data.get('extra_predictions', 0)
            if type(predictions) is not int or not 0 <= predictions <= 10000:
                raise ValueError('Extra predictions must be 0-10000')
            c.execute('INSERT OR REPLACE INTO grant_compute VALUES(?,?)',(ident,predictions))
            self.log(c,actor['email'],'grant.rotate' if rotate else 'grant.save',ident)
            return {'id':ident,'key':key,'email':address}

    def delete_grant(self, actor, ident):
        with self.connect(True) as c:
            if not c.execute('SELECT 1 FROM grants WHERE id=?',(ident,)).fetchone(): raise HTTPException(404,'Quota key not found')
            # Retain a disabled tombstone so re-adding an email cannot reset its daily usage.
            c.execute('UPDATE grants SET active=0,key_hash=? WHERE id=?',(fingerprint(secrets.token_urlsafe(48)),ident))
            self.log(c,actor['email'],'grant.revoke',ident)


def identities(request):
    token = getattr(request.state, 'quota_visitor', '') or request.cookies.get(COOKIE, '')
    if not re.fullmatch('[a-f0-9]{64}',token): token = fingerprint(client_ip(request)+'|'+request.headers.get('user-agent',''))
    return fingerprint(client_ip(request)), fingerprint(token), request.headers.get('x-venus-quota-email','')[:254], request.headers.get('x-venus-quota-key','')[:256]


def admit(request, scope, mode):
    from backend.web_v2.job_access import require_same_origin
    require_same_origin(request)
    if online(): return AccessStore().admit(scope,mode,*identities(request))
    return False


def status(request):
    if not online(): return dict(mode='local',enforced=False,limit=None,used=0,remaining=None)
    return AccessStore().status(*identities(request))


def check_new_conversation(request):
    """Check a draft without charging it; first execution still admits atomically."""
    from backend.web_v2.job_access import require_same_origin
    require_same_origin(request)
    quota = status(request)
    if quota['enforced'] and quota['remaining'] <= 0:
        raise HTTPException(429, {'code':'CHAT_DAILY_LIMIT_REACHED', 'message':'Today’s new conversation allowance has been used. Continue an existing conversation or add an extra quota key.', 'quota':quota})


class VisitorMiddleware:
    """Pure ASGI: attach identity without buffering event streams."""
    def __init__(self, app): self.app=app
    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope,receive,send)
        document = scope.get('method') == 'GET' and b'text/html' in dict(scope.get('headers',[])).get(b'accept',b'')
        if not scope['path'].startswith('/api/') and not document:
            return await self.app(scope,receive,send)
        # Establish identity before concurrent browser API requests.
        from starlette.requests import Request
        from starlette.responses import Response
        request=Request(scope)
        from backend.web_v2.http_security import require_same_origin
        try:
            require_same_origin(request)
        except HTTPException as exc:
            from starlette.responses import JSONResponse
            return await JSONResponse({'detail': exc.detail}, status_code=exc.status_code)(scope, receive, send)
        value=request.cookies.get(COOKIE,'')
        fresh=not re.fullmatch('[a-f0-9]{64}',value)
        if fresh: value=secrets.token_hex(32)
        scope.setdefault('state',{})['quota_visitor']=value
        if document or scope['path'] == '/api/runtime-config':
            from backend.web_v2.quota_feedback import visit
            from starlette.concurrency import run_in_threadpool
            await run_in_threadpool(lambda: visit(AccessStore(),fingerprint(value),document=document))
        async def wrapped(message):
            if message['type']=='http.response.start':
                headers=list(message.get('headers',[]))
                if scope['path'].startswith('/api/chat/sessions') or (document and fresh):
                    # Cookie-authorized history must never enter a shared cache.
                    headers=[(k,v) for k,v in headers if k.lower()!=b'cache-control']
                    headers.append((b'cache-control',b'private, no-store'))
                    headers.append((b'vary',b'Cookie'))
                if fresh:
                    response=Response()
                    response.set_cookie(COOKIE,value,max_age=365*86400,httponly=True,samesite='lax',secure=secure_request(request),path='/')
                    headers.extend((k,v) for k,v in response.raw_headers if k==b'set-cookie')
                message['headers']=headers
            await send(message)
        from backend.web_v2.compute_access import actor_context, request_key_context, inline_request_context
        actor_token = actor_context.set(identities(request)[:2])
        key = request.headers.get('idempotency-key', '')
        request_key = 'http:'+request.method+':'+scope['path']+':'+key if re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', key) else None
        key_token = request_key_context.set(request_key)
        inline_token = inline_request_context.set({})
        try:
            await self.app(scope,receive,wrapped)
        finally:
            inline_request_context.reset(inline_token)
            request_key_context.reset(key_token)
            actor_context.reset(actor_token)
