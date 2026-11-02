"""Atomic online conversation admission and administrator credential storage.

No secrets are stored in the repository. The database is scoped to the checkout.
"""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import hmac
import ipaddress
import os
from pathlib import Path
import re
import secrets
import sqlite3
import time
from zoneinfo import ZoneInfo

from fastapi import HTTPException

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


def client_ip(request):
    # A proxy must overwrite/append XFF and be explicitly trusted. Never trust
    # a client-supplied leftmost address. Uvicorn must use --no-proxy-headers.
    peer = request.client.host if request.client else 'unknown'
    networks = []
    for value in os.getenv('VENUS_TRUSTED_PROXIES', '').split(','):
        try: networks.append(ipaddress.ip_network(value.strip()))
        except ValueError: pass
    def trusted(value):
        try: return any(ipaddress.ip_address(value) in n for n in networks)
        except ValueError: return False
    if trusted(peer):
        chain = [x.strip() for x in request.headers.get('x-forwarded-for', '').split(',') if x.strip()] + [peer]
        for value in reversed(chain):
            try: ipaddress.ip_address(value)
            except ValueError: return peer
            if not trusted(value):
                peer = value
                break
    try:
        address = ipaddress.ip_address(peer)
        # Group IPv6 privacy addresses by /64.
        return str(ipaddress.ip_network(str(address) + '/64', strict=False)) if address.version == 6 else str(address)
    except ValueError: return peer


def secure_request(request):
    if request.url.scheme == 'https': return True
    peer = request.client.host if request.client else ''
    if request.headers.get('x-forwarded-proto', '').strip().lower() != 'https': return False
    for value in os.getenv('VENUS_TRUSTED_PROXIES', '').split(','):
        try:
            if ipaddress.ip_address(peer) in ipaddress.ip_network(value.strip()): return True
        except ValueError: pass
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
            CREATE TABLE IF NOT EXISTS grants(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, key_hash TEXT NOT NULL, extra INTEGER NOT NULL, active INTEGER NOT NULL, expires REAL, name TEXT NOT NULL, institution TEXT NOT NULL, contact TEXT NOT NULL, note TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS admins(id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, active INTEGER NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS admin_sessions(token_hash TEXT PRIMARY KEY, admin_id TEXT NOT NULL, expires REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts(subject TEXT NOT NULL, at REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS attempts_at ON attempts(subject,at);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, at REAL NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, target TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS science_slots(scope TEXT PRIMARY KEY, ip TEXT NOT NULL, visitor TEXT NOT NULL, grant_id TEXT, expires REAL NOT NULL);
            ''')
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
        base = self.limit()
        ip_used = c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND ip=? AND bucket='base'", (day,ip)).fetchone()[0]
        browser_used = c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND visitor=? AND bucket='base'", (day,visitor)).fetchone()[0]
        base_used = max(ip_used, browser_used)
        extra_used = grant_used = 0
        if grant:
            grant_used = c.execute('SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=?', (day,grant['id'])).fetchone()[0]
            extra_used = c.execute("SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=? AND bucket='extra'", (day,grant['id'])).fetchone()[0]
        extra = grant['extra'] if grant else 0
        remaining = max(0,base-base_used) + max(0,extra-extra_used)
        if grant: remaining = min(remaining, max(0,base+extra-grant_used))
        return dict(mode='online', enforced=True, limit=base+extra, used=base+extra-remaining,
                    remaining=remaining, base_limit=base, extra_limit=extra, resets_at=reset,
                    timezone=os.getenv('VENUS_QUOTA_TIMEZONE','Asia/Shanghai'), email=grant['email'] if grant else None,
                    application_email=CONTACT, unit='new_conversation', base_remaining=max(0,base-base_used))

    def status(self, ip, visitor, address='', key=''):
        day, reset = self.day()
        with self.connect() as c:
            return self._status(c,ip,visitor,self.grant(c,address,key),day,reset)

    def admit(self, scope, mode, ip, visitor, address='', key=''):
        day, reset = self.day()
        with self.connect(True) as c:
            # Access to the conversation itself is checked by each HTTP adapter.
            if c.execute('SELECT 1 FROM admissions WHERE scope=?',(scope,)).fetchone(): return False
            grant = self.grant(c,address,key)
            status = self._status(c,ip,visitor,grant,day,reset)
            if status['remaining'] <= 0:
                raise HTTPException(429, {'code':'CHAT_DAILY_LIMIT_REACHED','message':'Today’s new conversation allowance has been used. Continue an existing conversation or add an extra quota key.', 'quota':status})
            bucket = 'base' if status['base_remaining'] > 0 else 'extra'
            c.execute('INSERT INTO admissions VALUES(?,?,?,?,?,?,?,?)',(scope,day,ip,visitor,grant['id'] if grant else None,bucket,mode,time.time()))
            return True

    def science_slot(self, scope, ip, visitor, address='', key=''):
        with self.connect(True) as c:
            c.execute('DELETE FROM science_slots WHERE expires<?',(time.time(),))
            grant = self.grant(c,address,key)
            gid = grant['id'] if grant else ''
            existing = c.execute('SELECT * FROM science_slots WHERE scope=?',(scope,)).fetchone()
            if existing: return
            if c.execute('SELECT 1 FROM science_slots WHERE ip=? OR visitor=? OR grant_id=?',(ip,visitor,gid)).fetchone():
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
            return [{**dict(r),'used_today':c.execute('SELECT COUNT(*) FROM admissions WHERE day=? AND grant_id=?',(day,r['id'])).fetchone()[0]} for r in rows]

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
        if scope['type']!='http' or not scope['path'].startswith('/api/'):
            return await self.app(scope,receive,send)
        from starlette.requests import Request
        from starlette.responses import Response
        request=Request(scope)
        value=request.cookies.get(COOKIE,'')
        fresh=not re.fullmatch('[a-f0-9]{64}',value)
        if fresh: value=secrets.token_hex(32)
        scope.setdefault('state',{})['quota_visitor']=value
        async def wrapped(message):
            if message['type']=='http.response.start':
                headers=list(message.get('headers',[]))
                if fresh:
                    response=Response()
                    response.set_cookie(COOKIE,value,max_age=365*86400,httponly=True,samesite='lax',secure=secure_request(request),path='/')
                    headers.extend((k,v) for k,v in response.raw_headers if k==b'set-cookie')
                message['headers']=headers
            await send(message)
        await self.app(scope,receive,wrapped)
