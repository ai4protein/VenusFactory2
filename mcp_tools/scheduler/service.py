"""Loopback-only independent MCP process, reusable across Web restarts."""
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[2]
REPOSITORY_ID=hashlib.sha256(str(ROOT).encode()).hexdigest()[:16]


def port():return int(os.getenv('MCP_HTTP_PORT','8080'))
def base_url():return 'http://127.0.0.1:'+str(port())
def control_dir():
    from backend.web_v2.job_queue import JobQueue
    p=JobQueue().root/'mcp-service'/str(port());p.mkdir(parents=True,exist_ok=True,mode=0o700);return p


def service_key():
    path=control_dir()/'service.key'
    try:
        fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:pass
    else:
        with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(48))
    return path.read_text().strip()


def health():
    with urlopen(base_url()+'/health',timeout=2) as response:return json.load(response)


def ensure_service():
    import fcntl
    from backend.web_v2.job_queue import process_token
    control=control_dir()
    with (control/'startup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            state=health()
        except Exception:state=None
        if state:
            if state.get('repository_id')!=REPOSITORY_ID or state.get('service')!='venusfactory-tools':
                raise RuntimeError('MCP port belongs to a different service')
            return state
        service_key()
        bootstrap=control/'bootstrap.py'
        paths=[str(ROOT/'src'),str(ROOT),*(p for p in sys.path if p and Path(p).exists())]
        bootstrap.write_text('import sys,runpy,os\nsys.path='+repr(paths)+'\nos.chdir('+repr(str(ROOT))+')\nrunpy.run_module("mcp_tools.mcp_server",run_name="__main__")\n')
        env=dict(os.environ,MCP_HTTP_HOST='127.0.0.1',MCP_HTTP_PORT=str(port()),PYTHONDONTWRITEBYTECODE='1')
        with (control/'server.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-I','-u',str(bootstrap)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (control/'process.json').write_text(json.dumps({'pid':proc.pid,'token':process_token(proc.pid),'repository_id':REPOSITORY_ID}))
        deadline=time.monotonic()+45
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('MCP service failed to start; inspect its server log')
            try:
                state=health()
                if state.get('repository_id')==REPOSITORY_ID:return state
            except Exception:pass
            time.sleep(.3)
        raise RuntimeError('MCP service did not become ready')
