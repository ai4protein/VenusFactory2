"""Small detached supervisor with fixed workers; queued tasks own no process.

Command environments store references to credentials, never their values. The
supervisor inherits credentials at startup. A namespace isolates runtime modes
and configuration generations while all pools share the same resource limits.
"""
from __future__ import annotations
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .job_queue import JobQueue, PROJECT, TERMINAL, process_token, stop_process_tree
from .task_store import LeaseLost
from backend.runtime_identity import loaded_source_revision

ENV_PREFIXES=('VENUS','WEBUI','PYTHON','CUDA','OMP_','OPENBLAS_','MKL_','CONDA','HF_','HUGGINGFACE','LD_','LC_')
SECRET_WORDS=('KEY','SECRET','TOKEN','PASSWORD','CREDENTIAL','AUTH')


def command_environment(env):
    public={};references=[]
    for key,value in env.items():
        if any(word in key.upper() for word in SECRET_WORDS):references.append(key)
        else:public[key]=str(value)
    return public,sorted(references)


def namespace_for(env):
    # Only the digest is persisted. Changing credentials makes a fresh pool.
    public,refs=command_environment(env)
    stable={k:v for k,v in public.items() if k not in {'VENUS_QUEUE_JOB_ID','VENUS_QUEUE_OWNER','VENUS_QUEUE_LEASE_TOKEN','TEMP_OUTPUTS_DIR','SHLVL','_'}}
    if 'PYTHONPATH' in stable:stable['PYTHONPATH']=os.pathsep.join(dict.fromkeys(stable['PYTHONPATH'].split(os.pathsep)))
    stable.update({k:env[k] for k in refs})
    # The same checkout and environment can contain a new revision after restart.
    stable["__venus_source_revision__"] = loaded_source_revision()
    return hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()[:24]


class QueuedProcess:
    def __init__(self, queue, identity):self.queue,self.identity=queue,identity
    @property
    def pid(self):return self.queue.get(self.identity).get('pid') or 0
    def poll(self):
        row=self.queue.get(self.identity)
        return None if row['status'] not in TERMINAL else row['exit_code'] if row['exit_code'] is not None else 0 if row['status']=='completed' else 1
    def wait(self, timeout=None):
        until=time.monotonic()+timeout if timeout is not None else float('inf')
        while (code:=self.poll()) is None:
            if time.monotonic()>until:raise subprocess.TimeoutExpired('queued task',timeout)
            time.sleep(.05)
        return code
    def kill(self):
        row=self.queue.get(self.identity)
        if row['status']=='running' and process_token(row['pid'])==row['token']:os.kill(row['pid'],signal.SIGKILL)
        elif row['status']=='queued':self.queue.cancel(self.identity)
    def terminate(self):self.queue.cancel(self.identity)


@contextmanager
def pool_guard(q, namespace):
    folder=q.root/'pools'/namespace;folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    with (folder/'start.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        yield folder


def _ensure_pool_locked(q, namespace, env, folder):
    record=folder/'supervisor.json'
    if record.exists():
        try:
            r=json.loads(record.read_text())
            if process_token(r['pid'])==r['token']:return
        except (ValueError,KeyError):pass
    boot=dict(env,VENUS_JOB_QUEUE_ROOT=str(q.root),PYTHONPATH=str(PROJECT)+os.pathsep+str(PROJECT/'src')+os.pathsep+env.get('PYTHONPATH',''))
    for key in ('VENUS_QUEUE_JOB_ID','VENUS_QUEUE_OWNER','VENUS_QUEUE_LEASE_TOKEN'):boot.pop(key,None)
    with (folder/'supervisor.log').open('ab') as log:
        proc=subprocess.Popen([sys.executable,'-u','-m','backend.web_v2.worker_pool','supervise',namespace],env=boot,cwd=PROJECT,
                              stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    token=process_token(proc.pid)
    if not token:raise RuntimeError('Worker supervisor did not start')
    temp=record.with_suffix('.tmp')
    temp.write_text(json.dumps({'pid':proc.pid,'token':token}))
    temp.chmod(0o600)
    os.replace(temp,record)


def ensure_pool(q, namespace, env):
    with pool_guard(q, namespace) as folder:
        _ensure_pool_locked(q, namespace, env, folder)


def submit_command(command, *, owner, kind, resource, env, cwd, stdout, pass_fds, timeout, priority, metadata, queue,
                   session_key, request_key, request_hash):
    q=queue or JobQueue();requested=dict(os.environ if env is None else env)
    # Preserve the application's import environment when started from a bootstrap
    # (e.g. locally installed wheels), not just its inherited PYTHONPATH.
    paths=[str(PROJECT),str(PROJECT/'src'),*requested.get('PYTHONPATH','').split(os.pathsep),*sys.path]
    requested['PYTHONPATH']=os.pathsep.join(dict.fromkeys(p for p in paths if p))
    baseline={**os.environ,**requested}
    for key in ('VENUS_QUEUE_JOB_ID','VENUS_QUEUE_OWNER','VENUS_QUEUE_LEASE_TOKEN'):baseline.pop(key,None)
    baseline['VENUS_JOB_QUEUE_ROOT']=str(q.root)
    baseline['PYTHONPATH']=str(PROJECT)+os.pathsep+str(PROJECT/'src')+os.pathsep+baseline.get('PYTHONPATH','')
    public,refs=command_environment(requested)
    for key in ('VENUS_QUEUE_JOB_ID','VENUS_QUEUE_OWNER','VENUS_QUEUE_LEASE_TOKEN'):public.pop(key,None)
    refs=[key for key in refs if key!='VENUS_QUEUE_LEASE_TOKEN']
    namespace=namespace_for(baseline)
    leases=[str(Path(os.readlink(f'/proc/self/fd/{fd}')).resolve()) for fd in pass_fds]
    log_path=str(Path(stdout.name).resolve()) if stdout is not None else None
    spec={'command':[str(x) for x in command],'cwd':str(cwd),'timeout':float(timeout),
          'environment':public,'credential_refs':refs,'leases':leases,'log':log_path}
    from .compute_access import enqueue
    # Admission and pool retirement share a lock: a newly queued task must not
    # arrive between the supervisor's empty-queue check and its shutdown.
    with pool_guard(q, namespace) as folder:
        identity,created=enqueue(q,owner,kind,resource,spec,namespace,metadata=metadata,priority=priority,
                                  session_key=session_key,request_key=request_key,request_hash=request_hash)
        if created:
            command_file=q.files(identity)/'command.json'
            command_file.write_text(json.dumps({**spec,'pass_fds':[]}));command_file.chmod(0o600)
        try:
            if os.getenv('VENUS_WORKER_AUTOSTART','1')=='1':_ensure_pool_locked(q,namespace,baseline,folder)
        except BaseException:
            if created:q.finish(identity,'failed','Worker pool could not start')
            raise
    return identity,QueuedProcess(q,identity)


def execute(q, row):
    identity,lease,spec=row['id'],row['lease'],row['dispatch']
    env=dict(spec['environment'])
    for key in spec['credential_refs']:
        if key not in os.environ:
            q.finish(identity,'failed','Worker credentials are not configured',lease=lease);return
        env[key]=os.environ[key]
    env.update(VENUS_JOB_QUEUE_ROOT=str(q.root),VENUS_QUEUE_JOB_ID=identity,VENUS_QUEUE_LEASE_TOKEN=lease,VENUS_QUEUE_OWNER=row['owner'])
    env['PYTHONPATH']=str(PROJECT)+os.pathsep+str(PROJECT/'src')+os.pathsep+env.get('PYTHONPATH','')
    proc=None;child_token=None;locks=[];status='failed';message='Worker stopped';code=None
    parent=row['metadata'].get('parent_job')
    log_path=Path(spec['log']) if spec.get('log') else q.files(identity)/'worker.log'
    try:
        from mcp_tools.tools.capability_policy import CapabilityStore
        if row['metadata'].get('capability_tool'):
            CapabilityStore().assert_call(row['metadata']['capability_tool'],row['metadata'].get('capability_arguments',{}))
        for path in spec.get('leases',[]):
            lock=open(path,'a');locks.append(lock);until=time.monotonic()+15
            while True:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>until:raise RuntimeError('Execution lease is still held')
                    q.heartbeat(identity,lease);time.sleep(.1)
        with log_path.open('ab') as log:
            proc=subprocess.Popen([sys.executable,'-u','-m','backend.web_v2.worker_pool','child',identity,lease],env=env,cwd=spec['cwd'],
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        child_token=process_token(proc.pid);started=time.monotonic();beat=0
        while proc.poll() is None:
            now=time.monotonic()
            if now-beat>=1:q.heartbeat(identity,lease);beat=now
            current=q.get(identity)
            if current['lease']!=lease:raise LeaseLost('Task lease is no longer current')
            if current['cancel_requested']:
                status,message='cancelled','Cancelled';break
            if parent:
                p=q.get(parent)
                if p['status'] in TERMINAL or p['cancel_requested']:
                    status,message='cancelled','Parent stopped';break
            if now-started>spec['timeout']:
                status,message='failed','Execution time limit reached';break
            time.sleep(.1)
        else:
            code=proc.returncode
            status,message=('completed','Completed') if code==0 else ('failed','Process exited unsuccessfully')
        # Scientist's runner can stop normally at an input or pause checkpoint.
        if status=='completed' and row['metadata'].get('goal_folder') and (Path(row['metadata']['goal_folder'])/'goal.json').is_file():
            from agent.science_goal import GoalStore
            state=GoalStore(Path(row['metadata']['goal_folder'])).read()
            if state['status'] in {'paused','waiting_for_input','interrupted','cancelled'}:status=state['status'];message=status.replace('_',' ').capitalize()
            elif state['status'] != 'completed':status,message='failed','Research stopped: '+state['status'].replace('_',' ')
    except LeaseLost:
        return
    except Exception as exc:
        message='Task could not execute: '+type(exc).__name__
    finally:
        if proc:
            stop_process_tree(proc.pid,child_token)
            # Child may have exited first, leaving descendants in its group.
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        for lock in locks:lock.close()
        try:q.finish(identity,status,message,code,lease=lease)
        except LeaseLost:pass
        if row['metadata'].get('quota_scope'):
            from backend.web_v2.online_access import AccessStore
            AccessStore().release_science(row['metadata']['quota_scope'])


def child(identity, lease):
    q=JobQueue();q.attach_child(identity,lease,os.getpid())
    row=q.get(identity);spec=json.loads(row['dispatch'])
    if row['metadata'].get('capability_tool'):
        from mcp_tools.tools.capability_policy import CapabilityStore
        CapabilityStore().assert_call(row['metadata']['capability_tool'],row['metadata'].get('capability_arguments',{}))
    if row['metadata'].get('goal_folder'):
        from .science_goals import process_identity
        record=Path(row['metadata']['goal_folder'])/'worker.json'
        temp=record.with_suffix('.tmp');temp.write_text(json.dumps(process_identity(os.getpid())));os.replace(temp,record)
    command = list(spec['command'])
    # Persisted jobs from before the package move retain their original record.
    if '-m' in command:
        index = command.index('-m') + 1
        if index < len(command):
            for old, new in [('web_v2', 'backend.web_v2'), ('tool_runtime', 'mcp_tools.scheduler'), ('mcp_tools.tool_runtime', 'mcp_tools.scheduler')]:
                if command[index] == old or command[index].startswith(old + '.'):
                    command[index] = new + command[index][len(old):]
                    break
    os.execvpe(command[0],command,os.environ.copy())


def work(namespace,family):
    q=JobQueue()
    stopping=False
    def stop(*_):
        nonlocal stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    while not stopping:
        row=q.claim_next(namespace,family)
        if row:execute(q,row)
        else:time.sleep(.2)


def stop_workers(workers):
    for proc in workers.values():
        if proc.poll() is None:proc.terminate()
    deadline=time.monotonic()+5
    for proc in workers.values():
        try:proc.wait(timeout=max(.01,deadline-time.monotonic()))
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    workers.clear()


def idle_timeout():
    if os.getenv('VENUS_WORKER_AUTOSTART','1')!='1':return 0.
    try:value=float(os.getenv('VENUS_WORKER_IDLE_TIMEOUT','60'))
    except ValueError:return 60.
    return max(1.,value) if math.isfinite(value) and value>0 else 60.


def retire_idle_pool(q, namespace, workers, supervisor_guard=None):
    with pool_guard(q, namespace) as folder:
        with q.connection() as c:
            pending=c.execute("SELECT COUNT(*) FROM jobs WHERE namespace=? AND status IN ('queued','running')",(namespace,)).fetchone()[0]
        if pending:return False
        stop_workers(workers)
        record=folder/'supervisor.json'
        try:
            if json.loads(record.read_text())['pid']==os.getpid():record.unlink()
        except (FileNotFoundError,ValueError,KeyError):pass
        if supervisor_guard is not None:
            fcntl.flock(supervisor_guard,fcntl.LOCK_UN)
        return True


def supervise(namespace):
    q=JobQueue();folder=q.root/'pools'/namespace
    with (folder/'supervisor.lock').open('a') as guard:
        try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        workers={};stopping=False;idle_since=time.monotonic()
        def stop(*_):
            nonlocal stopping
            stopping=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        limits={'research':2,'cpu':2,'gpu':1,'api':4}
        try:
            while not stopping:
                for family,default in limits.items():
                    count=max(1,min(16,int(os.getenv('VENUS_WORKERS_'+family.upper(),str(default)))))
                    for slot in range(count):
                        key=(family,slot);old=workers.get(key)
                        if old is None or old.poll() is not None:
                            workers[key]=subprocess.Popen([sys.executable,'-u','-m','backend.web_v2.worker_pool','work',namespace,family],stdin=subprocess.DEVNULL)
                with q.connection(True) as c:
                    q._recover(c)
                    pending=c.execute("SELECT COUNT(*) FROM jobs WHERE namespace=? AND status IN ('queued','running')",(namespace,)).fetchone()[0]
                if pending:idle_since=time.monotonic()
                elif (timeout:=idle_timeout()) and time.monotonic()-idle_since>=timeout and retire_idle_pool(q,namespace,workers,guard):break
                time.sleep(.5)
        finally:
            stop_workers(workers)
            with q.connection(True) as c:q._recover(c)


if __name__=='__main__':
    if sys.argv[1]=='supervise':supervise(sys.argv[2])
    elif sys.argv[1]=='work':work(sys.argv[2],sys.argv[3])
    elif sys.argv[1]=='child':child(sys.argv[2],sys.argv[3])
