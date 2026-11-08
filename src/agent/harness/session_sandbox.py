"""Fail-closed local execution for session-authored Python and shell commands."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import signal
import stat
import sys
import time
import uuid
import tempfile
import json

from .sandbox_resources import admission, assert_active_job, memory_headroom


class SandboxUnavailable(RuntimeError):
    pass


def backend():
    name = os.getenv('VENUS_SESSION_SANDBOX', 'bwrap').strip().lower()
    if name not in {'bwrap', 'e2b', 'cubesandbox'}:
        raise SandboxUnavailable('Configure an isolated session backend; host execution is unavailable')
    if name == 'bwrap' and not shutil.which('bwrap'):
        raise SandboxUnavailable('Session sandbox is unavailable')
    return name


def check_workspace(workspace):
    """Bound traversal and storage, reject links before mounting into a sandbox."""
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir(): raise ValueError('Invalid session workspace')
    limit = (200 if os.getenv('VENUS_SCIENTIST_ONLINE') == '1' or os.getenv('WEBUI_V2_MODE') == 'online' else 1024) * 1024**2
    total = count = 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            p = Path(folder)/name
            try:
                info = p.lstat()
            except FileNotFoundError:
                # Atomic saves and temporary files can disappear after os.walk.
                # Never follow a replacement link or reuse metadata from another stat.
                continue
            if stat.S_ISLNK(info.st_mode): raise ValueError('Linked files are unavailable in session execution')
            regular = stat.S_ISREG(info.st_mode)
            if not regular and not stat.S_ISDIR(info.st_mode): raise ValueError('Only regular session files are supported')
            if regular and info.st_nlink != 1: raise ValueError('Hard-linked files are unavailable in session execution')
            count += 1
            if count > 10000: raise ValueError('Session file limit reached')
            if regular:
                total += info.st_size
                if total > limit: raise ValueError('Session storage limit reached')


def validate_execution(kind, payload, timeout):
    if kind not in {'python', 'shell'}: raise ValueError('Unsupported session execution')
    if type(timeout) is not int or not 1 <= timeout <= 300: raise ValueError('Timeout must be 1..300 seconds')
    if not isinstance(payload,str) or not payload or len(payload.encode()) > 200000: raise ValueError('Invalid command')
    if kind == 'python' and Path(payload).suffix.lower() != '.py':
        raise ValueError('Pass a saved .py script to run_session_python. Read data with read_session_file, or write a Python script that loads the data file.')



def command(workspace, kind, payload, timeout, *, scratch, memory):
    validate_execution(kind, payload, timeout)
    cmd = [shutil.which('bwrap') or 'bwrap', '--unshare-all', '--new-session', '--die-with-parent', '--as-pid-1', '--hostname', 'session', '--cap-drop', 'ALL']
    for p in ['/usr/bin','/usr/lib','/usr/lib64','/usr/share/zoneinfo','/usr/share/fonts','/bin','/lib','/lib64']:
        if Path(p).exists(): cmd += ['--ro-bind',p,p]
    cmd += ['--proc','/proc','--dev','/dev','--bind',str(scratch),'/tmp','--dir','/home','--dir','/opt',
            '--dir','/opt/venus-runtime/bin']
    # Mount code and shared libraries, never the environment's configuration/metadata.
    for name in ['lib','lib64']:
        p = Path(sys.prefix)/name
        if p.exists(): cmd += ['--ro-bind',str(p),'/opt/venus-runtime/'+name]
    executable = str(Path(sys.executable).resolve())
    for name in ['python','python3',f'python{sys.version_info.major}.{sys.version_info.minor}']:
        cmd += ['--ro-bind',executable,'/opt/venus-runtime/bin/'+name]
    cmd += ['--bind',str(Path(workspace).resolve()),'/workspace',
            '--ro-bind',str(Path(__file__).with_name('session_command_worker.py')),'/runner.py',
            '--ro-bind',str(Path(__file__).with_name('session_seccomp.py')),'/sandbox_seccomp.py',
            '--remount-ro','/','--remount-ro','/proc','--remount-ro','/dev',
            '--chdir','/workspace','--clearenv']
    env = {'HOME':'/tmp','PATH':'/opt/venus-runtime/bin:/usr/bin:/bin','LANG':'C.UTF-8',
           'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
           'VENUS_SANDBOX_MEMORY_BYTES':str(memory),'XDG_CACHE_HOME':'/tmp/cache',
           'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1'}
    if os.getenv('VENUS_SCIENTIST_ONLINE') == '1' or os.getenv('WEBUI_V2_MODE') == 'online': env['VENUS_SCIENTIST_ONLINE']='1'
    for k,v in env.items(): cmd += ['--setenv',k,v]
    return cmd+['--','/opt/venus-runtime/bin/python','-I','-u','/runner.py',kind,str(timeout),payload]


def tail(path, limit=12000):
    with Path(path).open('rb') as f:
        f.seek(max(0,Path(path).stat().st_size-limit));return f.read(limit).decode(errors='replace')


async def execute(workspace, control, kind, payload, timeout=120, *, on_progress=None):
    validate_execution(kind, payload, timeout)
    if os.getenv('VENUS_SCIENTIST_ONLINE') == '1' or os.getenv('WEBUI_V2_MODE') == 'online': timeout=min(timeout,60)
    assert_active_job()
    selected=backend()
    check_workspace(workspace)
    if selected == 'cubesandbox':
        from .session_cube import execute as cube_execute
        return await cube_execute(workspace,control,kind,payload,timeout,on_progress=on_progress)
    if selected == 'e2b':
        from .session_e2b import execute as e2b_execute
        return await e2b_execute(workspace,control,kind,payload,timeout,on_progress=on_progress)
    async with admission(workspace,on_progress=on_progress) as (store,identity,limits):
        return await _execute_bwrap(workspace,control,kind,payload,timeout,store,identity,limits,on_progress)


async def _settle(task, *, propagate=False):
    """Resource cleanup must finish even if cancellation arrives repeatedly."""
    cancelled=False
    while not task.done():
        try:await asyncio.shield(task)
        except asyncio.CancelledError:cancelled=True
    result=task.result()
    if cancelled and propagate:raise asyncio.CancelledError
    return result


async def _execute_bwrap(workspace, control, kind, payload, timeout, store, identity, limits, on_progress):
    from backend.web_v2.job_queue import QueueCancelled
    if Path(control).resolve().is_relative_to(Path(workspace).resolve()):
        raise ValueError('Sandbox control must be outside the session mount')
    control=Path(control);control.mkdir(parents=True,exist_ok=True)
    out=control/(identity+'.out');err=control/(identity+'.err')
    proc=None;waiting=None;pumps=[];started=time.monotonic();status='failed'
    scratch=tempfile.TemporaryDirectory(prefix='sandbox-',dir=control)
    def receipt():
        target=control/(identity+'.json')
        target.write_text(json.dumps({'run_id':identity,'backend':'bwrap','status':status,
            'elapsed_seconds':round(time.monotonic()-started,3),'memory_limit_bytes':limits.memory,
            'process_policy':'single-process-with-threads','temporary_files_removed':not Path(scratch.name).exists()}))
        target.chmod(0o600)
    output_bytes=0;output_exceeded=False
    async def drain(reader, target):
        nonlocal output_bytes, output_exceeded
        while chunk := await reader.read(65536):
            remaining=max(0,2*1024**2-output_bytes)
            target.write(chunk[:remaining]);target.flush()
            output_bytes+=len(chunk)
            if output_bytes>2*1024**2: output_exceeded=True
            await asyncio.sleep(0)
    async def reap():
        if proc is not None:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            await proc.wait()
        if waiting is not None:await waiting
        await asyncio.gather(*pumps,return_exceptions=True)
    try:
        store.scratch(identity,scratch.name)
        argv=command(workspace,kind,payload,timeout,scratch=scratch.name,memory=limits.memory)
        with out.open('wb') as stdout, err.open('wb') as stderr:
            # Clearing only the guest environment leaves the bwrap supervisor's
            # inherited credentials readable through procfs. Sanitize at launch.
            assert_active_job()
            spawning=asyncio.create_task(asyncio.create_subprocess_exec(*argv,stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,
                env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'},close_fds=True,start_new_session=True))
            try:
                try:proc=await asyncio.shield(spawning)
                finally:
                    # A cancellation during OS spawn must not orphan a child.
                    if proc is None:proc=await _settle(spawning)
                    pumps=[asyncio.create_task(drain(proc.stdout,stdout)),asyncio.create_task(drain(proc.stderr,stderr))]
                    waiting=asyncio.create_task(proc.wait())
                store.child(identity,proc.pid)
                status='running';receipt()
                async with asyncio.timeout(max(.01,timeout-(time.monotonic()-started))):
                    while not waiting.done():
                        assert_active_job()
                        if output_exceeded: raise ValueError('Session output limit reached')
                        check_workspace(workspace)
                        check_workspace(scratch.name)
                        if memory_headroom()<limits.reserve_free//2:raise MemoryError('Host memory reserve reached')
                        if min(shutil.disk_usage(workspace).free,shutil.disk_usage(scratch.name).free)<limits.disk_free:
                            raise ValueError('Host disk reserve reached')
                        if on_progress:on_progress({'status':'running','tool':'run_research_python' if kind=='python' else 'run_session_shell',
                            'script':payload if kind=='python' else '', 'elapsed_seconds':round(time.monotonic()-started,1),
                            'stdout_tail':tail(out),'stderr_tail':tail(err,6000)})
                        await asyncio.wait({waiting},timeout=.25)
                    await asyncio.gather(*pumps)
                if output_exceeded: raise ValueError('Session output limit reached')
                check_workspace(workspace)
                check_workspace(scratch.name)
            finally:
                await _settle(asyncio.create_task(reap()),propagate=True)
        status='completed' if proc.returncode==0 else 'failed'
        result={'run_id':identity,'backend':'bwrap','exit_code':proc.returncode,'stdout':tail(out),'stderr':tail(err,6000)}
        if on_progress:on_progress({**result,'status':'completed' if proc.returncode==0 else 'failed','stdout_tail':result['stdout'],'stderr_tail':result['stderr']})
        return result
    except (asyncio.CancelledError, QueueCancelled):
        status='cancelled';raise
    except TimeoutError:
        status='timeout';raise
    except BaseException:
        status='failed';raise
    finally:
        scratch.cleanup()
        receipt()


async def run_shell(workspace, control, command_text, timeout=60):
    from backend.web_v2.job_queue import resource_slot
    async with resource_slot('cpu',kind='session:shell'):
        return await execute(workspace,control,'shell',command_text,timeout)
