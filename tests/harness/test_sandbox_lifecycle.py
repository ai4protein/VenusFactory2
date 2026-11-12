"""Real, bounded adversarial probes: no fork bomb or host memory exhaustion."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from agent.harness import sandbox_resources as resources
from agent.harness.session_sandbox import execute


@pytest.fixture(autouse=True)
def isolated_budget(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SANDBOX_CONTROL_ROOT',str(tmp_path/'admission'))
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    monkeypatch.setenv('VENUS_SANDBOX_LOCAL_MEMORY_MB','512')
    monkeypatch.setenv('VENUS_SANDBOX_ONLINE_MEMORY_MB','512')
    monkeypatch.setenv('VENUS_SANDBOX_TOTAL_MEMORY_MB','1024')
    monkeypatch.setenv('VENUS_SANDBOX_MAX_PARALLEL','1')
    monkeypatch.setenv('VENUS_SANDBOX_WAIT_SECONDS','3')
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID',raising=False)
    monkeypatch.delenv('VENUS_QUEUE_OWNER',raising=False)


def script(tmp_path,text,name='workspace'):
    work=tmp_path/name;work.mkdir();(work/'run.py').write_text(text);return work


def empty_reservations():
    with resources.Reservations().locked() as rows:assert rows==[]


@pytest.mark.asyncio
async def test_forbidden_syscalls_and_memory_limit_are_kernel_enforced(tmp_path):
    work=script(tmp_path,'''import errno,json,mmap,os,resource,threading
from pathlib import Path
finished=[]
thread=threading.Thread(target=lambda:finished.append(True));thread.start();thread.join()
assert finished==[True]
for operation in [os.fork,lambda:os.memfd_create('ram'),lambda:os.sched_setaffinity(0,{0})]:
    try:operation()
    except OSError as error:assert error.errno in (errno.EPERM,errno.ENOSYS)
    else:raise AssertionError('restricted syscall succeeded')
limit=resource.getrlimit(resource.RLIMIT_AS)[1]
try:mmap.mmap(-1,limit*2)
except OSError as error:assert error.errno==errno.ENOMEM
else:raise AssertionError('memory limit bypass')
try:resource.setrlimit(resource.RLIMIT_AS,(limit*2,limit*2))
except (ValueError,PermissionError):pass
else:raise AssertionError('raised hard limit')
for path in ['/escape','/dev/memory','/home/private']:
    try:Path(path).write_text('blocked')
    except OSError:pass
    else:raise AssertionError('writable root')
Path('/tmp/temporary').write_text('removed later')
Path('result.json').write_text(json.dumps({'isolated':True,'memory':limit}))
print('bounded')
''')
    result=await execute(work,tmp_path/'control','python','run.py',10)
    assert result['exit_code']==0,result
    assert json.loads((work/'result.json').read_text())['memory']==512*1024**2
    assert not list((tmp_path/'control').glob('sandbox-*'))
    record=json.loads((tmp_path/'control'/(result['run_id']+'.json')).read_text())
    assert record['temporary_files_removed'] and record['status']=='completed'
    empty_reservations()


@pytest.mark.asyncio
async def test_waiting_allocates_no_sandbox_and_cancellation_releases_slot(tmp_path):
    work=script(tmp_path,"import time\nopen('ready','w').close()\ntime.sleep(20)")
    second=script(tmp_path,"open('ran','w').close()",'second')
    first=asyncio.create_task(execute(work,tmp_path/'a','python','run.py',20))
    for _ in range(100):
        if (work/'ready').exists():break
        await asyncio.sleep(.05)
    assert (work/'ready').exists()
    waiting=asyncio.create_task(execute(second,tmp_path/'b','python','run.py',10))
    await asyncio.sleep(.4)
    assert not (tmp_path/'b').exists() and not (second/'ran').exists()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):await waiting
    first.cancel()
    with pytest.raises(asyncio.CancelledError):await first
    empty_reservations()
    result=await execute(second,tmp_path/'b','python','run.py',10)
    assert result['exit_code']==0 and (second/'ran').exists()


@pytest.mark.asyncio
async def test_cancel_during_spawn_reaps_child(tmp_path,monkeypatch):
    from agent.harness import session_sandbox as sandbox
    original=asyncio.create_subprocess_exec;created=asyncio.Event();continue_spawn=asyncio.Event();children=[]
    async def delayed(*args,**kwargs):
        child=await original(*args,**kwargs);children.append(child);created.set()
        await continue_spawn.wait();return child
    monkeypatch.setattr(sandbox.asyncio,'create_subprocess_exec',delayed)
    work=script(tmp_path,'import time\ntime.sleep(20)')
    pending=asyncio.create_task(execute(work,tmp_path/'control','python','run.py',20))
    await asyncio.wait_for(created.wait(),5);pending.cancel();continue_spawn.set()
    with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(pending,10)
    assert children[0].returncode is not None
    assert not list((tmp_path/'control').glob('sandbox-*'))
    empty_reservations()


@pytest.mark.asyncio
async def test_stop_queued_parent_never_starts_guest(tmp_path,monkeypatch):
    from backend.web_v2.job_queue import JobQueue, QueueCancelled
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    queue=JobQueue();identity=queue.submit('owner','chat','research');queue.cancel(identity)
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID',identity)
    work=script(tmp_path,"open('ran','w').close()")
    with pytest.raises(QueueCancelled):await execute(work,tmp_path/'control','python','run.py',5)
    assert not (work/'ran').exists() and not (tmp_path/'control').exists()


@pytest.mark.asyncio
async def test_parent_stop_interrupts_running_guest_without_sse(tmp_path,monkeypatch):
    from backend.web_v2.job_queue import JobQueue, QueueCancelled
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    queue=JobQueue();identity=queue.submit('owner','chat','research');queue.bind(identity);assert queue.claim(identity)
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID',identity)
    work=script(tmp_path,"import time\nopen('ready','w').close()\ntime.sleep(5)\nopen('late','w').close()")
    pending=asyncio.create_task(execute(work,tmp_path/'control','python','run.py',10))
    for _ in range(100):
        if (work/'ready').exists():break
        await asyncio.sleep(.05)
    assert (work/'ready').exists();queue.cancel(identity)
    with pytest.raises(QueueCancelled):await asyncio.wait_for(pending,3)
    assert not (work/'late').exists();empty_reservations()


def test_reservations_are_shared_between_processes_and_modes(tmp_path,monkeypatch):
    work=tmp_path/'w';work.mkdir();store=resources.Reservations();limits=resources.Limits.load()
    first=store.acquire(work,limits);assert first
    other=tmp_path/'other';other.mkdir()
    probe="from agent.harness.sandbox_resources import *;import sys;print(Reservations().acquire(sys.argv[1],Limits.load()))"
    result=subprocess.run([sys.executable,'-c',probe,str(other)],env={**os.environ,'WEBUI_V2_MODE':'online'},capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='None'
    store.release(first);empty_reservations()


@pytest.mark.asyncio
async def test_memory_pressure_rejects_before_spawn(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SANDBOX_WAIT_SECONDS','1')
    monkeypatch.setattr(resources,'memory_headroom',lambda:0)
    work=script(tmp_path,"open('ran','w').close()")
    with pytest.raises(TimeoutError,match='resources'):
        await execute(work,tmp_path/'control','python','run.py',5)
    assert not (tmp_path/'control').exists() and not (work/'ran').exists()
    empty_reservations()


def test_worker_crash_destroys_guest_and_next_admission_reclaims_scratch(tmp_path):
    from backend.web_v2.job_queue import process_token
    work=script(tmp_path,"import time\nopen('ready','w').close()\ntime.sleep(2)\nopen('late','w').close()")
    control=tmp_path/'control'
    probe="import asyncio,sys;from agent.harness.session_sandbox import execute;asyncio.run(execute(sys.argv[1],sys.argv[2],'python','run.py',10))"
    child=subprocess.Popen([sys.executable,'-c',probe,str(work),str(control)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if (work/'ready').exists():break
            time.sleep(.05)
        assert (work/'ready').exists()
        with resources.Reservations().locked() as rows:
            assert len(rows)==1;pid=rows[0]['child_pid'];token=rows[0]['child_token']
        child.kill();child.wait(timeout=5)
        # Trigger the same orphan recovery used by the next admission.
        empty_reservations()
        for _ in range(50):
            if process_token(pid)!=token:break
            time.sleep(.05)
        assert process_token(pid)!=token
        assert not list(control.glob('sandbox-*'))
        time.sleep(2.1);assert not (work/'late').exists()
    finally:
        if child.poll() is None:child.kill();child.wait(timeout=5)


@pytest.mark.asyncio
async def test_impossible_allocation_fails_before_waiting_or_reserving(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_SANDBOX_LOCAL_MEMORY_MB', '8192')
    monkeypatch.setenv('VENUS_SANDBOX_TOTAL_MEMORY_MB', '16384')
    monkeypatch.setenv('VENUS_SANDBOX_MEMORY_RESERVE_MB', '4096')
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    monkeypatch.delenv('VENUS_SCIENTIST_ONLINE', raising=False)
    monkeypatch.setattr(resources, 'memory_ceiling', lambda: 8192 * resources.MIB)
    def forbidden(*args, **kwargs):
        raise AssertionError('Impossible allocations must not wait for a reservation')
    monkeypatch.setattr(resources.Reservations, 'acquire', forbidden)
    work = script(tmp_path, "open('ran','w').close()")
    with pytest.raises(ValueError, match='configuration cannot fit.*8192 MiB'):
        await execute(work, tmp_path/'control', 'python', 'run.py', 5)
    assert not (work/'ran').exists() and not (tmp_path/'control').exists()
    assert not (tmp_path/'admission').exists()


@pytest.mark.parametrize('insufficient_gib', [10, 12])
def test_static_capacity_never_reduces_sandbox_or_host_reserve(monkeypatch, insufficient_gib):
    monkeypatch.setattr(resources, 'memory_ceiling', lambda: 16 * 1024 * resources.MIB)
    limits = resources.Limits(8*1024*resources.MIB, 2, 16*1024*resources.MIB,
                              4*1024*resources.MIB, 2*1024*resources.MIB, 60)
    resources.validate_capacity(limits)
    assert limits.memory == 8*1024*resources.MIB and limits.reserve_free == 4*1024*resources.MIB
    monkeypatch.setattr(resources, 'memory_ceiling', lambda: insufficient_gib * 1024 * resources.MIB)
    with pytest.raises(ValueError, match='host reserve is not reduced'):
        resources.validate_capacity(limits)


def test_memory_ceiling_honors_parent_limits_and_is_not_free_memory(tmp_path, monkeypatch):
    usage = tmp_path/'usage';usage.write_text('200000000')
    monkeypatch.setattr(resources, 'cgroup_memory_files', lambda: iter([
        (tmp_path, str(16*1024*resources.MIB), usage), (tmp_path, str(8*1024*resources.MIB), usage)]))
    original = Path.read_text
    def read(self, *args, **kwargs):
        if str(self) == '/proc/meminfo':
            return 'MemTotal: 67108864 kB\nMemAvailable: 262144 kB\n'
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert resources.memory_ceiling() == 8*1024*resources.MIB
    assert resources.memory_headroom() == 256*resources.MIB
