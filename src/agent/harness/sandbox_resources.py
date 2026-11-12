"""Host-wide admission for ephemeral code sandboxes, independent of SSE clients.

The shared lock contains only host-owned reservations. It is never guest-mounted.
The existing job queue remains responsible for user task scheduling.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

MIB = 1024**2
ROOT = Path(__file__).resolve().parents[3]


def positive(name, default, low=1, high=65536):
    value = int(os.getenv(name, str(default)))
    if not low <= value <= high:
        raise ValueError(f'{name} must be {low}..{high}')
    return value


def online():
    return os.getenv('VENUS_SCIENTIST_ONLINE') == '1' or os.getenv('WEBUI_V2_MODE') == 'online'


@dataclass(frozen=True)
class Limits:
    memory: int
    parallel: int
    reserved: int
    reserve_free: int
    disk_free: int
    wait_seconds: int

    @classmethod
    def load(cls):
        mode = 'ONLINE' if online() else 'LOCAL'
        return cls(
            positive('VENUS_SANDBOX_'+mode+'_MEMORY_MB', 2048 if online() else 8192, 128)*MIB,
            positive('VENUS_SANDBOX_MAX_PARALLEL', 2, 1, 16),
            positive('VENUS_SANDBOX_TOTAL_MEMORY_MB', 16384, 128, 262144)*MIB,
            positive('VENUS_SANDBOX_MEMORY_RESERVE_MB', 4096, 256)*MIB,
            positive('VENUS_SANDBOX_DISK_RESERVE_MB', 2048, 256)*MIB,
            positive('VENUS_SANDBOX_WAIT_SECONDS', 60, 1, 300),
        )


def cgroup_headroom(folder, limit, used):
    """Available cgroup memory, excluding reclaimable clean file cache.

    memory.current/usage_in_bytes include reclaimable page cache. Counting that
    as resident application memory can permanently starve every sandbox after
    downloading weights. Clean file-backed pages on either LRU can be reclaimed by the kernel.
    Dirty pages, tmpfs, unevictable memory and anonymous RSS remain charged.
    Missing/malformed statistics retain the conservative raw-usage calculation.
    """
    ceiling, usage = int(limit), max(0, int(used))
    reclaimable = 0
    try:
        stats = {key: max(0, int(value)) for key, value in
                 (line.split() for line in (folder / 'memory.stat').read_text().splitlines())}
        prefix = 'total_' if 'total_inactive_file' in stats else ''
        file_lru = stats.get(prefix + 'inactive_file', 0) + stats.get(prefix + 'active_file', 0)
        cache = stats.get(prefix + 'cache', stats.get('file', 0))
        shared = stats.get(prefix + 'shmem', 0)
        dirty = stats.get(prefix + 'dirty', stats.get('file_dirty', 0))
        writeback = stats.get(prefix + 'writeback', stats.get('file_writeback', 0))
        reclaimable = max(0, min(file_lru, max(0, cache - shared)) - dirty - writeback)
    except (OSError, ValueError):
        pass
    return max(0, ceiling - usage + min(usage, reclaimable))


def cgroup_memory_files():
    """Walk every applicable memory controller, including tighter parent limits."""
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _, controllers, location = line.split(':', 2)
        if 'memory' in controllers.split(','):
            base=Path('/sys/fs/cgroup/memory'); names=('memory.limit_in_bytes','memory.usage_in_bytes')
        elif controllers == '':
            base=Path('/sys/fs/cgroup'); names=('memory.max','memory.current')
        else:
            continue
        current=base/location.lstrip('/')
        if not current.is_relative_to(base):continue
        for folder in [current,*current.parents]:
            if not folder.is_relative_to(base):break
            limit,used=folder/names[0],folder/names[1]
            if limit.exists() and used.exists():
                raw=limit.read_text().strip()
                if raw!='max':yield folder, raw, used


def memory_ceiling():
    """Total host/cgroup allocation, distinct from transient memory pressure."""
    values = [int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith('MemTotal:')]
    values.extend(int(raw) for _, raw, _ in cgroup_memory_files())
    if not values:raise RuntimeError('Memory allocation accounting unavailable')
    return min(values)


def validate_capacity(limits):
    if limits.memory>limits.reserved:
        raise ValueError('Sandbox memory exceeds the shared budget')
    ceiling = memory_ceiling()
    if limits.memory + limits.reserve_free >= ceiling:
        mode = 'ONLINE' if online() else 'LOCAL'
        raise ValueError(
            f'Sandbox configuration cannot fit this compute allocation ({ceiling//MIB} MiB): '
            f'the sandbox needs {limits.memory//MIB} MiB plus {limits.reserve_free//MIB} MiB of host reserve. '
            'Request a larger allocation or configure '
            f'VENUS_SANDBOX_{mode}_MEMORY_MB for this workload; '
            'the host reserve is not reduced automatically.')


def memory_headroom():
    """Minimum of host available memory and the process's actual cgroup budget."""
    values = [int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
              if line.startswith('MemAvailable:')]
    values.extend(cgroup_headroom(folder, raw, used.read_text()) for folder, raw, used in cgroup_memory_files())
    if not values:raise RuntimeError('Memory accounting unavailable')
    return min(values)


def assert_active_job():
    """Stopped/paused/expired parent tasks cannot acquire or retain a sandbox."""
    from backend.web_v2.job_queue import JobQueue, QueueCancelled, _held_slot
    identities={x for x in [_held_slot.get(),os.getenv('VENUS_QUEUE_JOB_ID')] if x}
    if not identities:return
    queue=JobQueue()
    for identity in identities:
        try:row=queue.get(identity)
        except FileNotFoundError:raise QueueCancelled('Task is no longer active') from None
        if row['status']!='running' or row['cancel_requested']:
            raise QueueCancelled('Task stopped')
        parent=row['metadata'].get('parent_job')
        if parent:
            try:ancestor=queue.get(parent)
            except FileNotFoundError:raise QueueCancelled('Parent task is no longer active') from None
            if ancestor['status']!='running' or ancestor['cancel_requested']:
                raise QueueCancelled('Parent task stopped')


class Reservations:
    def __init__(self, root=None):
        self.root=Path(root or os.getenv('VENUS_SANDBOX_CONTROL_ROOT') or ROOT/'runtime/control/sandboxes')
        self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        if self.root.is_symlink() or self.root.stat().st_uid!=os.getuid():
            raise ValueError('Invalid sandbox admission directory')
        self.root.chmod(0o700)
        self.path=self.root/'reservations.json'

    @contextmanager
    def locked(self):
        from backend.web_v2.job_queue import process_token, stop_process_tree
        descriptor=os.open(self.root/'lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        with os.fdopen(descriptor,'r+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if self.path.is_symlink():raise ValueError('Invalid reservation file')
            rows=json.loads(self.path.read_text()) if self.path.exists() else []
            current=[]
            for row in rows:
                if process_token(row['pid'])==row['token']:
                    current.append(row)
                else:
                    if row.get('child_pid'):stop_process_tree(row['child_pid'],row.get('child_token'))
                    scratch=row.get('scratch')
                    if scratch:
                        path=Path(scratch['path'])
                        if path.exists() and not path.is_symlink() and path.name.startswith('sandbox-'):
                            stat=path.stat()
                            if stat.st_uid==os.getuid() and (stat.st_dev,stat.st_ino)==(scratch['device'],scratch['inode']):
                                shutil.rmtree(path)
            yield current
            temp=self.root/('reservations.'+uuid.uuid4().hex+'.tmp')
            temp.write_text(json.dumps(current));temp.chmod(0o600);os.replace(temp,self.path)

    def acquire(self, workspace, limits):
        from backend.web_v2.job_queue import process_token
        identity=uuid.uuid4().hex
        owner=hashlib.sha256((os.getenv('VENUS_QUEUE_OWNER') or str(Path(workspace).resolve())).encode()).hexdigest()
        workspace_id=hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
        with self.locked() as rows:
            parallel=min([limits.parallel,*[r.get('parallel',limits.parallel) for r in rows]])
            budget=min([limits.reserved,*[r.get('reserved',limits.reserved) for r in rows]])
            if len(rows)>=parallel or any(r['owner']==owner or r['workspace']==workspace_id for r in rows):return None
            reserved=sum(r['memory'] for r in rows)
            # Conservative: count reservations even if some are already in RSS.
            if reserved+limits.memory>budget:return None
            if memory_headroom()<reserved+limits.memory+limits.reserve_free:return None
            if min(shutil.disk_usage(workspace).free,shutil.disk_usage(self.root).free)<limits.disk_free:return None
            token=process_token(os.getpid())
            if not token:raise RuntimeError('Sandbox worker identity unavailable')
            rows.append({'id':identity,'owner':owner,'workspace':workspace_id,'pid':os.getpid(),'token':token,
                         'memory':limits.memory,'parallel':limits.parallel,'reserved':limits.reserved,
                         'created':time.time(),'mode':'online' if online() else 'local'})
        return identity

    def child(self, identity, pid):
        from backend.web_v2.job_queue import process_token
        with self.locked() as rows:
            row=next(r for r in rows if r['id']==identity)
            row.update(child_pid=pid,child_token=process_token(pid))

    def scratch(self, identity, path):
        path=Path(path).resolve();stat=path.stat()
        with self.locked() as rows:
            row=next(r for r in rows if r['id']==identity)
            row['scratch']={'path':str(path),'device':stat.st_dev,'inode':stat.st_ino}

    def release(self, identity):
        with self.locked() as rows:rows[:]=[r for r in rows if r['id']!=identity]


@asynccontextmanager
async def admission(workspace, *, on_progress=None):
    limits=Limits.load()
    validate_capacity(limits)
    store=Reservations();identity=None;started=time.monotonic()
    try:
        while identity is None:
            assert_active_job()
            identity=store.acquire(workspace,limits)
            if identity is not None:break
            if time.monotonic()-started>=limits.wait_seconds:
                raise TimeoutError('Sandbox resources are busy; try again later')
            if on_progress:on_progress({'status':'queued','tool':'session_sandbox','elapsed_seconds':0,'stdout_tail':'','stderr_tail':''})
            await asyncio.sleep(.2)
        assert_active_job()
        yield store,identity,limits
    finally:
        if identity is not None:store.release(identity)
