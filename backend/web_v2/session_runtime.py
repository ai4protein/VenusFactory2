"""Single-node, cross-process session coordination and stable signing secrets."""
import asyncio
import fcntl
import hashlib
import os
from pathlib import Path
import secrets


class SharedSessionLock:
    def __init__(self, root, session_id):
        self.path=Path(root)/'session-locks'/(hashlib.sha256(session_id.encode()).hexdigest()+'.lock')
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        self.fd=None

    def locked(self):
        if self.fd is not None:return True
        with self.path.open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return True
        return False

    async def acquire(self):
        while True:
            if self.fd is None:
                f=self.path.open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:f.close()
                else:self.fd=f;return True
            await asyncio.sleep(.05)

    def release(self):
        if self.fd is None:raise RuntimeError('Lock is not acquired')
        f=self.fd;self.fd=None;f.close()

    async def __aenter__(self):await self.acquire();return self
    async def __aexit__(self,*_):self.release()


def persistent_secret(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    with path.with_suffix('.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        if path.exists():return path.read_text().strip()
        value=secrets.token_hex(32)
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
        return value
