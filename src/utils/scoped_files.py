"""Descriptor-anchored access to untrusted files inside a server-owned root.

Never validate a pathname and then reopen it: a running sandbox can replace
any path component between those operations. All reads and writes stay anchored
to directory descriptors, with links and special files refused at open time.
"""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import hashlib
import os
import stat
import uuid

def parts(relative):
    if not isinstance(relative, str) or not relative or '\\' in relative or '\0' in relative:
        raise ValueError('Use a relative workspace filename')
    value = PurePosixPath(relative)
    if value.is_absolute() or not value.parts or any(p.startswith('.') or ':' in p for p in value.parts):
        raise ValueError('Use a relative workspace filename')
    return value.parts


@contextmanager
def parent_fd(root, relative, *, create=False):
    names = parts(relative)
    if not hasattr(os, 'O_NOFOLLOW') or os.open not in os.supports_dir_fd:
        raise RuntimeError('Secure session file access is unavailable on this host')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(Path(root), flags)
    try:
        for name in names[:-1]:
            if create:
                try: os.mkdir(name, mode=0o700, dir_fd=fd)
                except FileExistsError: pass
            child = os.open(name, flags, dir_fd=fd)
            os.close(fd); fd = child
        yield fd, names[-1]
    finally:
        os.close(fd)

def _open_at(directory, name, *, max_bytes=None):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Regular session files without hard links are required')
        if max_bytes is not None and info.st_size > max_bytes:
            raise ValueError('Session file is too large')
        return os.fdopen(fd, 'rb')
    except BaseException:
        os.close(fd)
        raise


def open_file(root, relative, *, max_bytes=None):
    with parent_fd(root, relative) as (directory, name):
        return _open_at(directory, name, max_bytes=max_bytes)

def read_bytes(root, relative, *, max_bytes=100*1024**2):
    with open_file(root, relative, max_bytes=max_bytes) as stream:
        data = stream.read(max_bytes+1)
        if len(data) > max_bytes: raise ValueError('Session file is too large')
        return data

def open_absolute(path, *, max_bytes=None):
    """For an already-authorized local Library path; refuse links in any ancestor."""
    value = Path(path)
    if not value.is_absolute(): raise ValueError('An authorized absolute path is required')
    return open_file(value.anchor, value.relative_to(value.anchor).as_posix(), max_bytes=max_bytes)


@contextmanager
def atomic_writer(root, relative, *, expected_sha256=None, max_bytes=100*1024**2):
    with parent_fd(root, relative, create=expected_sha256 is None) as (directory, name):
        # Serialize cooperating writers across API/worker processes. The lock
        # stays on the directory descriptor and is released when it closes.
        import fcntl
        fcntl.flock(directory, fcntl.LOCK_EX)
        if expected_sha256 is not None:
            with _open_at(directory, name, max_bytes=max_bytes) as current:
                data = current.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError('Session file is too large')
            if hashlib.sha256(data).hexdigest() != expected_sha256:
                raise ValueError('File changed since it was read; read it again before editing')
        try:
            info = os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            info = None
        if info and (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1):
            raise ValueError('Linked files are unavailable')
        temporary = '.write-' + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                yield stream
                stream.flush()
            os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
        finally:
            try: os.unlink(temporary, dir_fd=directory)
            except FileNotFoundError: pass

def write_bytes(root, relative, data):
    with atomic_writer(root, relative) as stream:
        stream.write(data)


def replace_text(root, relative, old_text, new_text, expected_sha256, *, max_bytes=200000):
    """One exact UTF-8 replacement, conditional on the version that was read."""
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('Use the SHA256 returned by read_session_file')
    if not isinstance(old_text, str) or not old_text or not isinstance(new_text, str):
        raise ValueError('Provide nonempty old_text and replacement new_text')
    if old_text == new_text:
        raise ValueError('The replacement is unchanged')
    raw = read_bytes(root, relative, max_bytes=max_bytes)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('File changed since it was read; read it again before editing')
    if b'\x00' in raw:
        raise ValueError('Only UTF-8 text files can be edited')
    try:
        text = raw.decode('utf-8')
    except UnicodeError:
        raise ValueError('Only UTF-8 text files can be edited') from None
    position = text.find(old_text)
    if position < 0:
        raise ValueError('old_text was not found; read the current file before editing')
    if text.find(old_text, position + 1) >= 0:
        raise ValueError('old_text is ambiguous; include enough surrounding text for one exact match')
    data = (text[:position] + new_text + text[position + len(old_text):]).encode('utf-8')
    if len(data) > max_bytes:
        raise ValueError('Edited file is too large')
    with atomic_writer(root, relative, expected_sha256=expected_sha256, max_bytes=max_bytes) as stream:
        stream.write(data)
    return {'path': relative, 'before_sha256': expected_sha256,
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data), 'replacements': 1}

def copy_file(source_root, source_name, target_root, target_name, *, max_bytes=100*1024**2):
    with open_file(source_root, source_name, max_bytes=max_bytes) as src, atomic_writer(target_root, target_name) as dst:
        total = 0
        while data := src.read(1024*1024):
            total += len(data)
            if total > max_bytes: raise ValueError('Session file is too large')
            dst.write(data)
    return total

def chunks(stream, *, max_bytes=None):
    """Consume the already authorized descriptor, including bounded ZIP inputs."""
    total = 0
    try:
        while data := stream.read(65536):
            total += len(data)
            if max_bytes is not None and total > max_bytes: raise ValueError('Session file is too large')
            yield data
    finally:
        stream.close()
