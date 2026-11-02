"""Shared upload policy: local files are unbounded; online files are at most 10 MiB."""
import asyncio
import os
from pathlib import Path
import tempfile

from fastapi import HTTPException

ONLINE_FILE_BYTES = 10 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


def upload_limit():
    return ONLINE_FILE_BYTES if os.getenv('WEBUI_V2_MODE', 'local').strip().lower() == 'online' else None


def validate_upload_size(size):
    limit = upload_limit()
    if limit is not None and size > limit:
        raise ValueError('Each file must be 10 MB or smaller in online mode')


def check_upload_size(size):
    try:
        validate_upload_size(size)
    except ValueError as exc:
        raise HTTPException(413, str(exc)) from None


def preflight_uploads(files):
    for file in files:
        if getattr(file, 'size', None) is not None:
            check_upload_size(file.size)


async def upload_chunks(file):
    preflight_uploads([file])
    total = 0
    while True:
        limit = upload_limit()
        count = min(CHUNK_BYTES, limit - total + 1) if limit is not None else CHUNK_BYTES
        chunk = await file.read(count)
        if not chunk:
            break
        total += len(chunk)
        check_upload_size(total)  # Count actual bytes even if the reported size is missing or wrong.
        yield chunk


async def read_upload(file):
    content = bytearray()
    async for chunk in upload_chunks(file):
        content.extend(chunk)
    return bytes(content)


async def save_upload(file, destination):
    """Write in chunks; never leave an oversized/partial file or damage a replacement."""
    preflight_uploads([file])
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.upload-', dir=destination.parent)
    temporary = Path(name)
    try:
        total = 0
        with os.fdopen(fd, 'wb') as stream:
            async for chunk in upload_chunks(file):
                await asyncio.to_thread(stream.write, chunk)
                total += len(chunk)
        os.replace(temporary, destination)
        return total
    finally:
        temporary.unlink(missing_ok=True)
