"""Stream already-authorized file descriptors; never reopen untrusted paths."""
import asyncio
from urllib.parse import quote
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

# Applies even when a user opens a file URL directly, outside our sandboxed iframe.
ARTIFACT_HEADERS = {
    'Cache-Control': 'private, no-store',
    'Vary': 'Cookie',
    'X-Content-Type-Options': 'nosniff',
    'Referrer-Policy': 'no-referrer',
    'Content-Security-Policy': "sandbox; default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
}


def download(stream, name, *, media_type='application/octet-stream', headers=None):
    async def body():
        try:
            while data := await asyncio.to_thread(stream.read, 65536):
                yield data
        finally:
            stream.close()
    return StreamingResponse(body(), media_type=media_type,
        headers={**ARTIFACT_HEADERS, **(headers or {}), 'Content-Disposition': "attachment; filename*=UTF-8''" + quote(name, safe='')},
        background=BackgroundTask(stream.close))
