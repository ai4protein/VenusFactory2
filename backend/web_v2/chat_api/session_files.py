"""Authenticated previews and ZIP downloads for Agent and Scientist files."""
import asyncio
import mimetypes
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from backend.web_v2.chat_api._shared import _get_session_or_404, _assert_session_access
from backend.web_v2.science_goals import service
from backend.web_v2.research_files import session_roots, resolve_file, archive_files
from backend.web_v2.research_files import open_owned_file
from backend.web_v2.scoped_downloads import download
from pathlib import PurePosixPath

router = APIRouter()

async def roots(session_id, request):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    return session_roots(state, service, session_id)

@router.get('/sessions/{session_id}/files/content')
async def content(session_id: str, path: str, request: Request):
    owned = await roots(session_id, request)
    try:
        stream = open_owned_file(owned, path)
        file = PurePosixPath(path)
    except (ValueError, OSError):
        raise HTTPException(404, 'File not found')
    media = mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
    if file.suffix.lower() in {'.py','.r','.md','.txt','.log','.csv','.tsv','.pdb','.cif','.sdf','.mol','.mol2'}:
        media = 'text/plain; charset=utf-8'
    return download(stream, file.name, media_type=media, headers={'Cache-Control':'no-store', 'X-Content-Type-Options':'nosniff', 'Content-Security-Policy':"sandbox; default-src 'none'"})

class Selection(BaseModel):
    paths: list[str] = Field(min_length=1, max_length=200)

@router.post('/sessions/{session_id}/files/archive')
async def archive(session_id: str, payload: Selection, request: Request):
    owned = await roots(session_id, request)
    try:
        stream = await asyncio.to_thread(archive_files, owned, payload.paths)
    except (ValueError, OSError) as exc:
        raise HTTPException(400, 'Select up to 200 available files, 200 MB in total') from exc
    def chunks():
        try:
            while data := stream.read(1024*1024):
                yield data
        finally:
            stream.close()
    return StreamingResponse(chunks(), media_type='application/zip', headers={'Content-Disposition':'attachment; filename="research-files.zip"','Cache-Control':'no-store'})
