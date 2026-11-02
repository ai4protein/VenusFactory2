"""A Library of live, authenticated conversations; never a global disk listing."""
import asyncio
import json
import mimetypes
from pathlib import Path
import tempfile
from typing import Literal
import zipfile
import os
from utils.scoped_files import chunks
from backend.web_v2.scoped_downloads import download

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.web_v2.chat_api._shared import _get_session_or_404, _assert_session_access
from backend.web_v2.job_access import require_same_origin
from backend.web_v2.library_files import visible_file
from backend.web_v2.research_files import session_roots, list_files, resolve_file, open_owned_file
from backend.web_v2.research_jobs import ResearchJobs
from backend.web_v2.report_api import owner as report_owner
from backend.web_v2.science_goals import service

def library_origin(request: Request):
    require_same_origin(request)
    if os.environ.get("WEBUI_V2_MODE", "local").strip().lower() != "local":
        raise HTTPException(403, "Library is unavailable in online mode.")


router = APIRouter(prefix='/api/library', tags=['conversation library'], dependencies=[Depends(library_origin)])
PRIVATE = {'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff', 'Content-Security-Policy': "sandbox; default-src 'none'"}


class ConversationRef(BaseModel):
    identity: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,100}$')
    token: str = Field(default='', max_length=256, repr=False)


class LibraryQuery(BaseModel):
    sessions: list[ConversationRef] = Field(default_factory=list, max_length=100)
    include_reports: bool = True


class FileRef(ConversationRef):
    source: Literal['chat', 'report']
    key: str = Field(min_length=1, max_length=1000)


class ArchiveRequest(BaseModel):
    files: list[FileRef] = Field(min_length=1, max_length=200)


def token_request(request, token):
    headers = [(k, v) for k, v in request.scope['headers'] if k.lower() not in {b'x-session-access-token', b'authorization'}]
    headers.append((b'x-session-access-token', token.encode('ascii', errors='strict')))
    return Request({**request.scope, 'headers': headers})


async def chat(ref, request):
    state = await _get_session_or_404(ref.identity)
    try:
        scoped = token_request(request, ref.token)
    except UnicodeError:
        raise HTTPException(403, 'Conversation not found') from None
    _assert_session_access(state, scoped)
    roots = await asyncio.to_thread(session_roots, state, service, ref.identity)
    return state, roots


def file_item(source, identity, key, name, size, mtime, mode, title):
    return dict(id=f'{source}/{identity}/{key}', source=source, identity=identity, key=key,
                name=name, size=size, modified=mtime, mode=mode, title=title,
                extension=Path(name).suffix.lower().lstrip('.'))


def report_files(jobs, who, identity):
    identity, request = jobs.owned(who, identity)
    result = jobs.get(who, identity)
    payload = request['payload']
    data = {}
    if result.get('report'):
        data['report.md'] = result['report'].encode('utf-8')
        evidence = {key: result.get(key, []) for key in ['sources', 'sections', 'remote_services', 'predictions']}
        data['evidence.json'] = json.dumps(evidence, ensure_ascii=False, indent=2).encode('utf-8')
    if payload.get('structure_pdb'):
        data['input.pdb'] = payload['structure_pdb'].encode('utf-8')
    elif payload.get('sequence') and not payload.get('accession'):
        data['input.fasta'] = ('>input\n' + payload['sequence'] + '\n').encode('utf-8')
    return result, data


@router.post('/files')
async def files(payload: LibraryQuery, request: Request, response: Response, who=Depends(report_owner)):
    response.headers.update(PRIVATE)
    items, truncated, seen = [], False, set()
    for ref in payload.sessions:
        if ref.identity in seen:
            continue
        seen.add(ref.identity)
        try:
            state, roots = await chat(ref, request)
        except HTTPException as exc:
            if exc.status_code in {401, 403, 404}:
                continue
            raise
        result = await asyncio.to_thread(list_files, roots)
        truncated = truncated or result['truncated']
        mode = 'Scientist' if state.get('chat_mode') == 'science_expert' else 'Agent'
        title = str(state.get('title') or state.get('last_user_text') or mode)[:160]
        for row in result['files']:
            try:
                path = resolve_file(roots, row['key'])
                if not visible_file(path, roots[row['scope']]):
                    continue
            except (ValueError, OSError):
                continue
            items.append(file_item('chat', ref.identity, row['key'], row['name'], row['size'], row['mtime'], mode, title))
    if payload.include_reports:
        jobs = ResearchJobs()
        for record in await asyncio.to_thread(jobs.records.entries, 'report', who):
            try:
                result, artifacts = await asyncio.to_thread(report_files, jobs, who, record['identity'])
            except (FileNotFoundError, ValueError):
                continue
            for name, content in artifacts.items():
                items.append(file_item('report', result['id'], name, name, len(content), result['created'], 'Researcher', result['title']))
    return {'items': sorted(items, key=lambda row: row['modified'], reverse=True), 'truncated': truncated}


async def resolve(ref, request, who):
    if ref.source == 'chat':
        _, roots = await chat(ref, request)
        try:
            path = resolve_file(roots, ref.key)
            if not visible_file(path, roots[ref.key.split('/', 1)[0]]):
                raise FileNotFoundError()
        except (OSError, ValueError, KeyError):
            raise HTTPException(404, 'File not found') from None
        try:
            return path.name, open_owned_file(roots, ref.key)
        except (OSError, ValueError):
            raise HTTPException(404, 'File not found') from None
    try:
        _, artifacts = await asyncio.to_thread(report_files, ResearchJobs(), who, ref.identity)
        return ref.key, artifacts[ref.key]
    except (FileNotFoundError, KeyError, ValueError):
        raise HTTPException(404, 'File not found') from None


@router.post('/content')
async def content(payload: FileRef, request: Request, who=Depends(report_owner)):
    name, value = await resolve(payload, request, who)
    media = mimetypes.guess_type(name)[0] or 'application/octet-stream'
    if hasattr(value, 'read'):
        return download(value, name, media_type=media, headers=PRIVATE)
    return Response(value, media_type=media, headers={**PRIVATE, 'Content-Disposition': f'attachment; filename="{name}"'})


def build_archive(entries):
    stream = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    try:
        total = 0
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for key, value in entries:
                if hasattr(value, 'read'):
                    with archive.open(key, 'w') as out:
                        for data in chunks(value):
                            total += len(data)
                            if total > 200*1024**2: raise ValueError('Select at most 200 MB')
                            out.write(data)
                else:
                    total += len(value)
                    if total > 200*1024**2: raise ValueError('Select at most 200 MB')
                    archive.writestr(key, value)
        stream.seek(0)
        return stream
    except BaseException:
        stream.close()
        raise
    finally:
        for _, value in entries:
            if hasattr(value, 'close'): value.close()


@router.post('/archive')
async def archive(payload: ArchiveRequest, request: Request, who=Depends(report_owner)):
    entries, seen, total = [], set(), 0
    try:
        for ref in payload.files:
            key = (ref.source, ref.identity, ref.key)
            if key in seen:
                continue
            seen.add(key)
            name, value = await resolve(ref, request, who)
            entries.append((f'{ref.source}/{ref.identity}/{ref.key}', value))
            total += os.fstat(value.fileno()).st_size if hasattr(value, 'read') else len(value)
            if total > 200 * 1024 * 1024:
                raise HTTPException(413, 'Select at most 200 MB per download')
        stream = await asyncio.to_thread(build_archive, entries)
    finally:
        for _, value in entries:
            if hasattr(value, 'close'): value.close()
    def chunks():
        try:
            while data := stream.read(65536):
                yield data
        finally:
            stream.close()
    return StreamingResponse(chunks(), media_type='application/zip', headers={**PRIVATE, 'Content-Disposition': 'attachment; filename="library-files.zip"'})
