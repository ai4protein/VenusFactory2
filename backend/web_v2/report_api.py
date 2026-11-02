"""Researcher HTTP boundary. Every report runs in the shared, owner-scoped queue."""
import asyncio
import html
import json
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.web_v2.job_access import browser_owner
from backend.web_v2.research_jobs import ResearchJobs
from backend.web_v2.research_report import model_options, parsed_input, resolve_protein

router = APIRouter(prefix='/api/report', tags=['Researcher'])


def owner(request: Request, response: Response):
    return browser_owner(request, response, 'report')


class ResearchRequest(BaseModel):
    request_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{8,80}$")
    model_id: str = Field(max_length=80)
    protein: str = Field(default='', max_length=180)
    accession: str = Field(default='', max_length=20)
    sequence: str = Field(default='', max_length=10000)
    structure_pdb: str = Field(default='')
    structure_chain: str = Field(default='', max_length=1)
    focus: str = Field(default='', max_length=3000)
    analyses: list[str] = Field(min_length=1, max_length=32)
    lang: Literal['en', 'zh'] = 'en'
    material_ids: list[str] = Field(default_factory=list,max_length=12)


class ParseRequest(BaseModel):
    content: str


async def call(fn, *args):
    try:
        return await run_in_threadpool(fn, *args)
    except FileNotFoundError:
        raise HTTPException(404, 'Research not found') from None
    except (ValueError, KeyError) as exc:
        raise HTTPException(422, str(exc)[:200]) from None
    except RuntimeError:
        raise HTTPException(429, 'The queue is full. Please try again shortly.') from None


@router.get('/options')
async def options(who=Depends(owner)):
    return await call(model_options)


@router.get('/resolve')
async def resolve(query: str = Query(min_length=1, max_length=180), who=Depends(owner)):
    try:
        return {'items': await resolve_protein(query)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except Exception:
        raise HTTPException(502, 'UniProt is temporarily unavailable. Try an accession or upload a sequence.') from None


@router.post('/parse-input')
async def parse(payload: ParseRequest, who=Depends(owner)):
    from backend.web_v2.upload_limits import check_upload_size
    check_upload_size(len(payload.content.encode("utf-8")))
    parsed = await call(parsed_input, payload.content)
    rows = parsed['sequence_map']
    if not rows:
        raise HTTPException(422, 'No protein chain was found')
    return parsed


@router.post('/upload')
async def upload(file: UploadFile = File(...), who=Depends(owner)):
    from backend.web_v2.upload_limits import read_upload
    content = await read_upload(file)
    try:
        text = content.decode('utf-8-sig')
    except UnicodeError:
        raise HTTPException(422, 'Upload a FASTA, PDB or sequence text file') from None
    parsed = await call(parsed_input, text)
    rows = parsed['sequence_map']
    if not rows:
        raise HTTPException(422, 'No protein chain was found')
    return {**parsed, 'name': (file.filename or 'Sequence')[:160]}


@router.post('/materials')
async def upload_material(file: UploadFile = File(...), who=Depends(owner)):
    from backend.web_v2.research_materials import upload
    try:return await upload(ResearchJobs().root,who,file)
    except ValueError as exc:raise HTTPException(422,str(exc)[:200]) from None

@router.get('/jobs/{identity}/materials/{material_id}')
async def material_download(identity: str, material_id: str, who=Depends(owner)):
    from backend.web_v2.research_materials import download
    path,name=await call(download,ResearchJobs(),who,identity,material_id)
    return FileResponse(path,filename=name,headers={'Cache-Control':'private, no-store','X-Content-Type-Options':'nosniff'})

@router.post('/jobs', status_code=202)
@router.post('/generate', status_code=202)
async def create(payload: ResearchRequest, request: Request, who=Depends(owner)):
    from functools import partial
    from backend.web_v2.online_access import admit
    from backend.web_v2.upload_limits import check_upload_size
    check_upload_size(len(payload.structure_pdb.encode('utf-8')))
    return await call(partial(ResearchJobs().start, admit=lambda ident: admit(request,"report:"+ident,"researcher")), who, payload.model_dump())


@router.get('/jobs')
async def history(who=Depends(owner)):
    return {'items': await call(ResearchJobs().history, who)}


@router.get('/jobs/{identity}')
async def job(identity: str, who=Depends(owner)):
    return await call(ResearchJobs().get, who, identity)


@router.delete('/jobs/{identity}')
async def delete_job(identity: str, who=Depends(owner)):
    return await call(ResearchJobs().delete_online, who, identity)


@router.post('/jobs/{identity}/resume', status_code=202)
async def resume(identity: str, request: Request, who=Depends(owner)):
    from functools import partial
    from backend.web_v2.online_access import admit
    return await call(partial(ResearchJobs().resume, admit=lambda ident: admit(request, "report:"+ident, "researcher")), who, identity)


@router.post('/jobs/{identity}/cancel')
async def cancel(identity: str, who=Depends(owner)):
    return await call(ResearchJobs().cancel, who, identity)


@router.get('/jobs/{identity}/download')
async def download(identity: str, format: Literal['md', 'html', 'json'] = 'md', who=Depends(owner)):
    result = await call(ResearchJobs().get, who, identity)
    if not result.get('report'):
        raise HTTPException(409, 'The report is not ready yet')
    if format == 'json':
        content = json.dumps(result, ensure_ascii=False, indent=2)
        mime = 'application/json'
    elif format == 'html':
        # Escape content; the file is self-contained and prints cleanly to PDF.
        import markdown
        rendered = markdown.markdown(html.escape(result['report']), extensions=['tables', 'fenced_code'])
        content = '<!doctype html><meta charset="utf-8"><title>' + html.escape(result['title']) + '</title><style>body{font:16px/1.7 system-ui;max-width:900px;margin:48px auto;padding:24px;color:#222}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ddd;padding:10px}a{color:#365777}@media print{body{margin:0}}</style>' + rendered
        mime = 'text/html'
    else:
        content, mime = result['report'], 'text/markdown'
    return Response(content, media_type=mime, headers={'Content-Disposition': f'attachment; filename="{result["job_number"]}.{format}"', 'Cache-Control': 'private, no-store'})


@router.post('/generate/stream')
async def generate_stream(payload: ResearchRequest, request: Request, response: Response, who=Depends(owner)):
    service = ResearchJobs()
    from functools import partial
    from backend.web_v2.online_access import admit
    from backend.web_v2.upload_limits import check_upload_size
    check_upload_size(len(payload.structure_pdb.encode('utf-8')))
    result = await call(partial(service.start, admit=lambda ident: admit(request,"report:"+ident,"researcher")), who, payload.model_dump())
    async def stream():
        previous = ''
        while True:
            item = await call(service.get, who, result['id'])
            data = json.dumps(item, ensure_ascii=False)
            if data != previous:
                yield 'event: progress\ndata: ' + data + '\n\n'
                previous = data
            if item['status'] in {'completed', 'incomplete', 'failed', 'cancelled', 'interrupted'}:
                yield 'event: done\ndata: ' + data + '\n\n'; break
            yield ': heartbeat\n\n'
            await asyncio.sleep(2)
    headers = {'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'}
    stream_response = StreamingResponse(stream(), media_type='text/event-stream', headers=headers)
    # Preserve the browser identity established by the dependency on the first request.
    for k, v in response.raw_headers:
        if k == b'set-cookie': stream_response.raw_headers.append((k, v))
    return stream_response
