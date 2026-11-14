"""Browser-owned and operator-curated RSI repositories; no execution endpoints."""
import asyncio
from fastapi import APIRouter, HTTPException, Query, Request, Response
from starlette.responses import StreamingResponse
from backend.web_v2.http_security import require_same_origin
from backend.web_v2.rsi_plaza import RsiPlaza
from backend.web_v2.science_goals import service as science

router = APIRouter(prefix='/api/rsi-plaza', tags=['RSI cases'])


async def sessions(request):
    from backend.web_v2.chat_api._shared import get_visible_session_ids, _get_session_or_404, _assert_session_access
    require_same_origin(request)
    allowed = set()
    for sid in await get_visible_session_ids(request):
        try:
            state = await _get_session_or_404(sid)
            _assert_session_access(state, request)
            allowed.add(sid)
        except HTTPException:
            continue
    return allowed


async def call(fn, *args):
    try:
        return await asyncio.to_thread(fn, *args)
    except (FileNotFoundError, KeyError):
        raise HTTPException(404, 'Case or file not found') from None
    except (OSError, ValueError, TypeError):
        raise HTTPException(409, 'Case files changed or are unavailable') from None


@router.get('/cases')
async def listing(request: Request, response: Response, query: str = Query('', max_length=200),
                  view: str = Query('all', pattern='^(all|mine)$'), offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=100)):
    response.headers['Cache-Control'] = 'private, no-store'
    return await call(RsiPlaza(science).listing, await sessions(request), query, view, offset, limit)


@router.get('/cases/{identity}')
async def detail(identity: str, request: Request, response: Response):
    response.headers['Cache-Control'] = 'private, no-store'
    return await call(RsiPlaza(science).detail, identity, await sessions(request))


@router.get('/cases/{identity}/file')
async def preview(identity: str, request: Request, response: Response, path: str = Query(..., max_length=1000)):
    response.headers['Cache-Control'] = 'private, no-store'
    return await call(RsiPlaza(science).preview, identity, await sessions(request), path)


@router.get('/cases/{identity}/download')
async def download(identity: str, request: Request):
    archive = await call(RsiPlaza(science).download, identity, await sessions(request))
    def body():
        try:
            while chunk := archive.read(1024**2):
                yield chunk
        finally:
            archive.close()
    return StreamingResponse(body(), media_type='application/zip', headers={
        'Cache-Control': 'private, no-store', 'Content-Disposition': 'attachment; filename="rsi-case.zip"'})
