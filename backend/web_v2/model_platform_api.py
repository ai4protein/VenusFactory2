"""HTTP boundary for browser-private models, uploads, jobs and workflows."""
from urllib.parse import urlsplit
from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, Query
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from backend.web_v2.model_platform import Platform

router = APIRouter(prefix='/api/model-platform', tags=['Model platform'])
service = Platform()
COOKIE = 'vf_model_workspace'


def owner(request: Request, response: Response):
    from backend.web_v2.job_access import browser_owner
    return browser_owner(request, response, 'platform', service.session)


async def call(fn, *args, **kwargs):
    try:
        return await run_in_threadpool(fn, *args, **kwargs)
    except FileNotFoundError:
        raise HTTPException(404, 'Item not found')
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)[:300])


@router.get('/session')
def session(who=Depends(owner)):
    return {'ready': True}


@router.get('/catalog')
async def plaza_catalog(query: str = Query('', max_length=200), category: str = '', source: str = '', mine: bool = False,
                        kind: str = Query('', pattern='^(|model|tool)$'), offset: int = Query(0, ge=0),
                        limit: int = Query(24, ge=1, le=100), model_type: str = Query('', pattern='^(|foundation|pretrained|finetuned|trained|tool)$'), who=Depends(owner)):
    return await call(service.plaza_catalog, who, query, category, source, mine, kind, offset, limit, model_type)


@router.get('/models')
async def catalog(query: str = Query('', max_length=200), category: str = '', source: str = '', mine: bool = False,
                  mode: str = Query('advanced', pattern='^(quick|advanced)$'), offset: int = Query(0, ge=0), limit: int = Query(24, ge=1, le=100), who=Depends(owner)):
    return await call(service.catalog, who, query, category, source, mine, offset, limit, mode)


@router.get('/models/{identity}')
async def model(identity: str, who=Depends(owner)):
    return await call(service.model, who, identity, True)


@router.patch('/models/{identity}/visibility')
async def visibility(identity: str, body: dict, who=Depends(owner)):
    if not identity.startswith('user:'):
        raise HTTPException(422, 'Only your own models can be published')
    return await call(service.visibility, who, identity[5:], body.get('visibility'))


@router.get('/datasets')
async def datasets(who=Depends(owner)):
    return await call(service.datasets, who)


@router.post('/datasets')
async def upload(file: UploadFile = File(...), who=Depends(owner)):
    from backend.web_v2.upload_limits import read_upload
    content = await read_upload(file)
    return await call(service.upload, who, file.filename or 'data.csv', content)


@router.post('/train')
async def train(body: dict, who=Depends(owner)):
    return await call(service.start, who, 'train', body)


@router.post('/predict')
async def predict(body: dict, who=Depends(owner)):
    return await call(service.start, who, 'predict', body)


@router.get('/workflow-components')
async def workflow_components(query: str = Query('', max_length=200), category: str = '', source: str = '',
                              kind: str = Query('', pattern='^(|model|tool)$'), offset: int = Query(0, ge=0),
                              limit: int = Query(24, ge=1, le=100), who=Depends(owner)):
    return await call(service.workflow_components, who, query, category, source, kind, offset, limit)


@router.get('/workflow-components/{identity}')
async def workflow_component(identity: str, who=Depends(owner)):
    return await call(service.workflow_component, who, identity)


@router.get('/workflows')
async def workflows(who=Depends(owner)):
    return await call(service.workflows, who)


@router.post('/workflows')
async def save_workflow(body: dict, who=Depends(owner)):
    return await call(service.save_workflow, who, body)


@router.post('/workflows/run')
async def run_workflow(body: dict, who=Depends(owner)):
    return await call(service.start, who, 'workflow', body)


@router.get('/workflows/{identity}')
async def workflow(identity: str, who=Depends(owner)):
    return await call(service.workflow, who, identity)


@router.get('/job-history')
async def job_history(query: str = Query('', max_length=200), status: str = '', kind: str = '',
                      offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100), who=Depends(owner)):
    return await call(service.job_page, who, query=query, status=status, kind=kind, offset=offset, limit=limit)


@router.get('/jobs/{identity}')
async def job(identity: str, who=Depends(owner)):
    return await call(service.job, who, identity)


@router.get('/jobs/{identity}/result')
async def result(identity: str, who=Depends(owner)):
    identity = await call(service.resolve_job, who, identity)
    j = await call(service.owned, 'jobs', identity, who)
    if j['status'] != 'completed':
        raise HTTPException(409, 'Results are not ready')
    return FileResponse(service.root / 'results' / (identity + '.json'), media_type='application/json', filename='results.json', headers={'Cache-Control': 'no-store'})


@router.get('/jobs')
async def jobs(who=Depends(owner)):
    return await call(service.jobs, who)

@router.get('/jobs/{identity}/artifacts')
async def artifacts(identity: str, who=Depends(owner)):
    archive=await call(service.artifact_archive,who,identity)
    return FileResponse(archive,media_type='application/zip',filename='model-results.zip',headers={'Cache-Control':'no-store'})

@router.post('/jobs/{identity}/cancel')
async def cancel_job(identity: str, who=Depends(owner)):
    return await call(service.cancel_job, who, identity)

@router.post('/jobs/{identity}/retry')
async def retry_job(identity: str, who=Depends(owner)):
    return await call(service.retry_job, who, identity)
