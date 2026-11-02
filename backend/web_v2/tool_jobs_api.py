"""Browser adapter: private asynchronous tool jobs, never client-selected owners."""
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from mcp_tools.scheduler.client import rpc, ToolJobError
from mcp_tools.scheduler.catalog import WEB, native_tools
from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
from backend.web_v2.job_access import browser_owner, require_same_origin

router = APIRouter(prefix='/api/tools', tags=['Scientific tools'])
# These read/query tools can be submitted without a model/domain-specific form.
# Training, generated code and registered-model inference use their existing
# authorized forms or the session harness, sharing this MCP queue internally.
QUERY_TOOLS = WEB | {'query_ncbi', 'query_gget', 'query_opentargets', 'query_ensembl', 'query_pubchem', 'query_ncbi_blast',
                     'query_foldseek_remote', 'query_public_protein_service'}


class Submission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1, max_length=120)
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = Field(default=None, max_length=100)


async def owner_for(request, response, session_id):
    require_same_origin(request)
    if session_id:
        from backend.web_v2.chat_api._shared import _get_session_or_404, _assert_session_access
        state = await _get_session_or_404(session_id)
        _assert_session_access(state, request)
        response.headers['Cache-Control'] = 'private, no-store'
        return 'chat:' + session_id, Path(state['agent_session_dir']).resolve()
    return browser_owner(request, response, 'mcp'), None


def check_paths(arguments, workspace):
    for key, value in arguments.items():
        if isinstance(value, dict):
            check_paths(value, workspace)
        elif value and any(word in key.lower() for word in ('path', 'file', 'directory', 'output_dir')):
            if workspace is None:
                raise HTTPException(422, 'Choose a file from a chat workspace')
            values = value if isinstance(value, list) else [value]
            for item in values:
                if not isinstance(item, str):
                    raise HTTPException(422, 'Invalid file path')
                path = Path(item)
                resolved = (path if path.is_absolute() else workspace / path).resolve()
                if not resolved.is_relative_to(workspace):
                    raise HTTPException(422, 'Choose a file from this workspace')
            arguments[key] = [str((Path(x) if Path(x).is_absolute() else workspace / x).resolve()) for x in values] if isinstance(value, list) else str(resolved)


async def call(name, arguments):
    try:
        return await rpc(name, arguments)
    except ToolJobError as exc:
        raise HTTPException(exc.code, str(exc)) from None
    except Exception:
        raise HTTPException(503, 'Tool service is unavailable') from None


@router.get('/catalog')
async def catalog():
    return await call('tool_catalog', {})


@router.post('/jobs', status_code=202)
async def submit(body: Submission, request: Request, response: Response):
    owner, workspace = await owner_for(request, response, body.session_id)
    if body.name not in QUERY_TOOLS or body.name not in native_tools():
        raise HTTPException(422, 'Submit this tool through its model form or chat')
    if not is_tool_allowed(body.name):
        raise HTTPException(403, 'This tool is unavailable in online mode')
    check_paths(body.arguments, workspace)
    job = await call('submit_tool', {'name': body.name, 'arguments': body.arguments, 'owner': owner, 'session_id': body.session_id})
    response.headers['Location'] = '/api/tools/jobs/' + job['id']
    return job


@router.get('/jobs')
async def jobs(request: Request, response: Response, session_id: str | None = None):
    owner, _ = await owner_for(request, response, session_id)
    return await call('list_tool_jobs', {'owner': owner})


@router.get('/jobs/{identity}')
async def job(identity: str, request: Request, response: Response, session_id: str | None = None):
    owner, _ = await owner_for(request, response, session_id)
    return await call('tool_job', {'identity': identity, 'owner': owner})


@router.post('/jobs/{identity}/cancel')
async def cancel(identity: str, request: Request, response: Response, session_id: str | None = None):
    owner, _ = await owner_for(request, response, session_id)
    return await call('cancel_tool_job', {'identity': identity, 'owner': owner})
