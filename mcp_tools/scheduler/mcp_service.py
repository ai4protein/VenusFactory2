"""MCP queue gateway and authenticated scheduler control plane."""
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
import hashlib
import os
import secrets

from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from fastmcp.server.middleware import Middleware
from fastmcp.tools.tool import ToolResult
from mcp.types import ContentBlock
from pydantic import TypeAdapter
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from mcp_tools.scheduler.catalog import catalog, classify
from mcp_tools.scheduler.jobs import ToolJobs
from mcp_tools.scheduler.service import REPOSITORY_ID, service_key

scheduler=FastMCP('VenusFactory Tool Scheduler')


async def operation(fn, *args, **kwargs):
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except FileNotFoundError:
        return {'error': {'code': 404, 'message': 'Task not found'}}
    except ValueError as exc:
        full = 'queue is full' in str(exc).lower()
        return {'error': {'code': 429 if full else 422, 'message': 'Task queue is full' if full else 'Invalid tool arguments'}}


def authorize():
    value=get_http_headers(include_all=True).get('authorization','')
    if not secrets.compare_digest(value,'Bearer '+service_key()):raise PermissionError('Invalid service credential')


@scheduler.tool()
async def tool_catalog() -> dict:
    """List categorized tools, input schemas and execution resources."""
    authorize();return await asyncio.to_thread(catalog)


@scheduler.tool()
async def submit_tool(name: str, arguments: dict, owner: str, session_id: str | None=None, parent_job: str | None=None) -> dict:
    """Enqueue a registered tool and return its job number without waiting for computation."""
    authorize();return await operation(ToolJobs().submit,name,arguments,owner=owner,session_id=session_id,parent_job=parent_job)


@scheduler.tool()
async def submit_public_analysis(service: str, context: dict, owner: str, cache_dir: str | None=None, parent_job: str | None=None) -> dict:
    """Enqueue a report's public analysis, preserving its remote ticket cache."""
    authorize()
    return await operation(ToolJobs().submit,'query_public_protein_service',
                           {'service':service,'context':context,'cache_dir':cache_dir},owner=owner,route='public_service',parent_job=parent_job)


@scheduler.tool()
async def tool_job(identity: str, owner: str) -> dict:
    """Read a tool job belonging to the specified backend-authenticated owner."""
    authorize();return await operation(ToolJobs().get,identity,owner)


@scheduler.tool()
async def list_tool_jobs(owner: str) -> dict:
    """List only this owner's tool jobs."""
    authorize();return {'items':await asyncio.to_thread(ToolJobs().list,owner)}


@scheduler.tool()
async def cancel_tool_job(identity: str, owner: str) -> dict:
    """Cancel a queued or running tool job."""
    authorize();return await operation(ToolJobs().cancel,identity,owner)


class QueueMiddleware(Middleware):
    async def on_list_tools(self, context, call_next):
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        return [t for t in await call_next(context) if is_tool_allowed(t.name)]

    async def on_call_tool(self, context, call_next):
        from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
        name=context.message.name
        assert_tool_call_allowed(name, context.message.arguments or {})
        if os.getenv('VENUS_MCP_QUEUE_WORKER') == '1': return await call_next(context)
        if classify(name)['execution']=='inline':return await call_next(context)
        # Resolve before queuing so an unknown name never starts a worker.
        from mcp_tools.mcp_server import mcp
        if name not in {t.name for t in await mcp.list_tools()}:raise ValueError('Unknown tool')
        ctx=context.fastmcp_context
        sid=getattr(ctx,'session_id',None) or get_http_headers().get('mcp-session-id') or secrets.token_hex(16)
        owner='mcp:'+hashlib.sha256(str(sid).encode()).hexdigest()
        jobs=ToolJobs()
        job=await asyncio.to_thread(jobs.submit,name,context.message.arguments or {},owner=owner,route='mcp')
        final=await jobs.wait(job['id'],owner)
        result=final.get('result',{})
        if 'content' in result:
            content = [TypeAdapter(ContentBlock).validate_python(item) for item in result['content']]
            return ToolResult(content=content,structured_content=result.get('structured_content'),is_error=result.get('is_error',False))
        return ToolResult(content=result.get('message') or 'Tool task '+final['status'],is_error=True)


def application(mcp):
    public=mcp.http_app(path='/mcp')
    internal=scheduler.http_app(path='/mcp',stateless_http=True)
    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            await stack.enter_async_context(public.lifespan(public))
            await stack.enter_async_context(internal.lifespan(internal))
            yield
    async def health(request):
        return JSONResponse({'service':'venusfactory-tools','repository_id':REPOSITORY_ID,'pid':os.getpid(),'status':'ready'})
    return Starlette(routes=[Route('/health',health),Mount('/scheduler',internal),Mount('/',public)],lifespan=lifespan)
