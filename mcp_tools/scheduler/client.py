"""Backend-to-MCP asynchronous submission and status polling."""
import asyncio
import json
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from backend.web_v2.job_queue import TERMINAL
from mcp_tools.scheduler.service import base_url, service_key


class ToolJobError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


async def rpc(name, arguments):
    transport=StreamableHttpTransport(base_url()+'/scheduler/mcp',headers={'Authorization':'Bearer '+service_key()})
    async with Client(transport,timeout=30) as client:
        result=await client.call_tool(name,arguments)
        data = json.loads(result.content[0].text)
        if 'error' in data:
            raise ToolJobError(data['error']['code'], data['error']['message'])
        return data


async def invoke(name, arguments, *, owner, session_id=None, parent_job=None, on_status=None):
    args={'name':name,'arguments':arguments,'owner':owner,'session_id':session_id}
    if parent_job: args['parent_job']=parent_job
    job=await rpc('submit_tool',args)
    return await wait(job, owner, on_status)


async def wait(job, owner, on_status=None):
    previous=None
    try:
        while True:
            state=(job['status'],job['position'],json.dumps(job.get('progress'),sort_keys=True))
            if state!=previous and on_status:on_status({k:v for k,v in job.items() if k!='result'})
            previous=state
            if job['status'] in TERMINAL:
                result=job.get('result',{})
                # Preserve tool-level failure details for harness repair/iteration.
                # Infrastructure failures have no tool value and still raise.
                if job['status'] in {'completed','failed'} and 'value' in result:
                    return result['value']
                if job['status']!='completed' or result.get('is_error'):
                    raise RuntimeError(result.get('message') or 'Tool task '+job['status'])
                return result['value']
            await asyncio.sleep(.5)
            job=await rpc('tool_job',{'identity':job['id'],'owner':owner})
    except asyncio.CancelledError:
        try:await asyncio.shield(rpc('cancel_tool_job',{'identity':job['id'],'owner':owner}))
        finally:raise


async def report_service(service, context, update, cache_dir=None):
    import os
    owner=os.environ['VENUS_QUEUE_OWNER']
    job=await rpc('submit_public_analysis',{'service':service,'context':context,'owner':owner,
                                           'cache_dir':str(cache_dir) if cache_dir else None,'parent_job':os.environ.get('VENUS_QUEUE_JOB_ID')})
    def progress(row):
        if row.get('progress'):update({**row['progress'],'job_number':row['job_number']})
    value=await wait(job,owner,progress)
    update({**value,'job_number':job['job_number']})
    return value
