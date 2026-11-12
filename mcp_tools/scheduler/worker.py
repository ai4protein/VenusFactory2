"""One queued tool execution, supervised and cancellable as a process tree."""
import asyncio
import json
import os
from pathlib import Path
import sys

from mcp_tools.scheduler.jobs import ToolJobs, save_json


def mcp_result(value):
    from agent.chat_agent_utils import _tool_output_indicates_failure
    failed = value.is_error or any(
        _tool_output_indicates_failure(block.text)[0]
        for block in value.content if getattr(block, 'type', None) == 'text'
    )
    return {'content':[c.model_dump(mode='json') for c in value.content],
            'structured_content':value.structured_content,'is_error':failed}


async def execute(spec):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
    assert_tool_call_allowed(spec['name'], spec['arguments'])
    capture_spec = dict(spec, arguments=dict(spec['arguments']))
    if spec.get('workspace'):
        from agent.harness.session_boundary import bind_arguments
        from agent.harness.session_sandbox import check_workspace
        workspace = Path(spec['workspace']).resolve(strict=True)
        check_workspace(workspace)
        spec['arguments'] = bind_arguments(spec['name'], spec['arguments'], workspace)
        os.environ['VENUS_TOOL_WORKSPACE'] = str(workspace)
        os.environ['TEMP_OUTPUTS_DIR'] = str(workspace / 'outputs')
        os.environ['VENUS_RSI_WORKSPACE'] = str(workspace)
        os.environ['VENUS_RSI_REGISTRY_ROOT'] = str(workspace / 'registry')
        os.environ['VENUS_RSI_SKILLS_ROOT'] = str(workspace / 'skills')
        from agent.rsi.engine.context import get_resources
        get_resources.cache_clear()
    if spec['route']=='public_service':
        from mcp_tools.tools.research.services import run_service
        arguments=spec['arguments']
        progress=ToolJobs().queue.files(os.environ['VENUS_QUEUE_JOB_ID'])/'mcp-progress.json'
        value=await run_service(arguments['service'],arguments['context'],lambda row:save_json(progress,row),
                                cache_dir=arguments.get('cache_dir'))
        return {'value':value,'is_error':False}
    if spec['route']=='native':
        from mcp_tools.scheduler.analysis_evidence import AnalysisCapture
        from mcp_tools.scheduler.catalog import classify, native_tools
        tool=native_tools()[spec['name']]
        capture = (AnalysisCapture(capture_spec, workspace, classify(spec['name'], getattr(tool, 'metadata', None))['category'])
                   if spec.get('workspace') else None)
        value=await tool.ainvoke(spec['arguments'],config={'configurable':{'session_id':spec.get('session_id')}})
        from agent.chat_agent_utils import _tool_output_indicates_failure
        failed,_=_tool_output_indicates_failure(value)
        result = {'value':value,'is_error':failed}
        if capture is not None:
            evidence = capture.finish(result)
            if evidence is not None:
                result['analysis_evidence'] = evidence
        return result
    # Same MCP implementation as the HTTP catalog, but no recursive queue wrapper.
    os.environ['VENUS_MCP_QUEUE_WORKER']='1'
    from fastmcp import Client
    from mcp_tools.mcp_server import mcp
    from mcp_tools.scheduler.context import bind_mcp_session
    with bind_mcp_session(spec.get('session_id')):
        async with Client(mcp) as client:
            value=await client.call_tool(spec['name'],spec['arguments'],raise_on_error=False)
            return mcp_result(value)


def main():
    from dotenv import load_dotenv
    load_dotenv()
    jobs=ToolJobs();identity=os.environ['VENUS_QUEUE_JOB_ID']
    path=Path(sys.argv[1]).resolve()
    if path.parent != (jobs.queue.root/'mcp-inputs').resolve():raise ValueError('Invalid queued input')
    spec=json.loads(path.read_text())
    result_path=jobs.queue.files(identity)/'mcp-result.json'
    from backend.web_v2.usage_store import Observation
    observation=Observation('tool',spec['name'],identity='tool:'+identity)
    try:
        result=asyncio.run(execute(spec));save_json(result_path,result)
        observation.finish(failed=result['is_error'])
        return 1 if result['is_error'] else 0
    except Exception as exc:
        observation.finish(failed=True,error_type=type(exc).__name__)
        from backend.web_v2.redact import redact_for_frontend
        save_json(result_path,{'is_error':True,'error_type':type(exc).__name__,'message':redact_for_frontend(str(exc))[:500]})
        return 1
    finally:
        path.unlink(missing_ok=True)


if __name__=='__main__':sys.exit(main())
