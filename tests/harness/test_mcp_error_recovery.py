"""Official MCP client can recover from a tool failure without reconnecting."""
import sys
from types import SimpleNamespace
import pytest
from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from agent.harness.contracts import Profile, Budget, ModelBinding

@pytest.mark.asyncio
@pytest.mark.parametrize("harness", ["kimi-code","deepseek-harness","codex","claude-code"])
async def test_tool_failure_reaches_model_and_corrected_call_succeeds(tmp_path,monkeypatch,harness):
    from agent.harness import web_native_worker as worker
    work, control = tmp_path/"work",tmp_path/"control"
    work.mkdir();control.mkdir()
    invocations=[]
    async def list_files(directory: str=".") -> dict:
        """List files in a workspace directory."""
        invocations.append(directory)
        if directory != ".":
            raise ValueError("Use a workspace-relative directory")
        return {"files":["results.csv"]}
    tool=StructuredTool.from_function(coroutine=list_files,name="list_session_files",
                                     metadata={"workspace_relative_paths":True})
    async def adapter(ctx):
        parameters=StdioServerParameters(command=sys.executable,
            args=[str(ctx.task.capability/"web_tools.py")])
        async with stdio_client(parameters) as (read,write):
            async with ClientSession(read,write) as client:
                await client.initialize()
                failed=await client.call_tool("list_session_files",{"directory":"bad"})
                assert failed.isError
                assert "workspace-relative directory" in failed.content[0].text
                fixed=await client.call_tool("list_session_files",{"directory":"."})
                assert not fixed.isError and "results.csv" in fixed.content[0].text
                exhausted=await client.call_tool("list_session_files",{"directory":"."})
                assert exhausted.isError
                assert "budget" in exhausted.content[0].text.lower()
                with pytest.raises(McpError):
                    await client.call_tool("unknown_tool",{})
        assert invocations==["bad","."]
        return {"status":"completed","text":"Recovered within the original two-call allowance."}
    monkeypatch.setattr(worker,"builtin_registry",lambda:SimpleNamespace(
        resolve=lambda *args:(SimpleNamespace(run=adapter),None)))
    profile=Profile("science_agent",harness,
        ModelBinding("fixture","unused","http://127.0.0.1:1"),Budget(tool_calls=2))
    await worker.run({"profile":profile.public(),"workspace":str(work),"control":str(control),
        "session_id":"fixture","runtimes":{},"base_url":"http://127.0.0.1:1",
        "token":"fixture","text":"List the result files"},tools=[tool])
