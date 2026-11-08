"""Exercise the real session MCP HTTP boundary and local sandbox, without an LLM."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.harness import web_native_worker
from agent.harness.contracts import Budget, ModelBinding, Profile
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.session_boundary import session_tools


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['agent','scientist'])
async def test_mcp_mount_is_implicit_and_bound_to_each_session(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    # No shared scientific services or external model endpoints are contacted.
    monkeypatch.setattr('mcp_tools.scheduler.catalog.native_tools',lambda:{})
    sessions={};connections={};ready=asyncio.Event();written=set();inputs_ready=asyncio.Event();finished=[]
    verbs = ('write_session_file','run_session_python','read_session_file')

    class ProtocolClient:
        async def run(self,context):
            identity=context.task.workspace.name
            peer='second' if identity=='first' else 'first'
            connection=json.loads((context.task.capability/'connection.json').read_text())
            connections[identity]=connection
            if len(connections)==2:ready.set()
            await asyncio.wait_for(ready.wait(),10)
            async with httpx.AsyncClient(timeout=30,trust_env=False) as client:
                counter=0
                async def rpc(method,params=None,token=None):
                    nonlocal counter
                    counter+=1
                    return await client.post(connection['url'],headers={'Authorization':'Bearer '+(token or connection['token'])},json={'jsonrpc':'2.0','id':counter,'method':method,'params':params or {}})
                inspected=set()
                async def invoke(name,arguments):
                    if name not in inspected:
                        description=(await rpc('tools/call',{'name':'describe_tool','arguments':{'name':name}})).json()
                        assert 'error' not in description,description
                        inspected.add(name)
                    return await rpc('tools/call',{'name':'call_tool','arguments':{'name':name,'arguments':arguments}})
                async def call(name,arguments):
                    response=await invoke(name,arguments)
                    response.raise_for_status()
                    body=response.json()
                    assert 'error' not in body,body
                    assert not body['result']['isError'],body
                    if mode=='scientist':assert 'tool calls remain' in body['result']['content'][1]['text']
                    return json.loads(body['result']['content'][0]['text'])

                initialized=(await rpc('initialize',{'protocolVersion':'2024-11-05'})).json()
                assert initialized['result']['capabilities']['tools']=={}
                assert (await rpc('tools/list',token=connections[peer]['token'])).status_code==401
                advertised=(await rpc('tools/list')).json()['result']['tools']
                assert {'search_tools','describe_tool','call_tool'}<={tool['name'] for tool in advertised}
                assert set(verbs)<={tool['name'] for tool in advertised}
                for name in verbs:
                    described=(await rpc('tools/call',{'name':'describe_tool','arguments':{'name':name}})).json()
                    entry=json.loads(described['result']['content'][0]['text'])
                    assert entry['name']==name and entry['call_via']==name
                assert not {'agent_generated_code','python_repl','write_research_file','run_research_python'} & {tool['name'] for tool in advertised}
                for tool in advertised:
                    assert tool['inputSchema']['additionalProperties'] is False
                    assert not {'workspace','session_id','mounts','backend','control'} & tool['inputSchema']['properties'].keys()

                await call(verbs[0],{'path':'input.txt','content':identity})
                code='import os,json\nfrom pathlib import Path\nvalue={"cwd":os.getcwd(),"identity":Path("input.txt").read_text(),"peer_visible":Path('+repr(str(sessions[peer]/'input.txt'))+').exists()}\nPath("result.json").write_text(json.dumps(value))\nprint("verified")\n'
                await call(verbs[0],{'path':'verify.py','content':code})
                written.add(identity)
                if len(written)==2:inputs_ready.set()
                await asyncio.wait_for(inputs_ready.wait(),10)
                assert (sessions[peer]/'input.txt').exists()
                for field,value in [('workspace',str(sessions[peer])),('session_id',peer),('mounts',['/']),('backend','user'),('control','/tmp')]:
                    body=(await invoke(verbs[1],{'path':'verify.py',field:value})).json()
                    assert body['result']['isError'] is True,body
                    assert body['result']['content'][0]['text'].startswith('ValueError:'),body
                    assert 'session workspace is assigned by the server' in body['result']['content'][0]['text']
                    assert not (sessions[identity]/'result.json').exists()
                result=await call(verbs[1],{'path':'verify.py','timeout_seconds':10})
                assert result['backend']=='bwrap' and result['exit_code']==0,result
                saved=json.loads((await call(verbs[2],{'path':'result.json'}))['text'])
                assert saved=={'cwd':'/workspace','identity':identity,'peer_visible':False}
                for path in ['../'+peer+'/input.txt',str(sessions[peer]/'input.txt')]:
                    denied=(await invoke(verbs[2],{'path':path})).json()
                    assert denied['result']['isError'] is True
                if mode=='agent':
                    shell=await call('run_session_shell',{'command':'pwd','timeout_seconds':10})
                    assert shell['exit_code']==0 and shell['stdout'].strip()=='/workspace'
                finished.append(identity)
            return {'status':'completed','text':'Session MCP sandbox verified'}

    monkeypatch.setattr(web_native_worker,'builtin_registry',lambda:SimpleNamespace(resolve=lambda *args:(ProtocolClient(),None)))
    profile=Profile('science_agent','deepseek-harness',ModelBinding('fixture','unused','http://127.0.0.1:1/v1'),Budget(tool_calls=24))
    pending=[]
    for identity in ['first','second']:
        workspace=tmp_path/identity;workspace.mkdir();sessions[identity]=workspace
        control=tmp_path/(identity+'-control');control.mkdir()
        tools=session_tools(workspace,control/'tools') if mode=='agent' else ResearchWorkspace(workspace,control/'tools',{'steps':[]}).tools()
        request={'session_id':identity,'workspace':str(workspace),'control':str(control),'profile':profile.public(),'runtimes':{},'base_url':'http://127.0.0.1:1','token':'unused','text':'MCP protocol test'}
        pending.append(web_native_worker.run(request,tools=tools,event_sink=lambda event:None))
    await asyncio.wait_for(asyncio.gather(*pending),60)
    assert set(finished)=={'first','second'}
    assert (sessions['first']/'input.txt').read_text()=='first'
    assert (sessions['second']/'input.txt').read_text()=='second'
