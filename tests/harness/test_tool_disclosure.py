import json
from types import SimpleNamespace

import pytest

from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError, brief_description


def catalog(count=1, enabled=None):
    tools = {f'protein_model_{i}': SimpleNamespace(name=f'protein_model_{i}',
             description='Predict protein solubility.\n\nArgs:\n' + 'long parameter documentation ' * 300)
             for i in range(count)}
    return ToolDisclosure(tools, schema=lambda t: {'type': 'object', 'properties': {'sequence': {'type': 'string'}}, 'required': ['sequence']},
                          allowed=lambda t: enabled is None or t.name in enabled)


def test_ten_thousand_tools_do_not_expand_initial_schema_or_search_page():
    one, many = catalog(), catalog(10000)
    assert one.surface() == many.surface()
    assert len(json.dumps(many.surface())) < 2000
    page = many.metadata('search_tools', {'query': 'solubility', 'limit': 20})
    assert page['total'] == 10000 and page['next_offset'] == 20
    assert len(page['tools']) == 20
    assert all(set(row) == {'name', 'description'} and len(row['description']) <= 180 for row in page['tools'])
    next_page = many.metadata('search_tools', {'query': 'solubility', 'limit': 20, 'offset': 20})
    assert not {r['name'] for r in page['tools']} & {r['name'] for r in next_page['tools']}


def test_full_interface_is_returned_only_after_inspection_and_is_turn_scoped():
    c = catalog()
    with pytest.raises(DisclosureError, match='describe_tool'):
        c.resolve('call_tool', {'name': 'protein_model_0', 'arguments': {'sequence': 'ACDE'}})
    description = c.metadata('describe_tool', {'name': 'protein_model_0'})
    assert description['inputSchema']['required'] == ['sequence']
    assert 'long parameter documentation' in description['description']
    assert c.resolve('call_tool', {'name': 'protein_model_0', 'arguments': {'sequence': 'ACDE'}})[2] == {'sequence': 'ACDE'}
    with pytest.raises(DisclosureError):
        catalog().resolve('call_tool', {'name': 'protein_model_0', 'arguments': {}})
    with pytest.raises(DisclosureError):
        c.resolve('protein_model_0', {'sequence': 'ACDE'})


def test_disabled_tools_are_not_discoverable_and_cannot_be_called_after_revocation():
    enabled = {'protein_model_0'}
    c = catalog(3, enabled)
    assert c.metadata('search_tools', {})['total'] == 1
    c.metadata('describe_tool', {'name': 'protein_model_0'})
    enabled.clear()
    assert c.metadata('search_tools', {})['total'] == 0
    with pytest.raises(DisclosureError, match='unavailable'):
        c.resolve('call_tool', {'name': 'protein_model_0', 'arguments': {}})
    with pytest.raises(DisclosureError, match='unavailable'):
        c.metadata('describe_tool', {'name': 'protein_model_1'})


@pytest.mark.parametrize('args', [{'limit': 10000}, {'limit': True}, {'offset': -1}, {'query': 'x'*201}, {'schema': True}])
def test_index_requests_cannot_bypass_bounds(args):
    with pytest.raises(DisclosureError):
        catalog().metadata('search_tools', args)


def test_brief_descriptions_do_not_copy_argument_sections():
    assert brief_description('Predict binding. More information.\n\nArgs:\nsecret long schema') == 'Predict binding.'


def test_new_scientific_tool_defaults_are_bound_to_each_session(tmp_path):
    from pathlib import Path
    from langchain_core.tools import StructuredTool
    from agent.harness.web_native_worker import prepare_tool_arguments

    def compute(sequence: str, out_dir: str | None = None, output_file: str | None = None):
        """Calculate sequence properties and write an optional result file."""
        return sequence

    tool = StructuredTool.from_function(compute, name='calculate_new_property')
    for session in ('one', 'two'):
        workspace = tmp_path/session; workspace.mkdir()
        args = prepare_tool_arguments(tool.name, tool, {'sequence':'ACDE'}, workspace, lambda v:v)
        assert Path(args['out_dir']).is_relative_to(workspace)
        explicit = prepare_tool_arguments(tool.name, tool, {'sequence':'ACDE','output_file':'/workspace/result.json'}, workspace, lambda v:v)
        assert explicit['output_file'] == str(workspace/'result.json')
        with pytest.raises(ValueError, match='this conversation'):
            prepare_tool_arguments(tool.name, tool, {'sequence':'ACDE','out_dir':str(tmp_path/'other')}, workspace, lambda v:v)


@pytest.mark.asyncio
@pytest.mark.parametrize('harness', ['kimi-code', 'deepseek-harness', 'codex', 'claude-code'])
async def test_shared_protocol_dispatches_to_original_tool_and_records_real_identity(tmp_path, monkeypatch, harness):
    import httpx
    from langchain_core.tools import StructuredTool
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Profile, Budget, ModelBinding
    from mcp_tools.scheduler import catalog as scheduler
    from mcp_tools.tools import runtime_tool_policy

    calls, events = [], []
    async def compute(sequence: str):
        calls.append(sequence)
        return {'length': len(sequence)}
    tool = StructuredTool.from_function(coroutine=compute, name='protein_length',
              description='Calculate sequence length.\n\n' + 'Detailed notes. '*1000,
              metadata={'workspace_relative_paths': True})
    workspace = tmp_path/'work'; workspace.mkdir()
    control = tmp_path/'control'; control.mkdir()
    monkeypatch.setattr(scheduler, 'native_tools', lambda: {})
    monkeypatch.setattr(runtime_tool_policy, 'assert_tool_allowed', lambda name: None)

    async def adapter(ctx):
        from agent.harness.capability_boundary import CAPABILITY_BOUNDARY
        assert CAPABILITY_BOUNDARY in ctx.task.instructions
        connection = json.loads((ctx.task.capability/'connection.json').read_text())
        async with httpx.AsyncClient(trust_env=False) as client:
            async def rpc(method, params=None):
                return (await client.post(connection['url'], headers={'Authorization': 'Bearer '+connection['token']},
                         json={'jsonrpc':'2.0','id':1,'method':method,'params':params or {}})).json()
            async def call(name, args):
                return await rpc('tools/call', {'name': name, 'arguments': args})
            listing = (await rpc('tools/list'))['result']['tools']
            assert len(listing) == 5 and 'protein_length' not in {r['name'] for r in listing}
            assert len(json.dumps(listing)) < 6000
            assert 'error' in await call('protein_length', {'sequence':'ACDE'})
            assert 'error' in await call('call_tool', {'name':'protein_length','arguments':{'sequence':'ACDE'}})
            found = json.loads((await call('search_tools', {'query':'protein_length'}))['result']['content'][0]['text'])
            assert found['tools'][0] == {'name':'protein_length','description':'Calculate sequence length.'}
            described = json.loads((await call('describe_tool', {'name':'protein_length'}))['result']['content'][0]['text'])
            assert described['inputSchema']['required'] == ['sequence']
            result = await call('call_tool', {'name':'protein_length','arguments':{'sequence':'ACDE'}})
            assert json.loads(result['result']['content'][0]['text']) == {'length':4}
            return {'status':'completed','text':'The sequence has four residues.'}
    monkeypatch.setattr(worker,'builtin_registry',lambda:SimpleNamespace(resolve=lambda *args:(SimpleNamespace(run=adapter),None)))
    profile=Profile('science_agent',harness,ModelBinding('fixture','model','http://127.0.0.1:1/v1'),Budget(tool_calls=12))
    request={'profile':profile.public(),'workspace':str(workspace),'control':str(control),'session_id':'fixture',
             'runtimes':{},'base_url':'http://127.0.0.1:1/v1','token':'fixture','text':'Count sequence residues'}
    await worker.run(request,tools=[tool],event_sink=events.append)
    assert calls == ['ACDE']
    assert any(e.get('tool')=='protein_length' and e.get('status')=='completed' for e in events)
    assert any(e.get('kind')=='completed' for e in events)
