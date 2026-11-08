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
            assert (await call('call_tool', {'name':'protein_length','arguments':{'sequence':'ACDE'}}))['result']['isError'] is True
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

def test_overview_stays_bounded_and_tracks_enabled_tools():
    enabled = {'protein_model_0'}
    c = catalog(10000, enabled)
    overview = c.overview()
    assert sum(g['tool_count'] for g in overview['categories']) == 1
    assert len(json.dumps(overview)) < 1500
    assert 'inputSchema' not in json.dumps(overview)
    enabled.clear()
    assert c.overview()['categories'] == []
    assert c.metadata('search_tools', {})['categories'] == []

def grouped_catalog(enabled=None, schema=None):
    descriptions = {
        'query_uniprot': 'Find protein records.\n\nArgs:\nsecret_schema_word',
        'predict_custom_model': 'Predict protein properties.',
        'calculate_protein': 'Calculate protein properties.',
        'search_generated_capabilities': 'Find protein research methods.',
        'list_skills': 'Find scientific workflow guides.\n\nprivate_skill_body',
    }
    categories = ['databases', 'factory_models', 'compute', 'rsi', 'guides']
    tools = {name: SimpleNamespace(name=name, description=desc, metadata={'discovery_category': cat})
             for (name, desc), cat in zip(descriptions.items(), categories)}
    return ToolDisclosure(tools, schema=schema or (lambda t: {'type': 'object', 'properties': {}}),
                          allowed=lambda t: enabled is None or t.name in enabled)


def test_initial_overview_contains_categories_only_and_broad_search_narrows():
    c = grouped_catalog()
    for args in ({}, {'query': 'protein'}):
        result = c.metadata('search_tools', args)
        assert result['level'] == 'categories' and not result.get('tools')
        encoded = json.dumps(result)
        assert not any(name in encoded for name in c.mapping)
        assert 'secret_schema_word' not in encoded and 'private_skill_body' not in encoded
    assert len(json.dumps(catalog(10000).overview())) < 1500


def test_category_filter_and_short_search_never_read_schemas():
    def must_not_read(tool):
        raise AssertionError('Schema read before selecting a capability')
    c = grouped_catalog(schema=must_not_read)
    result = c.metadata('search_tools', {'category': 'databases', 'query': 'protein'})
    assert result['level'] == 'capabilities' and result['category'] == 'databases'
    assert result['tools'] == [{'name': 'query_uniprot', 'description': 'Find protein records.'}]
    assert c.metadata('search_tools', {'category': 'databases', 'query': 'secret_schema_word'})['tools'] == []
    fallback = c.metadata('search_tools', {'category': 'guides', 'query': 'private_skill_body'})
    assert fallback['match_mode'] == 'category_navigation'
    assert 'private_skill_body' not in json.dumps(fallback)
    assert c.metadata('search_tools', {'category': 'guides'})['tools'][0]['name'] == 'list_skills'


@pytest.mark.parametrize('args', [{'category': []}, {'category': None}, {'category': True}])
def test_category_validation(args):
    with pytest.raises(DisclosureError):
        grouped_catalog().metadata('search_tools', args)


def test_same_discovery_flow_for_local_and_online_with_live_permissions(monkeypatch):
    enabled = {'query_uniprot', 'calculate_protein'}
    results = []
    for mode in ('local', 'online'):
        monkeypatch.setenv('WEBUI_V2_MODE', mode)
        c = grouped_catalog(enabled)
        results.append((c.overview(), c.metadata('search_tools', {'category': 'compute'})))
        c.metadata('describe_tool', {'name': 'calculate_protein'})
        enabled.remove('calculate_protein')
        assert c.metadata('search_tools', {'category': 'compute'})['tools'] == []
        with pytest.raises(DisclosureError, match='unavailable'):
            c.resolve('call_tool', {'name': 'calculate_protein', 'arguments': {}})
        enabled.add('calculate_protein')
    assert results[0] == results[1]


def test_requested_platform_model_alias_remains_searchable_in_category(monkeypatch):
    from mcp_tools.tools.discovery import platform_models as registry
    c = ToolDisclosure({'zero_shot_mutation_structure_prediction': SimpleNamespace(
        name='zero_shot_mutation_structure_prediction', description='Score sequence mutations.')},
        schema=lambda t: {}, allowed=lambda t: True)
    monkeypatch.setattr(registry, 'platform_models', lambda query, names: [
        (10, 'prosst', {'name': 'ProSST-2048', 'tools': ['zero_shot_mutation_structure_prediction']})])
    result = c.metadata('search_tools', {'category': 'compute', 'query': 'ProSST-2048'})
    assert result['tools'][0]['name'] == 'zero_shot_mutation_structure_prediction'
    assert 'ProSST-2048' in result['tools'][0]['description']
    assert c.metadata('search_tools', {'category': 'databases', 'query': 'ProSST-2048'})['tools'] == []

@pytest.mark.parametrize("query", ["molecular weight isoelectric point protein sequence", "physicochemical properties"])
def test_physchem_is_discoverable_from_its_brief_description(query):
    import ast
    from pathlib import Path
    source = Path(__file__).resolve().parents[2] / "mcp_tools/tools/predict/tools_agent.py"
    func = next(n for n in ast.parse(source.read_text()).body
                if getattr(n, "name", "") == "calculate_physchem_from_sequence_tool")
    tool = SimpleNamespace(name="calculate_physchem_from_sequence", description=ast.get_docstring(func))
    c = ToolDisclosure({tool.name: tool}, schema=lambda t: {}, allowed=lambda t: True)
    assert c.metadata("search_tools", {"category": "compute", "query": query})["tools"][0]["name"] == tool.name

def test_guessed_category_returns_only_allowed_navigation_and_can_recover():
    c = grouped_catalog({'query_uniprot'})
    result = c.metadata('search_tools', {'category': 'sequence_analysis'})
    assert result['level'] == 'categories' and result['selection_error']
    assert result['tools'] == [] and not c.inspected
    assert [row['id'] for row in result['categories']] == ['databases']
    assert 'calculate_protein' not in json.dumps(result)
    assert c.metadata('search_tools', {'category':'databases','query':'protein'})['tools']
    category = c.surface()[0]['inputSchema']['properties']['category']
    assert category['enum'] == ['', 'databases']


def test_directly_exposed_checkpoint_can_use_either_dispatch_path_and_still_be_revoked():
    name = "submit_research_checkpoint"
    tool = SimpleNamespace(name=name, description="Save the current research result.")
    enabled = {name}
    c = ToolDisclosure({name:tool}, schema=lambda _:{"type":"object"},
                       allowed=lambda t:t.name in enabled, direct={name})
    assert name in {row["name"] for row in c.surface()}
    arguments = {"summary":"Measured a baseline"}
    assert c.resolve(name, arguments)[1:] == (tool, arguments)
    assert c.resolve("call_tool", {"name":name, "arguments":arguments})[1:] == (tool, arguments)
    assert c.metadata("describe_tool", {"name":name})["call_via"] == name
    enabled.clear()
    for method, args in [(name,arguments), ("call_tool",{"name":name,"arguments":arguments})]:
        with pytest.raises(DisclosureError, match="unavailable"):
            c.resolve(method,args)

@pytest.mark.parametrize("prefix",["mcp__venusfactory__","functions.mcp__venusfactory__"])
def test_harness_namespace_in_discovery_arguments_preserves_policy(prefix):
    enabled={"protein_model_0"}
    c=catalog(1,enabled)
    name=prefix+"protein_model_0"
    with pytest.raises(DisclosureError,match="describe_tool"):
        c.resolve("call_tool",{"name":name,"arguments":{}})
    assert c.metadata("describe_tool",{"name":name})["name"]=="protein_model_0"
    assert c.resolve("call_tool",{"name":name,"arguments":{}})[0]=="protein_model_0"
    assert c.metadata("search_tools",{"query":name})["tools"][0]["name"]=="protein_model_0"
    with pytest.raises(DisclosureError,match="unavailable"):
        c.metadata("describe_tool",{"name":"mcp__other_server__protein_model_0"})
    enabled.clear()
    with pytest.raises(DisclosureError,match="unavailable"):
        c.resolve("call_tool",{"name":name,"arguments":{}})

def test_cross_category_search_reports_matches_without_empty_tool_list():
    tools = {name: SimpleNamespace(name=name, description="ProSST-2048 capability.",
             metadata={"discovery_category": category})
             for name, category in (("score_variant", "compute"), ("fit_model", "training"))}
    enabled = set(tools)
    def no_schema(tool):
        raise AssertionError("Category lookup must not load a schema")
    c = ToolDisclosure(tools, schema=no_schema, allowed=lambda tool: tool.name in enabled)
    result = c.metadata("search_tools", {"query": "ProSST-2048"})
    assert result["level"] == "categories" and "tools" not in result
    assert result["match_status"] == "matches_found"
    assert result["matched_capabilities"] == 2
    assert "Found 2 enabled capabilities" in result["summary"]
    assert not any(name in json.dumps(result) for name in tools)
    assert not c.inspected
    for args in result["suggested_searches"]:
        assert args["query"] == "ProSST-2048"
        page = c.metadata("search_tools", args)
        assert len(page["tools"]) == 1
    enabled.clear()
    missing = c.metadata("search_tools", {"query": "ProSST-2048"})
    assert missing["match_status"] == "no_direct_match"
    assert missing["matched_capabilities"] == 0 and not missing["suggested_searches"]
    assert "not establish" in missing["summary"]

def test_unavailable_uninspected_names_do_not_suggest_an_impossible_describe_call():
    c = catalog(2, set())
    for name in ("protein_model_0", "guessed_name"):
        with pytest.raises(DisclosureError, match="unavailable") as error:
            c.resolve("call_tool", {"name":name, "arguments":{}})
        assert "Read describe_tool" not in str(error.value)


def test_unknown_category_with_named_query_recovers_without_expanding_schema():
    c = catalog(2, {'protein_model_0'})
    result = c.metadata('search_tools', {'category':'sequence_analysis','query':'protein_model_0'})
    assert result['level'] == 'capabilities'
    assert [row['name'] for row in result['tools']] == ['protein_model_0']
    assert result['requested_category'] == 'sequence_analysis'
    assert all(set(row) == {'name','description'} for row in result['tools'])
    assert not c.inspected
    with pytest.raises(DisclosureError, match='describe_tool'):
        c.resolve('call_tool', {'name':'protein_model_0','arguments':{'sequence':'AC'}})
    hidden = c.metadata('search_tools', {'category':'sequence_analysis','query':'protein_model_1'})
    assert not hidden.get('tools') and not c.inspected


@pytest.mark.parametrize('name',['search_tools','describe_tool','call_tool'])
@pytest.mark.parametrize('prefix',['','mcp__venusfactory__','functions.mcp__venusfactory__'])
def test_navigation_tools_can_describe_their_already_exposed_contract(name,prefix):
    c=catalog()
    result=c.metadata('describe_tool',{'name':prefix+name})
    public=next(tool for tool in c.surface() if tool['name']==name)
    assert result['name']==name and result['inputSchema']==public['inputSchema']
    assert result['call_via']==name
    assert not c.inspected
    with pytest.raises(DisclosureError,match='describe_tool'):
        c.resolve('call_tool',{'name':'protein_model_0','arguments':{'sequence':'AC'}})


def rsi_search_catalog(enabled=None, schema=None):
    descriptions = {
        "list_evolved_models": "Search Scientist model publications by text or engineering stage.",
        "predict_with_evolved_model": "Predict using a registered model.",
    }
    tools = {name: SimpleNamespace(name=name, description=description, metadata=None)
             for name, description in descriptions.items()}
    return ToolDisclosure(tools, schema=schema or (lambda t: {
        "type": "object", "properties": {"query": {"type": "string"}}
    }), allowed=lambda t: enabled is None or t.name in enabled)


def test_category_router_gives_explicit_next_action_without_loading_records():
    def no_schema(tool):
        pytest.fail("The index must not load a reader's contract yet")
    c = rsi_search_catalog(schema=no_schema)
    result = c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    assert result["records_searched"] is False
    assert result["next_action"] == {
        "name": "describe_tool", "arguments": {"name": "list_evolved_models"}
    }
    assert all(set(row) == {"name", "description"} for row in result["tools"])
    assert "inputSchema" not in result
    with pytest.raises(DisclosureError, match="describe_tool"):
        c.resolve("call_tool", {"name": "list_evolved_models", "arguments": {"query": "AT4G10320"}})


def test_selected_router_preserves_topic_and_refreshes_it_on_next_search():
    c = rsi_search_catalog()
    c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    description = c.metadata("describe_tool", {"name": "list_evolved_models"})
    assert description["suggested_arguments"] == {"query": "model AT4G10320"}
    c.metadata("search_tools", {"category": "rsi", "query": "solubility BRAF"})
    assert c.metadata("describe_tool", {"name": "list_evolved_models"})["suggested_arguments"] == {"query": "solubility BRAF"}


def test_navigation_hint_does_not_bypass_permission_or_invent_parameters():
    enabled = {"list_evolved_models", "predict_with_evolved_model"}
    c = rsi_search_catalog(enabled)
    c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    enabled.remove("list_evolved_models")
    with pytest.raises(DisclosureError, match="unavailable"):
        c.metadata("describe_tool", {"name": "list_evolved_models"})
    unavailable = c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    assert "next_action" not in unavailable
    c = rsi_search_catalog(schema=lambda t: {"type": "object", "properties": {"name": {"type": "string"}}})
    c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    assert "suggested_arguments" not in c.metadata("describe_tool", {"name": "list_evolved_models"})


def test_reader_hint_does_not_leak_into_a_new_unrelated_search():
    c = rsi_search_catalog()
    c.metadata("search_tools", {"category": "rsi", "query": "model AT4G10320"})
    assert c.metadata("describe_tool", {"name": "list_evolved_models"})["suggested_arguments"]
    c.metadata("search_tools", {"category": "rsi", "query": "list_evolved_models"})
    assert "suggested_arguments" not in c.metadata("describe_tool", {"name": "list_evolved_models"})
