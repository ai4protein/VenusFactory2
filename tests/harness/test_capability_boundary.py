from pathlib import Path
import ast
import pytest
from types import SimpleNamespace

from agent.harness.capability_boundary import CAPABILITY_BOUNDARY
from agent.harness.capability_guidance import discovery_instructions
from agent.harness.tool_disclosure import ToolDisclosure


@pytest.mark.parametrize('tools', [[], ['query_platform_manual'], ['search_research_tools'],
    ['mcp__venusfactory__query_platform_manual','mcp__venusfactory__search_research_tools']])
def test_direct_and_namespaced_catalogs_always_keep_capability_boundary(tools):
    assert CAPABILITY_BOUNDARY in discovery_instructions(tools)


def test_progressive_search_keeps_remote_service_boundary_in_short_description():
    root=Path(__file__).resolve().parents[2]
    files={'mcp_tools/tools/database/tools_agent.py':'download_mmseqs2_homologs_by_sequence_tool',
           'mcp_tools/tools/database/tools_mcp.py':'mcp_download_mmseqs2_homologs',
           'mcp_tools/tools/research/tools_agent.py':'query_public_protein_service_tool'}
    for filename, function in files.items():
        module=ast.parse((root/filename).read_text())
        func=next(n for n in module.body if getattr(n,'name','')==function)
        description=ast.get_docstring(func)
        tool=SimpleNamespace(name=function,description=description)
        disclosure=ToolDisclosure({function:tool},schema=lambda t:{},allowed=lambda t:True)
        result=disclosure.metadata('search_tools', {'query':function})['tools'][0]['description']
        assert len(result)<=180
        assert ('external ColabFold' in result and 'does not execute local MMseqs2' in result) or ('AlphaFold DB' in result and 'not new predictions' in result)


def test_manual_can_explain_missing_inference_without_claiming_query_is_prediction(monkeypatch):
    from mcp_tools.tools.platform_manual import query_manual
    from mcp_tools.tools import runtime_tool_policy as policy
    monkeypatch.setattr(policy,'assert_tool_allowed',lambda name:None)
    for lang,query,expected in [('en','AlphaFold MMseqs2 platform capability','does not expose an AlphaFold inference'),
                                ('zh','AlphaFold MMseqs2 平台能力边界','平台未提供对新序列运行 AlphaFold')]:
        result=query_manual(query=query,section='data',language=lang,limit=8)
        assert any(expected in row['content']for row in result['results'])
        assert all(row['url'].startswith('/'+lang+'/manual/') for row in result['results'])


def test_native_history_cannot_replace_current_host_capability_facts():
    from agent.harness.web_native_worker import native_prompt
    from agent.harness.capability_boundary import CURRENT_CAPABILITY_FACTS
    old='I can run AlphaFold and local MMseqs2.'
    rendered=native_prompt({'messages':[{'role':'assistant','content':old}],
        'memory':'Previously offered structure prediction.', 'text':'Was that accurate?'})
    assert old in rendered  # Preserve the conversation, including what needs correcting.
    assert rendered.index(old) < rendered.index(CURRENT_CAPABILITY_FACTS) < rendered.index('## Current user request')


@pytest.mark.asyncio
async def test_grounding_notes_do_not_leak_into_persisted_user_messages(tmp_path, monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict, messages_from_dict
    import langchain.agents
    from agent.harness import web_langgraph_worker as worker
    from agent.harness.capability_boundary import CURRENT_CAPABILITY_FACTS
    events=[]
    class Agent:
        async def astream(self, inputs, **kwargs):
            assert CURRENT_CAPABILITY_FACTS in inputs['messages'][-1].content
            yield 'values', {'messages':inputs['messages']+[AIMessage(content='My earlier claim was inaccurate.')]}
    def create_agent(model, **kwargs):
        assert CAPABILITY_BOUNDARY in kwargs['system_prompt']
        return Agent()
    monkeypatch.setattr(langchain.agents,'create_agent',create_agent)
    monkeypatch.setattr(worker,'emit',events.append)
    history=messages_to_dict([HumanMessage(content='Hello'),AIMessage(content='Earlier reply')])
    await worker.run({'workspace':str(tmp_path),'text':'Is that correct?','lang':'en','session_id':'test',
        'budget':{'tool_calls':2},'messages':history,'memory':'Old context'}, tools=[], model=object())
    result=next(e for e in events if e['kind']=='completed')
    saved=messages_from_dict(result['messages'])
    assert [m.content for m in saved]==['Hello','Earlier reply','Is that correct?','My earlier claim was inaccurate.']


@pytest.mark.parametrize('filename', ['tools_agent.py', 'tools_mcp.py'])
def test_alphafold_tools_keep_retrieval_and_supplied_result_analysis_distinct(filename):
    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / 'mcp_tools/tools/database' / filename).read_text())
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and 'alphafold' in node.name]
    assert len(functions) == 4
    for function in functions:
        tool = SimpleNamespace(name=function.name, description=ast.get_docstring(function))
        disclosure = ToolDisclosure({tool.name: tool}, schema=lambda t: {}, allowed=lambda t: True)
        brief = disclosure.metadata('search_tools', {'query': tool.name})['tools'][0]['description']
        assert 'does not run AlphaFold inference' in brief
        if function.name.startswith(('analyze_', 'mcp_analyze_')):
            assert brief.startswith('Analyze') and 'supplied' in brief
            assert 'pLDDT' in brief if 'plddt' in function.name else 'PAE' in brief
        else:
            assert brief.startswith('Retrieve existing AlphaFold DB')
