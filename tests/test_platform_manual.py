import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_policy(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path/'runtime'))
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:False)


def test_manual_wiki_and_legacy_files_share_one_source():
    spec=importlib.util.spec_from_file_location('sync_platform_docs',ROOT/'scripts/sync_platform_docs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name,text in module.generated_files().items():
        assert (ROOT/name).read_text(encoding='utf-8')==text,name
    manifest=json.loads((ROOT/'docs/manual/catalog.json').read_text())
    assert len(manifest['sections'])==13
    keys={s['key'] for s in manifest['sections']}
    import re
    for item in manifest['sections']:
        for lang in ['en','zh']:
            content=(ROOT/'docs/manual'/item['files'][lang]).read_text()
            assert content.startswith('# ')
            assert set(re.findall(r'\(manual:([a-z-]+)\)',content))<=keys


@pytest.mark.parametrize('query,expected,lang',[
    ('How do I upload files into Library?', 'library','en'),
    ('线上训练为什么只能 ESM2-8M？', 'training','zh'),
    ('Scientist research plan independent review', 'scientist','en'),
    ('三个模式共享每天几个对话额度', 'settings','zh'),
    ('workflow save connect cards', 'workflows','en'),
])
def test_search_returns_citable_relevant_tutorials(query,expected,lang):
    from mcp_tools.tools.platform_manual import query_manual
    result=query_manual(query,limit=5)
    assert result['language']==lang
    assert expected in {row['section'] for row in result['results']}
    for row in result['results']:
        assert row['url'].startswith('/'+lang+'/manual/') and '#header-' in row['url']
        assert len(row['content_sha256'])==64
        assert row['citation_markdown'].endswith(']('+row['url']+')')
        assert 'https://' not in row['citation_markdown']
        assert row['content']


def test_legacy_section_reads_current_content_and_response_is_bounded():
    from mcp_tools.tools.platform_manual import query_manual
    result=query_manual(section='quick-tools',language='zh',limit=8)
    assert {r['section'] for r in result['results']}=={'models'}
    assert sum(len(r['content']) for r in result['results'])<=14000
    assert len(query_manual()['catalog'])==13
    assert not query_manual('zzzzunknownword')['results']


@pytest.mark.parametrize('kwargs',[{'section':'../../.env'},{'section':'/etc/passwd'},{'limit':999},{'language':'../../'},{'query':'x'*1001}])
def test_manual_cannot_read_arbitrary_paths_or_unbounded_content(kwargs):
    from mcp_tools.tools.platform_manual import query_manual
    with pytest.raises(ValueError):query_manual(**kwargs)


def test_symlink_cannot_expose_private_files(tmp_path, monkeypatch):
    from mcp_tools.tools import platform_manual as manual
    docs=tmp_path/'docs';docs.mkdir();secret=tmp_path/'secret';secret.write_text('private')
    (docs/'catalog.json').symlink_to(secret)
    monkeypatch.setattr(manual,'ROOT',docs)
    with pytest.raises(ValueError):manual.query_manual()


@pytest.mark.parametrize('online',[False,True])
def test_tool_available_in_both_modes_but_respects_admin_switch(online,monkeypatch):
    from mcp_tools.tools.platform_manual import query_manual
    from mcp_tools.tools.capability_policy import CapabilityStore, CapabilityDisabled
    from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:online)
    assert is_tool_allowed('query_platform_manual')
    assert query_manual(section='training')['mode']==('online' if online else 'local')
    CapabilityStore().set('tool:query_platform_manual',False,0,{},'test-admin')
    with pytest.raises(CapabilityDisabled):query_manual(section='training')


@pytest.mark.asyncio
async def test_mcp_and_agent_read_identical_tutorials():
    from fastmcp import Client, FastMCP
    from mcp_tools.tools.platform_manual import register_manual_tool
    from mcp_tools.tools.discovery.tools_agent import query_platform_manual_tool
    server=FastMCP('manual-test');register_manual_tool(server)
    args={'query':'online credits daily limit','language':'en'}
    direct=json.loads(query_platform_manual_tool.invoke(args))
    async with Client(server) as client:
        catalog=await client.list_tools()
        assert set(catalog[0].inputSchema['properties'])=={'query','section','language','limit'}
        result=await client.call_tool('query_platform_manual',args)
        assert result.data==direct


def test_harness_guidance_uses_registered_manual_name_and_inline_tool():
    from agent.harness.capability_guidance import discovery_instructions
    from mcp_tools.scheduler.catalog import classify
    from mcp_tools.tools.tools_agent_hub import ALL_TOOLS,MCP_TOOLS,FASTAPI_TOOLS
    name='mcp__venusfactory__query_platform_manual'
    assert name in discovery_instructions([name])
    assert 'query_platform_manual' not in discovery_instructions(['search_research_tools'])
    for group in [ALL_TOOLS,MCP_TOOLS,FASTAPI_TOOLS]:
        assert sum(t.name=='query_platform_manual' for t in group)==1
    assert classify('query_platform_manual')['execution']=='inline'
    assert classify('query_platform_manual')['category']=='platform_help'
