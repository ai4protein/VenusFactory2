"""Public RSI discovery: bounded pages, curator ownership and live model switches."""
import json
import pytest
from fastmcp import FastMCP, Client
from agent.harness import research_workspace as rw
from agent.harness import capability_catalog as index
from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools
from mcp_tools.tools import capability_policy as policy
from mcp_tools.tools.discovery import service as discovery


@pytest.mark.asyncio
async def test_mcp_passes_page_offset_to_registry(monkeypatch):
    seen=[]
    def search(query='', limit=10, library=None, offset=0):
        seen.append((query,limit,offset))
        return {'total':13,'items':[{'id':'page-two'}],'next_offset':None}
    monkeypatch.setattr(rw,'search_capabilities',search)
    server=FastMCP('registry pagination fixture')
    register_discovery_tools(server)
    async with Client(server) as client:
        result=await client.call_tool('search_generated_capabilities',{'query':'solubility','limit':3,'offset':10})
    assert result.data['items'][0]['id']=='page-two'
    assert seen==[('solubility',3,10)]


@pytest.fixture
def metadata_catalog(tmp_path, monkeypatch):
    library=tmp_path/'catalog'
    monkeypatch.setattr(rw,'LIBRARY',library)
    switches=policy.CapabilityStore(path=tmp_path/'switches.sqlite3')
    monkeypatch.setattr(policy,'CapabilityStore',lambda:switches)
    rows=[]
    for i in range(10003):
        identity=f'{i:064x}'
        card={'id':identity,'name':f'Synthetic model {i}','description':'Enzyme solubility fixture',
              'version':'1.0','kind':'model','author':'Fixture','author_email':'fixture@example.org',
              'review_status':'approved','visibility':'public' if i!=10000 else 'private',
              'publication':{'policy':'operator_selected.v1' if i!=10002 else 'unreviewed'},
              'scope':{'proteins':['fixture enzyme'],'properties':['solubility'],'scenarios':['test only']},
              'validation':{'private_details':['not for search']*80}}
        rows.append((identity,'goal-'+str(i),card['name'],'1.0','enzyme solubility fixture',json.dumps(card),'fixture',float(i)))
    with index.database(library) as db:
        db.executemany('INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)',rows)
        db.executemany('INSERT INTO capability_families VALUES (?,?)',[(f'{0:064x}','family'),(f'{10001:064x}','family')])
        db.execute('INSERT INTO capability_heads VALUES (?,?)',('family',f'{0:064x}'))
    for i in (42,43):
        identity=f'{i:064x}'
        switches.set('model:capability:'+identity,False,0,
                     {'id':'capability:'+identity,'adapter':'capability','tools':['run_generated_capability']},'fixture')
    def no_inventory(*args,**kwargs):
        raise AssertionError('Search must not load the entire registry into Python')
    monkeypatch.setattr(rw,'capability_cards',no_inventory)
    monkeypatch.setattr(index,'cards',no_inventory)
    monkeypatch.setattr(discovery,'platform_models',lambda *a:[])
    monkeypatch.setattr(discovery,'model_cards',lambda:[])
    return library, switches


def test_ten_thousand_metadata_entries_are_paged_before_materialization(metadata_catalog, monkeypatch):
    library, switches = metadata_catalog
    decode=json.loads; decoded=[]
    def count_decode(value,*args,**kwargs):
        decoded.append(1)
        return decode(value,*args,**kwargs)
    monkeypatch.setattr(index.json,'loads',count_decode)
    first=rw.search_capabilities('enzyme solubility',3,library=library)
    second=rw.search_capabilities('enzyme solubility',3,library=library,offset=3)
    assert first['total']==second['total']==9998
    assert [item['id'] for item in first['items']]==[f'{i:064x}' for i in (9999,9998,9997)]
    assert [item['id'] for item in second['items']]==[f'{i:064x}' for i in (9996,9995,9994)]
    assert first['next_offset']==3 and second['next_offset']==6
    assert len(decoded)<=10
    assert all('validation' not in item and 'publication' not in item for item in first['items'])
    assert len(json.dumps(first))<2500
    last=rw.search_capabilities('',3,library=library,offset=9996)
    assert len(last['items'])==2 and last['next_offset'] is None


@pytest.mark.parametrize('query', ['', 'predict enzyme solubility', '预测可溶性'])
def test_unified_search_loads_only_the_selected_public_page(metadata_catalog, monkeypatch, query):
    library, switches = metadata_catalog
    decode = json.loads
    decoded = []

    def count_decode(value, *args, **kwargs):
        decoded.append(1)
        return decode(value, *args, **kwargs)

    monkeypatch.setattr(index.json, 'loads', count_decode)
    first = discovery.search(query, kind='models', limit=3, tool_catalog=[])
    second = discovery.search(query, kind='models', limit=3, offset=3, tool_catalog=[])
    assert first['matching_models'] == second['indexed_models'] == 9998
    assert [item['entry_id'] for item in first['results']] == [f'{i:064x}' for i in (0, 1, 2)]
    assert [item['entry_id'] for item in second['results']] == [f'{i:064x}' for i in (3, 4, 5)]
    assert first['next_offset'] == 3 and second['next_offset'] == 6
    assert len(decoded) <= 10
    assert len(json.dumps(first)) < 3500
    assert all('validation' not in item and 'publication' not in item for item in first['results'])
    assert all(item['kind'] == 'research_capability' for item in first['results'])
    last = discovery.search(query, kind='models', limit=3, offset=9996, tool_catalog=[])
    assert len(last['results']) == 2 and last['next_offset'] is None
    switches.set('tool:run_generated_capability', False, 0, {}, 'fixture')
    monkeypatch.setattr(index, 'ranked_references', lambda *a, **kw: pytest.fail('Disabled catalog read'))
    disabled = discovery.search(kind='models', tool_catalog=[])
    assert disabled['results'] == [] and disabled['indexed_models'] == 0


def test_unified_ranking_merges_families_and_preserves_method_identity(tmp_path, monkeypatch):
    from mcp_tools.tools.discovery.matching import query_groups, relevance
    library = tmp_path/'catalog'
    monkeypatch.setattr(rw, 'LIBRARY', library)
    switches = policy.CapabilityStore(path=tmp_path/'switches.sqlite3')
    monkeypatch.setattr(policy, 'CapabilityStore', lambda:switches)
    monkeypatch.setattr(discovery, 'model_cards', lambda:[])
    records = [
        ('a', 'Solubility assay', 'method', 'Measure enzyme response'),
        ('b', 'Enzyme estimator', 'model', 'Solubility '*100),
        ('c', 'Solubility predictor', 'model', 'Enzyme response'),
        ('d', 'Straße assay', 'method', 'Unicode name'),
    ]
    with index.database(library) as db:
        for identity, name, kind, description in records:
            card = {'id':identity, 'name':name, 'description':description, 'kind':kind,
                    'visibility':'public', 'publication':{'policy':'operator_selected.v1'},
                    'scope':{'proteins':['enzyme'], 'properties':['solubility'],
                             'scenarios':['rare buffer condition', 'x'*90, 'two', 'three']},
                    'validation':{'private_details':'not part of discovery'}}
            search = (name+' '+description+' enzyme solubility rare buffer condition').casefold()
            db.execute('INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)',
                       (identity,identity,name,'1.0',search,json.dumps(card),'fixture',0))
    tool = {'name':'solubility_tool', 'description':'Analyze an enzyme', 'parameters':{}}
    platform = {'kind':'platform_model', 'entry_id':'platform', 'name':'Platform model'}
    def platform_models(query, names):
        score = relevance(query_groups(query), 'Enzyme model', 'Predict solubility')
        return [(score, 'platform', platform)] if score is not None else []
    monkeypatch.setattr(discovery, 'platform_models', platform_models)
    ranked = discovery.search('solubility', tool_catalog=[tool], limit=30)
    assert [r.get('entry_id', r['name']) for r in ranked['results']] == ['a', 'c', 'solubility_tool', 'b', 'd', 'platform']
    for query in ('solubility', ''):
        full = discovery.search(query, tool_catalog=[tool], limit=30)
        pages = [discovery.search(query, tool_catalog=[tool], limit=2, offset=o) for o in range(0, 6, 2)]
        assert [r for page in pages for r in page['results']] == full['results']
        assert all(page['indexed_models'] == 5 for page in pages)
    predicted = discovery.search('预测可溶性', kind='models', tool_catalog=[])
    assert {r['entry_id'] for r in predicted['results']} == {'b','c','platform'}
    scoped = discovery.search('rare buffer condition', kind='models', tool_catalog=[])
    assert {r['entry_id'] for r in scoped['results']} == {'a','b','c','d'}
    assert all(r['scope_truncated'] and len(r['scope']['scenarios']) == 3 for r in scoped['results'])
    assert next(r for r in ranked['results'] if r.get('entry_id') == 'b')['description'] == ('Solubility '*100)[:240]
    assert discovery.search('strasse', kind='models', tool_catalog=[])['results'][0]['entry_id'] == 'd'


def test_selected_page_rechecks_publication_after_index_read(tmp_path, monkeypatch):
    library = tmp_path/'catalog'
    monkeypatch.setattr(rw, 'LIBRARY', library)
    monkeypatch.setattr(discovery, 'platform_models', lambda *a:[])
    monkeypatch.setattr(discovery, 'model_cards', lambda:[])
    switches = policy.CapabilityStore(path=tmp_path/'switches.sqlite3')
    monkeypatch.setattr(policy, 'CapabilityStore', lambda:switches)
    card = {'id':'withdrawn', 'name':'Withdrawn publication', 'kind':'model',
            'description':'Solubility', 'visibility':'public',
            'publication':{'policy':'operator_selected.v1'}}
    with index.database(library) as db:
        db.execute('INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)',
                   ('withdrawn','topic',card['name'],'1.0','solubility',json.dumps(card),'fixture',0))
    project = index.discovery_cards
    def withdraw(*args, **kwargs):
        with index.database(library) as db:
            db.execute("UPDATE capabilities SET card=json_set(card,'$.visibility','private')")
        return project(*args, **kwargs)
    monkeypatch.setattr(index, 'discovery_cards', withdraw)
    assert discovery.search(kind='models', tool_catalog=[])['results'] == []


@pytest.mark.asyncio
async def test_reviewed_operator_release_is_discovered_invoked_and_revoked_over_mcp(tmp_path, monkeypatch):
    from functools import partial
    from fastmcp.exceptions import ToolError
    from agent.harness.capability_review import review_proposal
    from agent.harness.curated_publication import publish_candidate
    from mcp_tools.tools import runtime_tool_policy
    from mcp_tools.tools.discovery.research_capabilities import search_generated_capabilities
    from test_capability_admission import candidate, spec, reviewer
    shared=tmp_path/'shared'
    monkeypatch.setattr(rw,'LIBRARY',shared)
    monkeypatch.setattr(runtime_tool_policy,'is_online_mode',lambda:False)
    switches=policy.CapabilityStore(path=tmp_path/'switches.sqlite3')
    monkeypatch.setattr(policy,'CapabilityStore',lambda:switches)
    monkeypatch.setattr(discovery,'model_cards',lambda:[])
    monkeypatch.setattr(discovery,'platform_models',lambda *a:[])
    service=await candidate(tmp_path,spec())
    bundle=service.proposals[0]
    receipt=await review_proposal(bundle,reviewer=reviewer,control=tmp_path/'review',runner=rw.run_python)
    assert receipt['status']=='approved'
    rw.publish_proposals([bundle],'fixture-goal',tmp_path/'private')
    for name in ('search_capabilities','describe_capability','run_capability'):
        monkeypatch.setattr(rw,name,partial(getattr(rw,name),library=shared))
    server=FastMCP('operator release fixture')
    register_discovery_tools(server)
    async with Client(server) as client:
        assert (await client.call_tool('search_generated_capabilities',{})).data['total']==0
        assert (await client.call_tool('search_research_tools',{'kind':'models'})).data['matching_models']==0
        card=publish_candidate(bundle,'fixture-goal',library=shared,author_email='author@example.org',curator='test operator')
        monkeypatch.setattr(runtime_tool_policy,'is_online_mode',lambda:True)
        page=(await client.call_tool('search_generated_capabilities',{'query':'length'})).data
        native=json.loads(search_generated_capabilities.invoke({'query':'length'}))
        assert native==page
        assert page['items'][0]['id']==card['id']
        assert page['items'][0]['author_email']=='author@example.org'
        assert 'validation' not in page['items'][0]
        generic=(await client.call_tool('search_research_tools',{'query':'length','kind':'models'})).data
        assert generic['results'][0]['entry_id']==card['id']
        assert generic['results'][0]['author_email']=='author@example.org'
        assert generic['indexed_models']==1 and 'validation' not in generic['results'][0]
        detail=(await client.call_tool('describe_generated_capability',{'capability_id':card['id']})).data
        assert detail['input_schema']['required']==['sequence']
        assert detail['review_status']=='approved'
        result=(await client.call_tool('run_generated_capability',{'capability_id':card['id'],'inputs':{'sequence':'ACDE'}})).data
        assert result=={'value':4}
        with pytest.raises(ToolError):
            await client.call_tool('run_generated_capability',{'capability_id':card['id'],'inputs':{'sequence':3}})
        switches.set('model:capability:'+card['id'],False,0,
                     {'id':'capability:'+card['id'],'adapter':'capability','tools':['run_generated_capability']},'test operator')
        assert (await client.call_tool('search_generated_capabilities',{})).data['total']==0
        assert (await client.call_tool('search_research_tools',{'kind':'models'})).data['matching_models']==0
        with pytest.raises(ToolError):
            await client.call_tool('run_generated_capability',{'capability_id':card['id'],'inputs':{'sequence':'ACDE'}})
