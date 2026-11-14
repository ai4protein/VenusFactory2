"""RSI cases: bounded archives, real scientific review and retired chat discovery."""
import json

import pytest
from fastmcp import Client, FastMCP

from agent.harness import capability_catalog as index
from agent.harness import research_workspace as rw
from mcp_tools.tools import capability_policy as policy
from mcp_tools.tools.discovery import service as discovery
from mcp_tools.tools.discovery.tools_mcp import register_discovery_tools


@pytest.mark.asyncio
async def test_mcp_does_not_register_or_load_retired_rsi_discovery(monkeypatch):
    from fastmcp.exceptions import ToolError

    def no_registry(*args, **kwargs):
        pytest.fail('Retired conversational tools must not load the research registry')
    monkeypatch.setattr(rw, 'search_capabilities', no_registry)
    server = FastMCP('retired RSI discovery fixture')
    register_discovery_tools(server)
    async with Client(server) as client:
        names = {tool.name for tool in await client.list_tools()}
        retired = {'search_generated_capabilities', 'describe_generated_capability', 'run_generated_capability'}
        assert not names & retired
        with pytest.raises(ToolError):
            await client.call_tool('search_generated_capabilities', {'query':'solubility', 'limit':3, 'offset':10})


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


@pytest.mark.parametrize('mode', ['local', 'online'])
@pytest.mark.parametrize('query', ['', 'predict enzyme solubility', '预测可溶性'])
def test_unified_search_never_materializes_rsi_registry(metadata_catalog, monkeypatch, query, mode):
    monkeypatch.setenv('WEBUI_V2_MODE', mode)
    def no_registry(*args, **kwargs):
        pytest.fail('Conversational discovery must not scan RSI case metadata')
    for name in ('search', 'ranked_references', 'discovery_cards'):
        monkeypatch.setattr(index, name, no_registry)
    for offset in (0, 3, 9996):
        page = discovery.search(query, kind='models', limit=3, offset=offset, tool_catalog=[])
        assert page['results'] == []
        assert page['matching_models'] == page['indexed_models'] == 0
        assert page['next_offset'] is None


def test_plaza_paginates_ten_thousand_curated_cases_without_model_loading(metadata_catalog, monkeypatch):
    from backend.web_v2.rsi_plaza import RsiPlaza

    library, _ = metadata_catalog
    monkeypatch.setenv('WEBUI_V2_MODE', 'online')
    decode = json.loads
    decoded = []
    def count_decode(value, *args, **kwargs):
        decoded.append(1)
        return decode(value, *args, **kwargs)
    monkeypatch.setattr(index.json, 'loads', count_decode)
    plaza = RsiPlaza(None, library)
    monkeypatch.setattr(plaza, 'resolve', lambda *args: pytest.fail('Metadata pages must not load repositories'))
    first = plaza.listing(set(), query='enzyme solubility', limit=3)
    second = plaza.listing(set(), query='enzyme solubility', offset=3, limit=3)
    # Execution switches do not hide read-only case archives; private, unreviewed,
    # and superseded records remain excluded by the publication catalog.
    assert first['total'] == second['total'] == 10000
    assert [r['id'] for r in first['items']] == ['capability:' + f'{i:064x}' for i in (9999, 9998, 9997)]
    assert [r['id'] for r in second['items']] == ['capability:' + f'{i:064x}' for i in (9996, 9995, 9994)]
    assert first['next_offset'] == 3 and second['next_offset'] == 6
    assert len(decoded) <= 10
    assert len(json.dumps(first)) < 3500
    assert all('validation' not in item and 'publication' not in item for item in first['items'])
    last = plaza.listing(set(), limit=3, offset=9999)
    assert len(last['items']) == 1 and last['next_offset'] is None


def test_rsi_metadata_preserves_method_identity_without_entering_tool_ranking(tmp_path, monkeypatch):
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
    # Scientific metadata keeps names, method/model types and bounded scope cards.
    references = index.ranked_references(library, lambda name, text, kind: relevance(query_groups('solubility'), name, text), public_only=True)
    identities = [identity for identity, score in sorted(references, key=lambda r: (-r[1], r[0]))]
    assert identities == ['a', 'c', 'b', 'd']
    cards = index.discovery_cards(library, identities, public_only=True)
    assert cards['a']['kind'] == 'method' and cards['b']['kind'] == 'model'
    assert cards['b']['description'] == ('Solubility '*100)[:240]
    assert all(r['scope_truncated'] and len(r['scope']['scenarios']) == 3 for r in cards.values())
    unicode = index.ranked_references(library, lambda name, text, kind: relevance(query_groups('strasse'), name, text), public_only=True)
    assert [identity for identity, score in unicode if score is not None] == ['d']
    # Chat ranks only enabled existing tools and ordinary platform models.
    ranked = discovery.search('solubility', tool_catalog=[tool], limit=30)
    assert [r.get('entry_id', r['name']) for r in ranked['results']] == ['solubility_tool', 'platform']
    for query in ('solubility', ''):
        full = discovery.search(query, tool_catalog=[tool], limit=30)
        pages = [discovery.search(query, tool_catalog=[tool], limit=1, offset=o) for o in range(2)]
        assert [r for page in pages for r in page['results']] == full['results']
        assert all(page['indexed_models'] == 1 for page in pages)
    predicted = discovery.search('预测可溶性', kind='models', tool_catalog=[])
    assert {r['entry_id'] for r in predicted['results']} == {'platform'}
    assert discovery.search('rare buffer condition', kind='models', tool_catalog=[])['results'] == []
    assert discovery.search('strasse', kind='models', tool_catalog=[])['results'] == []


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
    references = index.ranked_references(library, lambda *args: 1, public_only=True)
    assert references == [('withdrawn', 1)]
    with index.database(library) as db:
        db.execute("UPDATE capabilities SET card=json_set(card,'$.visibility','private')")
    assert index.discovery_cards(library, [identity for identity, _ in references], public_only=True) == {}
    assert discovery.search(kind='models', tool_catalog=[])['results'] == []


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['tool', 'model'])
async def test_reviewed_operator_release_is_a_case_archive_and_revocation_is_live(tmp_path, monkeypatch, kind):
    import zipfile
    from fastmcp.exceptions import ToolError
    from test_capability_admission import candidate, model_candidate, reviewer, spec
    from agent.harness.capability_review import review_proposal
    from agent.harness.curated_publication import publish_candidate
    from backend.web_v2.rsi_plaza import RsiPlaza
    from mcp_tools.tools.discovery import generated_scope

    shared = tmp_path/'shared'
    monkeypatch.setattr(rw, 'LIBRARY', shared)
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    monkeypatch.setattr('backend.web_v2.rsi_legacy_cases.cards', lambda query='': [])
    switches = policy.CapabilityStore(path=tmp_path/'switches.sqlite3')
    monkeypatch.setattr(policy, 'CapabilityStore', lambda:switches)
    monkeypatch.setattr(discovery, 'model_cards', lambda:[])
    monkeypatch.setattr(discovery, 'platform_models', lambda *a:[])
    service = await model_candidate(tmp_path) if kind == 'model' else await candidate(tmp_path, spec())
    bundle = service.proposals[0]
    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path/'review', runner=rw.run_python)
    assert receipt['status'] == 'approved', receipt
    if kind == 'model':
        assert receipt['automatic']['evaluation']['training_reproduced']
        assert receipt['automatic']['evaluation']['score'] == 1
    private = tmp_path/'private'
    private_entry = rw.publish_proposals([bundle], 'fixture-goal', private)[0]
    assert index.require_registered(private, private_entry['id'], receipt['receipt_sha256'])['visibility'] == 'private'
    # The scientific engine retains offline reproduction without exposing an Agent tool.
    assert await rw.run_capability(private_entry['id'], {'sequence':'ACDE'}, library=private) == {'value':4}
    with pytest.raises(Exception):
        await rw.run_capability(private_entry['id'], {'sequence':3}, library=private)
    plaza = RsiPlaza(None, shared)
    assert plaza.listing(set())['items'] == []
    card = publish_candidate(bundle, 'fixture-goal', library=shared, author_email='author@example.org', curator='test operator')
    identity = 'capability:' + card['id']
    detail = plaza.detail(identity, set())
    assert detail['author_email'] == 'author@example.org' and detail['visibility'] == 'public'
    assert detail['id'] == plaza.listing(set())['items'][0]['id']
    assert {'README.md', 'REPORT.md', 'length.py'} <= {r['path'] for r in detail['files']}
    source = plaza.preview(identity, set(), 'length.py')['text']
    assert 'def run(inputs)' in source
    archive = plaza.download(identity, set())
    with archive, zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {r['path'] for r in detail['files']}
        assert z.read('length.py').decode() == source
        assert z.read('REPORT.md').decode() == plaza.preview(identity, set(), 'REPORT.md')['text']
        if kind == 'model': assert 'weights.json' in z.namelist()
    server = FastMCP('RSI release is a downloadable case')
    register_discovery_tools(server)
    async with Client(server) as client:
        assert not ({'search_generated_capabilities', 'describe_generated_capability', 'run_generated_capability'} & {t.name for t in await client.list_tools()})
        for mode in ('local', 'online'):
            monkeypatch.setenv('WEBUI_V2_MODE', mode)
            assert (await client.call_tool('search_research_tools', {'query':'length', 'kind':'models'})).data['matching_models'] == 0
            assert generated_scope.search('length')['items'] == []
            with pytest.raises(RuntimeError, match='RSI Plaza'): generated_scope.describe(card['id'])
            with pytest.raises(ToolError):
                await client.call_tool('run_generated_capability', {'capability_id':card['id'], 'inputs':{'sequence':'ACDE'}})
    # An execution switch cannot create or withdraw a read-only publication.
    switches.set('model:'+identity, False, 0, {'id':identity, 'adapter':'capability', 'tools':['run_generated_capability']}, 'test operator')
    assert plaza.detail(identity, set())['id'] == identity
    with index.database(shared) as db:
        db.execute("UPDATE capabilities SET card=json_set(card,'$.visibility','private') WHERE id=?", (card['id'],))
    assert plaza.listing(set())['items'] == []
    for call, arguments in ((plaza.detail, ()), (plaza.preview, ('length.py',)), (plaza.download, ())):
        with pytest.raises(ValueError, match='not been published by the operator'):
            call(identity, set(), *arguments)
    # Restore the separate compute switch for private offline reproduction.
    switches.set('model:'+identity, True, 1, {'id':identity, 'adapter':'capability', 'tools':['run_generated_capability']}, 'test operator')
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    assert generated_scope.search('length')['items'] == []
    # Withdrawing a case hides it without destroying the original research evidence.
    assert await rw.run_capability(private_entry['id'], {'sequence':'ACDE'}, library=private) == {'value':4}
    assert (shared/card['id']/'length.py').read_text() == source
