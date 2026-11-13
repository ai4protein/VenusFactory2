import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastmcp import Client
import httpx
import pytest

from mcp_tools.tools.research import knowledge_queries as k
from mcp_tools.tools.research.tools_agent import KNOWLEDGE_TOOLS
from mcp_tools.tools.research.tools_mcp import mcp


@pytest.mark.parametrize('schema,args', [
    (k.OpenTargetsInput, {'query': ''}),
    (k.OpenTargetsInput, {'query': 'BRCA2', 'limit': 100}),
    (k.OpenTargetsInput, {'query': 'BRCA2', 'identifier': 'ENSG00000139618'}),
    (k.OpenTargetsInput, {'operation': 'target', 'identifier': 'BRCA2'}),
    (k.OpenTargetsInput, {'operation': 'target', 'identifier': 'ENSMUSG00000041147'}),
    (k.OpenTargetsInput, {'operation': 'drug', 'identifier': 'CHEMBL25', 'page': 2}),
    (k.OpenTargetsInput, {'operation': 'drug', 'identifier': 'CHEMBL25', 'resource': 'drugs'}),
    (k.OpenTargetsInput, {'query': 'x', 'url': 'http://127.0.0.1'}),
    (k.EnsemblInput, {'operation': 'lookup', 'identifier': 'BRCA2'}),
    (k.EnsemblInput, {'identifier': '../secret'}),
    (k.EnsemblInput, {'identifier': 'BRCA2', 'species': 'https://host'}),
    (k.EnsemblInput, {'identifier': 'BRCA2', 'target_species': 'mus_musculus'}),
    (k.EnsemblInput, {'identifier': 'BRCA2', 'sequence_type': 'genomic'}),
    (k.EuropePMCInput, {'query': '   '}),
    (k.EuropePMCInput, {'query': 'protein', 'cursor': 'x' * 2001}),
    (k.PubChemInput, {'query': '-1', 'namespace': 'cid'}),
    (k.PubChemInput, {'query': 'aspirin\ncommand'}),
])
def test_queries_validate_identifiers_and_reject_arbitrary_network_or_commands(schema, args):
    with pytest.raises(ValueError):
        schema(**args)


@pytest.mark.asyncio
async def test_ot_search_uses_variables_and_retains_total_release_and_sources():
    obj = {'data': {'meta': {'dataVersion': {'year': '2026', 'month': '10'}},
                    'search': {'total': 12, 'hits': [{'id': 'ENSG00000139618', 'name': 'BRCA2', 'entity': 'target'}]}}}
    with patch.object(k, 'fetch', new=AsyncMock(return_value=obj)) as fetch:
        value = await k.opentargets_query(query='BRCA2 "query"', entity='target', limit=1, page=1)
    body = fetch.call_args.kwargs['body']
    assert body['variables']['query'] == 'BRCA2 "query"'
    assert 'BRCA2' not in body['query']
    assert body['variables']['page'] == {'index': 1, 'size': 1}
    assert value['count'] == 1 and value['total'] == 12 and value['has_more']
    assert value['version']['dataVersion']['year'] == '2026'
    assert value['retrieved_at'] and value['sources'][0]['url'].startswith('https://platform.opentargets.org/')


@pytest.mark.asyncio
async def test_ot_clinical_candidates_use_current_schema_and_slice_unpaged_rows():
    rows = [{'id': str(i), 'maxClinicalStage': 'Phase II', 'drug': {'id': 'CHEMBL25'}} for i in range(5)]
    with patch.object(k, 'fetch', new=AsyncMock(return_value={'data': {'target': {'id': 'ENSG00000139618', 'drugAndClinicalCandidates': {'count': 5, 'rows': rows}}}})) as fetch:
        value = await k.opentargets_query(operation='target', identifier='ENSG00000139618', resource='drugs', limit=2, page=1)
    assert [r['id'] for r in value['records']] == ['2', '3']
    assert value['total'] == 5 and value['has_more']
    query = fetch.call_args.kwargs['body']['query']
    assert 'drugAndClinicalCandidates' in query and 'knownDrugs' not in query
    assert '$page' not in query and 'page' not in fetch.call_args.kwargs['body']['variables']


@pytest.mark.asyncio
async def test_ot_graphql_failure_is_not_an_empty_search():
    with patch.object(k, 'fetch', new=AsyncMock(return_value={'data': {'target': None}, 'errors': [{'message': 'bad field'}]})):
        failed = await k.opentargets_query(operation='target', identifier='ENSG00000139618')
    with patch.object(k, 'fetch', new=AsyncMock(return_value={'data': {'target': None}})):
        empty = await k.opentargets_query(operation='target', identifier='ENSG00000139618')
    assert failed['status'] == 'unavailable' and 'records' not in failed
    assert empty['status'] == 'empty' and empty['records'] == []


@pytest.mark.asyncio
async def test_drug_does_not_use_retired_approval_fields():
    obj = {'data': {'drug': {'id': 'CHEMBL25', 'maximumClinicalStage': 'Approved'}}}
    with patch.object(k, 'fetch', new=AsyncMock(return_value=obj)) as fetch:
        value = await k.opentargets_query(operation='drug', identifier='CHEMBL25')
    assert value['records'][0]['maximumClinicalStage'] == 'Approved'
    document = fetch.call_args.kwargs['body']['query']
    assert 'maximumClinicalTrialPhase' not in document and 'isApproved' not in document


@pytest.mark.asyncio
async def test_ensembl_homology_species_and_intact_sequence():
    obj = {'data': [{'homologies': [{'id': 'ENSMUSG00000041147', 'species': 'mus_musculus'}]}]}
    with patch.object(k, 'fetch', new=AsyncMock(return_value=obj)) as fetch:
        value = await k.ensembl_query(operation='homology', identifier='ENSG00000139618.19', target_species='mus_musculus')
    assert value['records'][0]['species'] == 'mus_musculus'
    assert fetch.call_args.args[0].endswith('/homology/id/homo_sapiens/ENSG00000139618')
    assert fetch.call_args.args[1]['target_species'] == 'mus_musculus'
    seq = 'ACDE' * 1000
    with patch.object(k, 'fetch', new=AsyncMock(return_value=[{'id': 'ENSP00000369497', 'seq': seq}])):
        value = await k.ensembl_query(operation='sequence', identifier='ENSP00000369497', limit=1)
    assert value['records'][0]['seq'] == seq


@pytest.mark.asyncio
async def test_europepmc_preserves_abstract_provenance_and_cursor():
    obj = {'version': '6.9', 'hitCount': 25, 'nextCursorMark': 'opaque-next', 'resultList': {'result': [
        {'id': '123', 'source': 'MED', 'pmid': '123', 'abstractText': 'Measured activity.', 'isRetracted': 'Y'},
    ]}}
    with patch.object(k, 'fetch', new=AsyncMock(return_value=obj)) as fetch:
        value = await k.europepmc_query(query='hydrolase', limit=1, cursor='opaque-before')
    assert fetch.call_args.args[1]['cursorMark'] == 'opaque-before'
    assert value['next_cursor'] == 'opaque-next' and value['total'] == 25
    assert value['records'][0]['isRetracted'] == 'Y'
    assert value['records'][0]['evidence_level'] == 'abstract'
    assert value['records'][0]['url'] == 'https://europepmc.org/article/MED/123'
    with patch.object(k, 'fetch', new=AsyncMock(return_value={'hitCount': 0, 'resultList': {'result': []}})):
        assert (await k.europepmc_query(query='nothing'))['status'] == 'empty'


@pytest.mark.asyncio
async def test_pubchem_smiles_is_form_data_and_not_a_path():
    raw = json.dumps({'PropertyTable': {'Properties': [{'CID': 2244, 'SMILES': 'C/C=C\\C', 'MolecularWeight': '180.16'}]}}).encode()
    with patch.object(k, 'request', new=AsyncMock(return_value=raw)) as request:
        value = await k.pubchem_query(namespace='smiles', query='C/C=C\\C')
    assert request.call_args.args[0] == 'POST'
    assert request.call_args.kwargs['data'] == {'smiles': 'C/C=C\\C'}
    assert 'C/C=' not in request.call_args.args[1]
    assert value['sources'][0]['url'].endswith('/2244')
    assert value['evidence_level'] == 'database_properties'


@pytest.mark.asyncio
async def test_bounded_transport_and_failure_without_sensitive_exception_text():
    with patch.object(k, 'request', new=AsyncMock(side_effect=httpx.ReadTimeout('secret-in-error'))):
        value = await k.ensembl_query(identifier='BRCA2')
    assert value['status'] == 'unavailable' and 'secret-in-error' not in json.dumps(value)
    with patch.object(k, 'request', new=AsyncMock(return_value=b'{"data":{}}')) as request:
        await k.fetch(k.OT_URL, body={'query': 'fixed'})
    assert request.call_args.kwargs['limit'] == 1_000_000
    assert request.call_args.kwargs['json_body'] == {'query': 'fixed'}
    from mcp_tools.tools.research.services import request as real_request
    for url in ['https://127.0.0.1/private', 'https://rest.ensembl.org:8000/lookup/id/x']:
        with pytest.raises(ValueError):
            await real_request('GET', url)


def test_nested_lists_report_truncation():
    value, fields = k.bounded_lists({'targets': list(range(100))}, limit=3)
    assert value == {'targets': [0, 1, 2]} and fields == ['$.targets']


@pytest.mark.asyncio
async def test_new_tools_available_through_mcp_native_and_queue_catalog():
    from mcp_tools.scheduler.catalog import classify
    obj = {'data': {'search': {'total': 1, 'hits': [{'id': 'ENSG00000139618', 'entity': 'target'}]}}}
    with patch.object(k, 'fetch', new=AsyncMock(return_value=obj)):
        async with Client(mcp) as client:
            assert set(k.KNOWLEDGE_TOOL_NAMES) <= {t.name for t in await client.list_tools()}
            response = await client.call_tool('query_opentargets', {'query': 'BRCA2'})
        native = next(t for t in KNOWLEDGE_TOOLS if t.name == 'query_opentargets')
        value = await native.ainvoke({'query': 'BRCA2'})
    assert json.loads(response.content[0].text)['records'] == value['records']
    for name in k.KNOWLEDGE_TOOL_NAMES:
        assert classify(name)['execution'] == 'queued'
        assert classify(name)['category'] == ('web_search' if name == 'query_europepmc' else 'protein_database')
        assert classify(name)['resource'] == 'api:' + name.removeprefix('query_')


def test_gget_drug_compatibility_uses_current_api(monkeypatch):
    import requests
    from mcp_tools.tools.research.gget_worker import execute
    response = requests.Response(); response.status_code = 200
    response._content = b'{"data":{"target":{"drugAndClinicalCandidates":{"rows":[{"id":"candidate","maxClinicalStage":"Phase I"}]}}}}'
    with patch.object(requests, 'post', return_value=response) as post:
        data = execute({'operation': 'opentargets', 'identifiers': ['ENSG00000139618'], 'resource': 'drugs', 'limit': 2}, SimpleNamespace())
    assert data[0]['maxClinicalStage'] == 'Phase I'
    assert 'drugAndClinicalCandidates' in post.call_args.kwargs['json']['query']


def mapped_entry(taxon=9606, gene='ENSG00000139618.19'):
    return {'organism': {'taxonId': taxon, 'scientificName': 'Homo sapiens' if taxon == 9606 else 'Mus musculus'},
            'uniProtKBCrossReferences': [{'database': 'Ensembl', 'properties': [{'key': 'GeneId', 'value': gene}]}]}


def test_researcher_does_not_invent_cross_species_or_ambiguous_target_mappings():
    from backend.web_v2.research_report import knowledge_plan
    assert len(knowledge_plan(mapped_entry(), ['function'])) == 4
    assert len(knowledge_plan(mapped_entry(10090, 'ENSMUSG00000041147'), ['function'])) == 1
    assert knowledge_plan({}, ['function']) == []
    entry = mapped_entry()
    entry['uniProtKBCrossReferences'][0]['properties'].append({'key': 'GeneId', 'value': 'ENSG00000141510'})
    assert knowledge_plan(entry, ['function']) == []


@pytest.mark.asyncio
async def test_researcher_uses_shared_queue_owner_and_parent(monkeypatch):
    from backend.web_v2.research_report import knowledge_tool
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID', 'parent-report')
    monkeypatch.setenv('VENUS_QUEUE_OWNER', 'report:owner')
    monkeypatch.delenv('VENUS_MCP_QUEUE_WORKER', raising=False)
    value = {'status': 'empty', 'records': []}
    with patch('mcp_tools.tools.runtime_tool_policy.assert_tool_call_allowed') as gate, patch('mcp_tools.scheduler.client.invoke', new=AsyncMock(return_value=value)) as invoke:
        assert await knowledge_tool('query_europepmc', {'query': 'enzyme'}) == value
    gate.assert_called_once()
    invoke.assert_awaited_once_with(
        'query_europepmc', {'query': 'enzyme'}, owner='report:owner',
        parent_job='parent-report', on_status=None,
    )


@pytest.mark.asyncio
async def test_disabled_database_blocks_report_and_cached_native_tool():
    from backend.web_v2.research_report import knowledge_tool
    from mcp_tools.tools.capability_policy import CapabilityDisabled
    # Loading the hub applies the same live gate to already-held native references.
    from mcp_tools.tools.tools_agent_hub import ALL_TOOLS
    native = next(t for t in ALL_TOOLS if t.name == 'query_opentargets')
    with patch('mcp_tools.tools.capability_policy.CapabilityStore', autospec=True) as store, patch.object(k, 'fetch', new=AsyncMock()) as fetch:
        store.return_value.tool_enabled.return_value = False
        store.return_value.assert_call.side_effect = CapabilityDisabled('Disabled by administrator')
        with pytest.raises(CapabilityDisabled):
            await knowledge_tool('query_opentargets', {'query': 'BRCA2'})
        with pytest.raises(CapabilityDisabled):
            await native.ainvoke({'query': 'BRCA2'})
    fetch.assert_not_called()
