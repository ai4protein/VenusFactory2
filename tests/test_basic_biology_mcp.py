import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from mcp_tools.tools.research import basic_queries as b, services as s
from mcp_tools.tools.research.tools_agent import PUBLIC_PROTEIN_TOOLS
from mcp_tools.tools.research.tools_mcp import mcp


@pytest.mark.asyncio
async def test_ncbi_search_returns_actual_summaries_and_total():
    with patch.object(b, 'get_json', new=AsyncMock(side_effect=[
        {'esearchresult': {'count': '24', 'idlist': ['3039']}},
        {'result': {'uids': ['3039'], '3039': {'uid': '3039', 'name': 'HBA1'}}},
    ])) as fetch:
        result = await b.ncbi_query(database='gene', query='HBA1 AND human', limit=1)
    assert result['count'] == 1 and result['total'] == 24
    assert result['records'][0]['name'] == 'HBA1'
    assert fetch.call_args.args[1]['id'] == '3039'
    assert 'ncbi.nlm.nih.gov' in result['sources'][0]['url']


@pytest.mark.asyncio
async def test_ncbi_empty_is_not_failure_or_made_up_records():
    with patch.object(b, 'get_json', new=AsyncMock(return_value={'esearchresult': {'count': '0', 'idlist': []}})) as fetch:
        result = await b.ncbi_query(query='no matching protein')
    assert result['status'] == 'empty' and result['records'] == []
    assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_ncbi_fasta_and_abstracts():
    with patch.object(b, 'request', new=AsyncMock(return_value=b'>NP_000549.1\nMVLSPADKTNVKAAWGKVG')):
        result = await b.ncbi_query(action='sequence', ids=['NP_000549.1'])
    assert result['fasta'].endswith('KVG') and result['count'] == 1
    xml = b'<PubmedArticleSet><PubmedArticle><PMID>123</PMID><ArticleTitle>A <i>protein</i> study</ArticleTitle><Abstract><AbstractText>Measured binding.</AbstractText></Abstract></PubmedArticle></PubmedArticleSet>'
    with patch.object(b, 'request', new=AsyncMock(return_value=xml)):
        result = await b.ncbi_query(database='pubmed', action='abstracts', ids=['123'])
    assert result['records'][0]['title'] == 'A protein study'
    assert result['records'][0]['abstract'] == 'Measured binding.'


@pytest.mark.parametrize('args', [
    {'database': 'anything', 'query': 'HBA1'}, {'query': ''},
    {'action': 'sequence', 'database': 'gene', 'ids': ['3039']},
    {'action': 'summary', 'ids': ['https://private']}, {'query': 'a', 'limit': 1000},
    {'database': 'gene', 'action': 'abstracts', 'ids': ['3039']},
    {'action': 'summary', 'ids': ['3039'], 'query': 'wrong'},
])
def test_ncbi_rejects_invalid_contracts(args):
    with pytest.raises(ValueError): b.NCBIInput(**args)


@pytest.mark.parametrize('args', [
    {'operation': 'gpt', 'identifiers': ['x']},
    {'operation': 'pdb', 'identifiers': ['../secret']},
    {'operation': 'info', 'identifiers': ['HBA1']},
    {'operation': 'pdb', 'identifiers': ['1A3N'], 'resource': 'pdb'},
    {'operation': 'seq', 'identifiers': ['ENSG00000206172']*6},
    {'operation': 'archs4', 'identifiers': ['HBA1'], 'species': 'worm'},
    {'operation': 'enrichr', 'identifiers': ['HBA1']},
    {'operation': 'pdb', 'identifiers': ['1A3N'], 'save': True},
])
def test_gget_rejects_arbitrary_commands_heavy_modules_and_unbounded_inputs(args):
    with pytest.raises(ValueError): b.GgetInput(**args)


@pytest.mark.asyncio
async def test_blast_submit_then_resume_never_duplicates():
    with patch.object(s, 'request', new=AsyncMock(return_value=b'RID = TESTRID123\n')) as call:
        first = await b.similarity_query('blast', sequence='A'*20)
    assert first['status'] == 'pending' and first['job_id'] == 'TESTRID123'
    assert first['next_poll'] > first['started']
    assert call.await_count == 1 and call.call_args.args[0] == 'POST'
    with patch.object(s, 'request', new=AsyncMock(return_value=b'<BlastOutput/>')) as call:
        second = await b.similarity_query('blast', job_id=first['job_id'])
    assert second['status'] == 'empty'
    assert call.await_count == 1 and call.call_args.args[0] == 'GET'


@pytest.mark.asyncio
async def test_foldseek_remote_resume_needs_no_structure_or_local_engine():
    with patch.object(s, 'get_json', new=AsyncMock(side_effect=[{'status': 'COMPLETE'}, {'results': [{'db': 'pdb100', 'alignments': [[{'target': '1A3N_A', 'seqId': 100}]]}]}])) as call:
        result = await b.similarity_query('foldseek_remote', job_id='FOLDSEEK123')
    assert result['status'] == 'completed' and result['count'] == 1
    assert call.await_count == 2
    with pytest.raises(ValueError): await b.similarity_query('foldseek_remote', job_id='FOLDSEEK123', accession='P69905')


@pytest.mark.asyncio
async def test_foldseek_waiting_returns_promptly():
    with patch.object(s, 'get_json', new=AsyncMock(return_value={'status': 'RUNNING'})), patch.object(s.asyncio, 'sleep', new=AsyncMock()) as sleep:
        result = await b.similarity_query('foldseek_remote', job_id='FOLDSEEK123')
    assert result['status'] == 'pending'
    sleep.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('body', [b'<html>Service maintenance</html>', b'Status = UNKNOWN'])
async def test_blast_errors_are_not_reported_as_queued(body):
    with patch.object(s, 'request', new=AsyncMock(return_value=body)):
        result = await b.similarity_query('blast', job_id='TESTRID123')
    assert result['status'] == 'unavailable'
    assert result['job_id'] == 'TESTRID123'


@pytest.mark.asyncio
async def test_mcp_and_native_tools_share_validated_queries():
    expected = {'status': 'completed', 'records': [{'name': 'HBA1'}]}
    with patch.object(b, 'get_json', new=AsyncMock(return_value={'result': {'uids': ['3039'], '3039': {'name': 'HBA1'}}})):
        async with Client(mcp) as client:
            names = {t.name for t in await client.list_tools()}
            assert set(b.BASIC_PUBLIC_TOOL_NAMES) <= names
            response = await client.call_tool('query_ncbi', {'database': 'gene', 'action': 'summary', 'ids': ['3039']})
        native = next(t for t in PUBLIC_PROTEIN_TOOLS if t.name == 'query_ncbi')
        result = await native.ainvoke({'database': 'gene', 'action': 'summary', 'ids': ['3039']})
    assert json.loads(response.content[0].text)['records'] == expected['records'] == result['records']


def test_online_discovery_includes_remote_tools_but_denies_local_foldseek():
    from mcp_tools.tools.tools_agent_hub import get_tools, PI_SEARCH_TOOL_NAMES
    from mcp_tools.tools.runtime_tool_policy import filter_tools, is_tool_allowed
    from mcp_tools.tools.discovery.service import search
    from agent.harness.capability_guidance import discovery_instructions
    with patch('mcp_tools.tools.runtime_tool_policy.is_online_mode', return_value=True):
        allowed, _ = filter_tools(get_tools())
        names = {t.name for t in allowed}
        assert set(b.BASIC_PUBLIC_TOOL_NAMES) <= names
        assert set(b.BASIC_PUBLIC_TOOL_NAMES) <= PI_SEARCH_TOOL_NAMES
        assert not is_tool_allowed('download_foldseek_results_by_pdb_file')
        assert not is_tool_allowed('fake_foldseek_remote')
        for key in b.BASIC_PUBLIC_TOOL_NAMES:
            assert key in json.dumps(search(key, kind='tools'))
        text = discovery_instructions(['mcp__venusfactory__'+n for n in names])
        # Availability comes from live discovery rather than an exhaustive prompt.
        assert 'mcp__venusfactory__search_research_tools' in text


@pytest.mark.asyncio
async def test_gget_worker_has_no_api_key_environment_and_is_reaped_on_timeout(tmp_path, monkeypatch):
    runtime = tmp_path/'.venvs/public-query/bin/python';runtime.parent.mkdir(parents=True);runtime.touch()
    monkeypatch.setattr(b, 'ROOT', tmp_path)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'never-forward-this')
    @asynccontextmanager
    async def slot(*args, **kwargs): yield 'test'
    class Input:
        def write(self, data): self.data=data
        async def drain(self): pass
        def close(self): pass
    class Proc:
        returncode=None
        stdin=Input()
        stdout=None
        killed=False
        def kill(self): self.killed=True;self.returncode=-9
        async def wait(self): return self.returncode
    proc=Proc()
    with patch('backend.web_v2.job_queue.resource_slot', slot), patch.object(b.asyncio, 'create_subprocess_exec', new=AsyncMock(return_value=proc)) as start, patch.object(b, 'read_limited', new=AsyncMock(side_effect=asyncio.TimeoutError)):
        result = await b.gget_query(operation='pdb', identifiers=['1A3N'])
    assert result['status'] == 'unavailable' and proc.killed
    assert 'OPENROUTER_API_KEY' not in start.call_args.kwargs['env']
    assert start.call_args.args[0] == str(runtime)


@pytest.mark.asyncio
async def test_output_is_bounded():
    stream = asyncio.StreamReader(); stream.feed_data(b'a'*30);stream.feed_eof()
    with pytest.raises(ValueError): await b.read_limited(stream, 20)


def test_gget_http_wrapper_blocks_other_hosts_and_upgrades_ensembl(monkeypatch):
    import requests
    from mcp_tools.tools.research.gget_worker import install_http_bounds
    seen=[]
    def request(session, method, url, **kwargs):
        seen.append((url,kwargs));r=requests.Response();r.status_code=200;r._content=b'{}';r._content_consumed=True;return r
    monkeypatch.setattr(requests.sessions.Session, 'request', request)
    sources=install_http_bounds()
    requests.get('http://rest.ensembl.org/lookup/id/ENSG00000206172')
    assert seen[0][0].startswith('https://') and seen[0][1]['allow_redirects'] is False
    assert sources[0]['title']=='rest.ensembl.org'
    requests.get('https://bgee.org/api/',params={'page':'gene'})
    assert seen[-1][0] == 'https://www.bgee.org/api/'
    with pytest.raises(ValueError): requests.get('http://127.0.0.1:7862/private')
    with pytest.raises(ValueError): requests.get('https://rest.ensembl.org:1234/')


def test_pdb_metadata_uses_bounded_requests(monkeypatch):
    import requests
    from mcp_tools.tools.research.gget_worker import pdb_urlopen
    r=requests.Response();r.status_code=200;r._content=b'{"rcsb_id":"1A3N"}'
    with patch.object(requests, 'get', return_value=r) as get:
        result=pdb_urlopen('https://data.rcsb.org/rest/v1/core/entry/1A3N')
    assert json.load(result)['rcsb_id']=='1A3N'
    assert result.status==200 and get.call_count==1
