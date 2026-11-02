import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from mcp_tools.tools.research import services as s
from backend.web_v2 import research_report as r, research_longform as lf
from backend.web_v2.job_queue import JobQueue


def pdb(chain='A', residue='ALA'):
    return 'HEADER PRIVATE ANNOTATION\n'+''.join(f'ATOM  {i:5d}  CA  {residue} {chain}{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}  1.00 90.00           C  \n' for i in range(1,13))+'END\n'


class PublicProteinTests(unittest.IsolatedAsyncioTestCase):
    def test_pdb_chain_preserved_headers_removed(self):
        cleaned,seq,chain=s.structure_chain(pdb())
        self.assertEqual((seq,chain),('A'*12,'A'))
        self.assertNotIn('PRIVATE',cleaned)
        with self.assertRaises(ValueError):s.structure_chain(pdb(),'B')
        parsed=r.parsed_input(pdb())
        self.assertIn('ATOM',parsed['structure_pdb'])
        self.assertEqual(parsed['sequence_map']['A'],'A'*12)
        self.assertEqual(r.parsed_input('A'*12)['structure_pdb'],'')

    def test_blast_metrics(self):
        raw=b'<BlastOutput><BlastOutput_query-len>100</BlastOutput_query-len><Hit><Hit_accession>P12345</Hit_accession><Hit_def>protein</Hit_def><Hit_hsps><Hsp><Hsp_align-len>50</Hsp_align-len><Hsp_identity>40</Hsp_identity><Hsp_query-from>10</Hsp_query-from><Hsp_query-to>59</Hsp_query-to><Hsp_evalue>1e-20</Hsp_evalue><Hsp_bit-score>80</Hsp_bit-score></Hsp></Hit_hsps></Hit></BlastOutput>'
        hit=s.blast_hits(raw)[0]
        self.assertEqual(hit['identity_pct'],80)
        self.assertEqual(hit['query_coverage_pct'],50)
        self.assertEqual(hit['evalue'],1e-20)
        with self.assertRaises(ValueError):s.blast_hits(b'<!ENTITY a "x"><BlastOutput/>')

    def test_foldseek_hit_semantics(self):
        hit={'target':'1abc_A','eval':1e-5,'seqId':0.3,'score':41}
        rows=s.foldseek_hits({'results':[{'db':'pdb100','alignments':[[hit]]}]})
        self.assertEqual(rows[0]['eval'],1e-5)
        self.assertNotIn('tmscore',rows[0])
        self.assertEqual(s.foldseek_hits({'results':[]} ),[])

    async def test_remote_endpoints_and_size_bounds(self):
        @asynccontextmanager
        async def slot(*args,**kwargs):yield 'test'
        real=httpx.AsyncClient
        with patch('backend.web_v2.job_queue.resource_slot',slot),patch.object(s,'rate_gate',new=AsyncMock()),patch.object(s.httpx,'AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(lambda req:httpx.Response(200,content=b'123456')),**kw)):
            with self.assertRaises(ValueError):await s.request('GET','http://127.0.0.1/private')
            with self.assertRaises(ValueError):await s.request('GET','https://www.ebi.ac.uk/large',limit=5)
        with patch('backend.web_v2.job_queue.resource_slot',slot),patch.object(s,'rate_gate',new=AsyncMock()),patch.object(s.httpx,'AsyncClient',side_effect=lambda **kw:real(transport=httpx.MockTransport(lambda req:httpx.Response(404)),**kw)):
            with self.assertRaises(ValueError):await s.request('POST','https://search.foldseek.com/api/ticket')

    async def test_blast_resume_does_not_submit_again(self):
        row={'job_id':'EXISTING','next_poll':0}
        with patch.object(s,'request',new=AsyncMock(return_value=b'<BlastOutput/>')) as call:
            result=await s.blast({'sequence':'A'*20},row,lambda:None)
        self.assertEqual(result['count'],0)
        self.assertEqual(call.call_args.args[0],'GET')
        with patch.object(s,'request',new=AsyncMock()) as call:
            with self.assertRaises(ValueError):await s.blast({'sequence':'A'*20},{'submission_started':True},lambda:None)
            call.assert_not_called()

    async def test_checkpoint_and_empty_result(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(s,'annotation',new=AsyncMock(return_value={'count':0,'entries':[]})) as call:
            first=await s.run_service('interpro',{'accession':'P69905'},cache_dir=folder)
            second=await s.run_service('interpro',{'accession':'P69905'},cache_dir=folder)
            self.assertEqual(first['status'],'empty');self.assertEqual(first,second)
            self.assertEqual(call.await_count,1)
            self.assertIn('empty',s.source(first)['text'])

    async def test_timeout_retains_remote_ticket(self):
        async def timeout(context,row,emit):
            row.update(job_id='KNOWN',url='https://blast.ncbi.nlm.nih.gov/Blast.cgi?RID=KNOWN');emit()
            raise asyncio.TimeoutError()
        with patch.object(s,'blast',side_effect=timeout):
            row=await s.run_service('blast',{'sequence':'A'*20})
        self.assertEqual(row['status'],'pending');self.assertEqual(row['job_id'],'KNOWN')
        self.assertIsNone(s.source(row))

    async def test_missing_identity_is_not_success(self):
        row=await s.run_service('interpro',{})
        self.assertEqual(row['status'],'not_applicable')
        self.assertIsNone(s.source(row))

    async def test_global_ncbi_spacing_and_daily_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            queue=JobQueue(Path(folder));clock=[1000.]
            async def sleep(delay):clock[0]+=delay
            with patch('backend.web_v2.job_queue.JobQueue',return_value=queue),patch.object(s.time,'time',side_effect=lambda:clock[0]),patch.object(s.asyncio,'sleep',side_effect=sleep):
                await s.rate_gate('blast.ncbi.nlm.nih.gov',rid='test')
                await s.rate_gate('blast.ncbi.nlm.nih.gov',rid='test')
                self.assertEqual(clock[0],1060)
                await s.rate_gate('blast.ncbi.nlm.nih.gov')
                self.assertEqual(clock[0],1070)
                for _ in range(50):await s.rate_gate('search.foldseek.com',submit=True)
                with self.assertRaises(ValueError):await s.rate_gate('search.foldseek.com',submit=True)

    async def test_report_collects_remote_evidence_without_local_compute(self):
        async def run(key,context,update,cache_dir=None):
            self.assertEqual(context['sequence'],'A'*20)
            row={'id':key,'label':'NCBI BLAST','status':'completed','url':'https://blast.ncbi.nlm.nih.gov','evidence_level':'sequence_similarity','data':{'count':1,'hits':[{'accession':'P69905'}]}}
            update(row);return row
        with patch.object(s,'run_service',side_effect=run),patch.object(r,'is_online_mode',return_value=True):
            result=await r.run_research({'analyses':['blast'],'sequence':'A'*20,'model_id':'free'},lambda x:None,synthesize=False)
        self.assertEqual(result['sources'][0]['kind'],'public_service')
        self.assertEqual(result['remote_services'][0]['status'],'completed')

    async def test_specialist_receives_public_evidence_and_exports_actual_results(self):
        row={'id':'blast','label':'NCBI BLAST','status':'completed','count':1,'url':'https://blast.ncbi.nlm.nih.gov','data':{'count':1,'hits':[{'accession':'P69905','identity_pct':99.2}]}}
        source={**s.source(row),'id':'S1'}
        async def collect(payload,update,**kwargs):
            value={'protein':{'name':'Test'},'sources':[source],'remote_services':[row],'steps':[],'sections':[],'partial':False}
            update(value);return value
        calls=[]
        async def complete(model,messages,tokens):
            calls.append(messages)
            return ('protein evidence '*220+'[S1]',False)
        with patch.object(r,'run_research',side_effect=collect),patch.object(r,'complete',side_effect=complete):
            result=await lf.run_long_report({'model_id':'free','analyses':['blast']},lambda x:None)
        identity=next(json.loads(m[1]['content']) for m in calls if lf.CHAPTERS[0][3] in m[0]['content'])
        self.assertEqual(identity['sources'][0]['kind'],'public_service')
        self.assertIn('99.2',result['report']);self.assertIn('Public service results',result['report'])

    def test_online_free_remote_services_remain_enabled(self):
        with patch.object(r,'is_online_mode',return_value=True):
            opts={x['id']:x for x in r.analysis_options()}
            for key in s.SERVICE_IDS:self.assertTrue(opts[key]['available'])
            self.assertFalse(opts['mutation']['available'])
        from mcp_tools.tools.research.tools_agent import PublicProteinInput
        with self.assertRaises(ValueError):PublicProteinInput(service='custom-url')

    async def test_mcp_lists_public_service(self):
        from mcp_tools.tools.research.tools_mcp import mcp
        tools=await mcp.list_tools()
        self.assertIn('query_public_protein_service',[t.name for t in tools])

    async def test_mcp_and_scientist_invoke_same_adapter(self):
        from fastmcp import Client
        from mcp_tools.tools.research.tools_mcp import mcp
        from mcp_tools.tools.research.tools_agent import query_public_protein_service_tool
        result={'id':'blast','status':'completed','count':1}
        with patch('mcp_tools.tools.research.tools_agent.run_service',new=AsyncMock(return_value=result)) as run:
            async with Client(mcp) as client:
                response=await client.call_tool('query_public_protein_service',{'service':'blast','sequence':'A'*20})
                self.assertIn('completed',response.content[0].text)
            value=await query_public_protein_service_tool.ainvoke({'service':'blast','sequence':'A'*20})
            self.assertEqual(value,result)
            self.assertEqual(run.await_count,2)

    def test_online_discovery_finds_public_tools(self):
        from mcp_tools.tools.tools_agent_hub import get_tools,PI_SEARCH_TOOL_NAMES
        from mcp_tools.tools.runtime_tool_policy import filter_tools
        from mcp_tools.tools.discovery.service import search
        with patch('mcp_tools.tools.runtime_tool_policy.is_online_mode',return_value=True):
            names={t.name for t in filter_tools(get_tools())[0]}
            self.assertIn('query_public_protein_service',names)
            self.assertIn('query_public_protein_service',PI_SEARCH_TOOL_NAMES)
            self.assertIn('query_public_protein_service',json.dumps(search('NCBI BLAST',kind='tools')))

    def test_api_retains_uploaded_coordinates_and_rejects_mismatched_chain(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.web_v2.report_api import router
        from backend.web_v2.research_jobs import ResearchJobs
        app=FastAPI();app.include_router(router)
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'VENUS_REPORT_ROOT':folder+'/reports','VENUS_JOB_QUEUE_ROOT':folder+'/queue'}),TestClient(app) as client:
            result=client.post('/api/report/parse-input',json={'content':pdb()})
            self.assertEqual(result.status_code,200)
            self.assertIn('ATOM',result.json()['structure_pdb'])
            with patch('backend.web_v2.research_jobs.model_option'):
                body={'model_id':'free','sequence':'M'*12,'structure_pdb':pdb(),'structure_chain':'A','analyses':['foldseek_remote']}
                result=client.post('/api/report/jobs',json=body)
            self.assertEqual(result.status_code,422)
            self.assertIn('must match',result.json()['detail'])

    async def test_public_wrapper_validates_before_network(self):
        from mcp_tools.tools.research.tools_agent import public_analysis
        with patch('mcp_tools.tools.research.tools_agent.get_json',new=AsyncMock()) as fetch:
            with self.assertRaises(ValueError):await public_analysis('blast',sequence='invalid-protein')
            with self.assertRaises(ValueError):await public_analysis('foldseek_remote',sequence='M'*12,structure_pdb=pdb())
            fetch.assert_not_called()


if __name__=='__main__':unittest.main()
