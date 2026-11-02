import asyncio
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from backend.web_v2 import research_predictions as p
from backend.web_v2.research_activity import ActivityJournal
from backend.web_v2 import research_report as r
from backend.web_v2 import research_longform as lf


class PredictionTests(unittest.IsolatedAsyncioTestCase):
    def test_selects_only_ready_compatible_weights_with_property_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'src').mkdir()
            cfg={'web_ui': {'dataset_mapping_function': {'Solubility':['sol'], 'Stability':['stable'], 'Kcat':['kcat'], 'Optimal PH':['ph']},
                 'residue_mapping_function': {'Binding Site':['bind']}, 'model_mapping_function': {'ESM2':'esm2','Ankh':'ankh'},
                 'model_residue_mapping_function': {'ESM2':'esm2','Ankh':'ankh'}, 'model_adapter_mapping_function': {'esm2':'esm','ankh':'ankh'}}}
            (root/'src/constant.json').write_text(json.dumps(cfg))
            for dataset in ['sol','stable','kcat','ph','bind']:
                for model in ['esm','ankh']:
                    folder=root/'ckpt'/dataset/model;folder.mkdir(parents=True)
                    (folder/'lr5e-4_bt12k_ga8.pt').write_bytes(b'test')
                    (folder/'lr5e-4_bt12k_ga8.json').write_text('{}')
            rows=p.candidates({},root=root)
            self.assertEqual([x['task'] for x in rows[:3]],['Solubility','Stability','Binding Site'])
            self.assertEqual(len(rows),5)
            self.assertFalse(any(x['task'] in {'Kcat','Optimal PH'} for x in rows))
            enzyme=p.candidates({'entry':{'comments':[{'commentType':'CATALYTIC ACTIVITY'}]}},root=root)
            self.assertTrue(any(x['task']=='Optimal PH' for x in enzyme))
            self.assertFalse(any(x['task']=='Kcat' for x in enzyme))
            (root/'ckpt/sol/esm/lr5e-4_bt12k_ga8.pt').unlink()
            self.assertFalse(any(x['task']=='Solubility' and x['model']=='ESM2' for x in p.candidates({},root=root)))

    async def test_real_output_required_and_later_models_still_run(self):
        specs=[{'id':str(i),'task':'Solubility','model':'ESM2','dataset':'sol','tool':'predict_protein_function','adapter_path':'unused'} for i in range(3)]
        calls=[];updates=[]
        async def execute(spec, fasta, output, progress):
            calls.append(spec['id']);self.assertIn('MKTAYIAKQRQISFVK',fasta.read_text())
            progress({'status':'running','job_number':'VF-test','position':0})
            if spec['id']=='0':return {'status':'error'}
            if spec['id']=='1':return {'status':'success'} # Missing CSV must not be called success.
            output.write_text('header,sequence,score\nquery,MKTAYIAKQRQISFVK,0.8\n')
            return {'status':'success'}
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'candidates',return_value=specs),patch.object(p,'is_online_mode',return_value=False):
            rows=await p.run_predictions({'sequence':'MKTAYIAKQRQISFVK'},lambda x:updates.append(json.loads(json.dumps(x))),tmp,execute=execute)
        self.assertEqual(calls,['0','1','2'])
        self.assertEqual([x['status'] for x in rows],['unavailable','unavailable','completed'])
        self.assertEqual(rows[2]['values'],[{'score':'0.8'}])
        self.assertGreater(len(updates),3)
        self.assertIn('Model predictions, not experimental',p.report_appendix(rows))

    async def test_online_never_dispatches_gpu_and_cancellation_propagates(self):
        execute=AsyncMock()
        with patch.object(p,'is_online_mode',return_value=True):
            with self.assertRaises(ValueError):await p.run_predictions({},lambda x:None,'unused',execute=execute)
        execute.assert_not_awaited()
        spec={'id':'a','task':'Solubility','model':'ESM2','dataset':'sol','tool':'predict_protein_function','adapter_path':'unused'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(p,'candidates',return_value=[spec]),patch.object(p,'is_online_mode',return_value=False):
            with self.assertRaises(asyncio.CancelledError):
                await p.run_predictions({'sequence':'MKTAYIAKQRQISFVK'},lambda x:None,tmp,execute=AsyncMock(side_effect=asyncio.CancelledError))

    async def test_predictions_enter_evidence_sections_and_report(self):
        row={'id':'a','task':'Solubility','model':'ESM2','dataset':'DeepSol','status':'completed','sequence_sha256':'digest','values':[{'score':0.8}],'evidence_level':'model_prediction'}
        async def predict(context,update,folder): update([row]);return [row]
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'is_online_mode',return_value=False),patch.object(p,'run_predictions',side_effect=predict):
            state=await r.run_research({'analyses':['ai_predictions'],'model_id':'free','sequence':'MKTAYIAKQRQISFVK'},lambda x:None,synthesize=False,service_cache=Path(tmp)/'remote')
        self.assertEqual(state['sources'][0]['kind'],'prediction')
        self.assertEqual(state['sections'][0]['data']['dataset'],'DeepSol')
        state['chapters']=[]
        self.assertIn('AI model predictions',lf.assemble(state,'en'))
        self.assertIn('DeepSol',lf.assemble(state,'en'))

    def test_activity_keeps_actual_transitions_without_repeated_poll_entries(self):
        journal=ActivityJournal()
        state={'phase':'collecting','steps':[{'id':'identity','status':'running'}]}
        self.assertEqual(len(journal.observe(state)),2)
        self.assertEqual(len(journal.observe(state)),2)
        state['steps'][0]['status']='completed'
        events=journal.observe(state)
        self.assertEqual(len(events),3)
        self.assertEqual(events[-1]['status'],'completed')
        self.assertTrue(events[-1]['at'])

    async def test_mcp_prediction_preserves_parent_and_owner(self):
        spec={'tool':'predict_protein_function','task':'Solubility','model':'ESM2','adapter_path':'ckpt'}
        invoke=AsyncMock(return_value={'status':'success'})
        with patch.object(p,'is_online_mode',return_value=False),patch.dict('os.environ',{'VENUS_QUEUE_OWNER':'alice','VENUS_QUEUE_JOB_ID':'parent'}),patch('mcp_tools.scheduler.client.invoke',invoke):
            await p.queued_prediction(spec,Path('query.fasta'),Path('result.csv'),lambda x:None)
        self.assertEqual(invoke.call_args.kwargs['owner'],'alice')
        self.assertEqual(invoke.call_args.kwargs['parent_job'],'parent')

    async def test_prediction_does_not_wait_for_remote_search(self):
        started=asyncio.Event()
        async def predict(context,update,folder):started.set();return []
        async def remote(key,context,update,**kwargs):
            await asyncio.wait_for(started.wait(),timeout=1)
            return {'id':key,'status':'empty'}
        with tempfile.TemporaryDirectory() as tmp,patch.object(r,'is_online_mode',return_value=False),patch.object(p,'run_predictions',side_effect=predict),patch.object(r.public_services,'run_service',side_effect=remote),patch.object(r.public_services,'source',return_value=None),patch.dict('os.environ',{'VENUS_QUEUE_JOB_ID':''}):
            await r.run_research({'analyses':['ai_predictions','alphafold'],'model_id':'free','sequence':'MKTAYIAKQRQISFVK'},lambda x:None,synthesize=False,service_cache=Path(tmp)/'remote')
        self.assertTrue(started.is_set())

if __name__=='__main__':unittest.main()
