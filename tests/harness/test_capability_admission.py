import copy
import json
from pathlib import Path
import pytest
from agent.harness.research_workspace import ResearchWorkspace, publish_proposals, run_capability, search_capabilities, describe_capability, run_python
from agent.harness.capability_review import review_proposal, COMMON_CHECKS, MODEL_CHECKS

def spec(kind='tool'):
    return {'kind':kind,'author':'Test team','version':'1.0','scope':{'proteins':['fixture only'],'properties':['sequence length'],'scenarios':['test'],'limitations':['not biological validation']},
        'input_schema':{'type':'object','required':['sequence'],'properties':{'sequence':{'type':'string','minLength':1}},'additionalProperties':False},
        'output_schema':{'type':'object','required':['value'],'properties':{'value':{'type':'number'}},'additionalProperties':False},
        'dependencies':[],'cases':[{'inputs':{'sequence':'AC'},'expected':{'value':2}},{'inputs':{'sequence':''},'reject':True}],
        'evidence_files':['README.md']}

async def reviewer(context):
    assert not {'PI','SC','progress','previous_episodes','execution'} & set(context)
    return {'decision':'approve','summary':'Interface and bounded fixture scope checked','issues':[],
        'checks':{k:{'passed':True,'reason':'Checked supplied source and host receipt'} for k in context['required_checks']},
        'tests':[{'inputs':{'sequence':'ACDE'},'expected':{'value':4}},{'inputs':{'sequence':3},'reject':True}]}

async def candidate(tmp_path,review_spec=None):
    service=ResearchWorkspace(tmp_path/'work',tmp_path/'control',{'steps':[]})
    await service.write_research_file('length.py','def run(inputs):\n return {"value":len(inputs["sequence"])}\n')
    await service.write_research_file('README.md','Deterministic length calculation only; no biological accuracy claim.')
    await service.propose_research_capability('Length','Fixture length','length.py',[],[{'sequence':'A'}],review_spec)
    return service

@pytest.mark.asyncio
async def test_execution_critic_or_forged_manifest_do_not_admit(tmp_path):
    service=await candidate(tmp_path)
    with pytest.raises((ValueError,OSError)):publish_proposals(service.proposals,'g',tmp_path/'library')
    receipt=await review_proposal(service.proposals[0],reviewer=reviewer,control=tmp_path/'review',runner=run_python)
    assert receipt['status']=='needs_revision'
    assert not search_capabilities(library=tmp_path/'library')['items']

@pytest.mark.asyncio
async def test_independent_review_to_discovery_schema_checked_call_and_private_evidence(tmp_path):
    service=await candidate(tmp_path,spec());bundle=service.proposals[0]
    result=await review_proposal(bundle,reviewer=reviewer,control=tmp_path/'review',runner=run_python)
    assert result['status']=='approved',result
    library=tmp_path/'library';entry=publish_proposals([bundle],'g',library)[0]
    assert not (library/entry['id']/'README.md').exists()
    assert search_capabilities('fixture',library=library)['items'][0]['review_status']=='approved'
    assert describe_capability(entry['id'],library)['input_schema']['type']=='object'
    assert await run_capability(entry['id'],{'sequence':'ABCDE'},library=library)=={'value':5}
    with pytest.raises(Exception):await run_capability(entry['id'],{'sequence':None},library=library)
    (library/entry['id']/'length.py').write_text('def run(x):return {}')
    with pytest.raises(ValueError):await run_capability(entry['id'],{'sequence':'A'},library=library)

@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['wrong_output','missing_check','reused_test','review_unavailable'])
async def test_review_cannot_approve_without_complete_independent_evidence(tmp_path,fault):
    service=await candidate(tmp_path,spec())
    async def broken(context):
        if fault=='review_unavailable':raise RuntimeError('offline')
        result=await reviewer(context)
        if fault=='wrong_output':result['tests'][0]['expected']['value']=999
        if fault=='missing_check':del result['checks']['scope']
        if fault=='reused_test':result['tests'][0]['inputs']['sequence']='AC'
        return result
    result=await review_proposal(service.proposals[0],reviewer=broken,control=tmp_path/'review',runner=run_python)
    assert result['status']=='needs_revision' and result['issues']
    with pytest.raises(ValueError):publish_proposals(service.proposals,'g',tmp_path/'library')

async def model_candidate(tmp_path,leak=False,metric=1.0,no_refit=False,worse=False):
    service=ResearchWorkspace(tmp_path/'work',tmp_path/'control',{'steps':[]})
    code='import json\ndef run(inputs):\n return {"value":len(inputs["sequence"])*json.load(open("weights.json"))["scale"]}\n'
    await service.write_research_file('length.py',code)
    await service.write_research_file('weights.json','{"scale":1}')
    await service.write_research_file('README.md','Deliberately simple fixture: test targets are length+1. MAE is 1. No performance superiority claimed.')
    await service.write_research_file('train.py','pass\n' if no_refit else 'import json\ndata=json.load(open("eval.json"))\nrows=data["splits"]["train"]\nscale=sum(r["target"]-len(r["inputs"]["sequence"]) for r in rows)/len(rows)\njson.dump({"scale":scale},open("weights.json","w"))\n')
    rows=lambda start,count:[{'id':str(i),'group':str(i),'inputs':{'sequence':'A'*i},'target':i+1} for i in range(start,start+count)]
    data={'task':'regression','provenance':'synthetic software fixture','grouping':'one independent fixture per group','splits':{'train':rows(1,5),'validation':rows(6,2),'test':rows(8,3)},'claimed_metric':metric}
    if worse:
        for row in data['splits']['test']:row['target']=4
        data['claimed_metric']=5
    if leak:data['splits']['test'][0]['group']='1'
    await service.write_research_file('eval.json',json.dumps(data))
    contract=spec('model');contract.update(evidence_files=['README.md','train.py','eval.json'],evaluation_file='eval.json',prediction_key='value',reproduction_script='train.py',fitted_files=['weights.json'])
    await service.propose_research_capability('Model','Fixture model','length.py',['weights.json'],[{'sequence':'A'}],contract)
    return service

@pytest.mark.asyncio
@pytest.mark.parametrize('fault',['none','worse','leak','metric','no_refit'])
async def test_model_host_recomputes_splits_metrics_baseline_and_fresh_training(tmp_path,fault):
    service=await model_candidate(tmp_path,leak=fault=='leak',metric=100 if fault=='metric' else 1,no_refit=fault=='no_refit',worse=fault=='worse')
    result=await review_proposal(service.proposals[0],reviewer=reviewer,control=tmp_path/'review',runner=run_python)
    assert result['status']==('approved' if fault in {'none','worse'} else 'needs_revision'),result
    if fault=='worse':
        evaluation=result['automatic']['evaluation']
        assert evaluation['score']==5 and evaluation['baseline']==0
    if fault=='none':
        evaluation=result['automatic']['evaluation']
        assert evaluation['score']==1 and evaluation['training_reproduced']

@pytest.mark.asyncio
async def test_admission_revision_continues_even_after_research_steps_complete(tmp_path,monkeypatch):
    from agent.harness import autonomous_research as ar
    from agent.harness import Profile
    class Relay:
        calls=0;base_url='unused';token='fixture'
        def __init__(self,**kw):Path(kw['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(ar,'EvidenceGateway',Relay)
    monkeypatch.setattr(ar,'audit_requests',lambda p:{'passed':True,'model_calls':5})
    monkeypatch.setattr(ar,'publish_proposals',lambda proposals,goal_id:publish_proposals(proposals,goal_id,tmp_path/'library'))
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})
    store=ar.create(tmp_path/'goal',profile,'Develop a reusable length calculation tool',dict(max_cycles=3,model_calls=36,tool_calls=36,wall_seconds=360))
    attempts=[];reviews=[]
    async def consult(role,context):
        if role=='PI':
            if attempts:
                assert context['progress']['publication_reviews'][0]['status']=='needs_revision'
                assessment=ar.AutonomousResearch(store,'fixture',{}).evaluate(store.read())
                assert assessment['criteria']['research_plan_executed']['passed']
                assert assessment['criteria']['research_outputs_reviewed']['passed']
                assert not assessment['criteria']['capability_admission_resolved']['passed']
            return 'Calculate supplied lengths.'
        if role=='CB':return json.dumps({'summary':'Length tool','steps':[{'id':'calc','title':'Calculate','acceptance':'Run and save outputs'}],'needs_input':[]})
        if role=='SC':return json.dumps({'decision':'ready_for_host_validation','summary':'Research evidence is complete','issues':[]})
        result=await reviewer(context);reviews.append(1)
        if len(reviews)==1:result.update(decision='revise',issues=['Document applicability more clearly'])
        return json.dumps(result)
    async def execute(request,service,events):
        attempts.append(1)
        await service.write_research_file('length.py','def run(inputs):\n return {"value":len(inputs["sequence"])}\n')
        await service.write_research_file('README.md','Length only; no biological inference. Revision '+str(len(attempts)))
        await service.propose_research_capability('Length','Length only','length.py',[],[{'sequence':'A'}],spec())
        await service.submit_research_checkpoint('Measured fixture length',['calc'],['length.py','README.md'],[])
        events.extend([{'kind':'tool_call_start'},{'kind':'completed','text':'Replayed candidate','event_audit':{'passed':True}}])
    state=await ar.AutonomousResearch(store,'fixture',{},consult=consult,executor=execute,tools=[]).run()
    assert state['status']=='completed',state
    assert len(attempts)==2 and state['cycles'][0]['status']=='needs_revision'
    assert state['progress']['research_completed'] and state['progress']['publication_reviews'][0]['status']=='approved'
