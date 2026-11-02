"""Real CPU/sandbox execution; synthetic protocol tests, not biological results."""
import copy
import json
from pathlib import Path
import zipfile

import pytest

from agent.harness.research_workspace import ResearchWorkspace, run_python, publish_proposals, run_capability
from agent.harness.research_development import Development, validate_data, build_delivery, model_goal
from agent.harness.capability_review import review_proposal


def data():
    rows=lambda start,count:[{'id':str(i),'group':str(i),'inputs':{'x':i/10},'target':(i/10)**2} for i in range(start,start+count)]
    return {'task':'regression','provenance':'Synthetic quadratic fixture; no biological claim','grouping':'Unique synthetic groups',
        'splits':{'train':rows(1,8),'validation':rows(9,3),'test':rows(12,3)},'claimed_metric':0.0}


SOURCE='''import json
def run(inputs):
    x=inputs.get('x')
    if type(x) not in (int,float): raise ValueError('Numeric x required')
    state=json.load(open('weights.json'))
    return {'value':sum(w*x**i for i,w in enumerate(state['coefficients']))}
'''
TRAIN='''import json,numpy as np
data=json.load(open('eval.json'));rows=data['splits']['train']
coef=np.linalg.lstsq(np.array([[1,r['inputs']['x'],r['inputs']['x']**2] for r in rows]),np.array([r['target'] for r in rows]),rcond=None)[0]
json.dump({'coefficients':coef.tolist()},open('weights.json','w'))
'''


async def measured(tmp_path):
    service=ResearchWorkspace(tmp_path/'workspace',tmp_path/'control',{'steps':[],'deliverable':'model'},development_root=tmp_path/'development')
    await service.write_research_file('predict.py',SOURCE)
    await service.write_research_file('eval.json',json.dumps(data()))
    await service.write_research_file('weights.json',json.dumps({'coefficients':[.3]}))
    baseline=await service.evaluate_research_model('baseline','predict.py',['weights.json'],'eval.json','value','Constant baseline','supervised',variant='baseline')
    await service.write_research_file('train.py',TRAIN)
    result=await service.run_research_python('train.py',20);assert result['exit_code']==0,result
    candidate=await service.evaluate_research_model('quadratic learner','predict.py',['weights.json'],'eval.json','value','Test a nonlinear representation with sparse observations','few_shot',baseline['id'])
    return service,baseline,candidate


def review_spec(identity):
    return {'kind':'model','author':'Software protocol tests','version':'1.0','scope':{'proteins':['none: synthetic fixture'],'properties':['quadratic response'],'scenarios':['software testing'],'limitations':['not biological validation; not a novel algorithm']},
        'input_schema':{'type':'object','required':['x'],'properties':{'x':{'type':'number'}},'additionalProperties':False},
        'output_schema':{'type':'object','required':['value'],'properties':{'value':{'type':'number'}},'additionalProperties':False},
        'dependencies':['numpy'],'cases':[{'inputs':{'x':2},'expected':{'value':4}},{'inputs':{},'reject':True}],
        'evidence_files':['README.md','train.py','eval.json'],'evaluation_file':'eval.json','prediction_key':'value','reproduction_script':'train.py',
        'fitted_files':['weights.json'],'development_experiment_id':identity}


async def independent(context):
    assert {'learning_strategy','evolution_evidence'}<=set(context['required_checks'])
    assert context['host_evidence']['development']['generations'][0]['variant']=='baseline'
    return {'decision':'approve','summary':'Synthetic software fixture only; host replay and training reproducibility checked. No biological or novelty claim.',
        'checks':{k:{'passed':True,'reason':'Controlled fixture with explicit nonbiological scope'} for k in context['required_checks']},'issues':[],
        'tests':[{'inputs':{'x':3},'expected':{'value':9}},{'inputs':{'x':'bad'},'reject':True}]}


@pytest.mark.asyncio
@pytest.mark.parametrize('omit_claim',[False,True])
async def test_measured_lineage_independent_review_delivery_and_agent_reuse(tmp_path,omit_claim):
    service,baseline,candidate=await measured(tmp_path)
    if omit_claim:
        value=data();value.pop('claimed_metric')
        await service.write_research_file('eval.json',json.dumps(value))
    assert candidate['validation']['score']<1e-10 and candidate['comparison']['improved_on_validation']
    assert candidate['parent']==baseline['id'] and not candidate['test_used_for_feedback']
    # A fresh episode sees the same host ledger.
    next_episode=ResearchWorkspace(service.workspace,tmp_path/'episode2',{'steps':[]},development_root=tmp_path/'development')
    assert (await next_episode.get_research_strategy_guide())['development']['count']==2
    await service.write_research_file('README.md','Synthetic quadratic regression using a fixed nonlinear basis. Not biological accuracy, novelty, or statistically proven RSI. Eight observations; no PLM.')
    await service.propose_research_capability('Fixture learner','Nonlinear small-data regression fixture','predict.py',['weights.json'],[{'x':1.5}],review_spec(candidate['id']))
    bundle=service.proposals[0]
    receipt=await review_proposal(bundle,reviewer=independent,control=tmp_path/'review',runner=run_python)
    assert receipt['status']=='approved',receipt
    delivery=build_delivery(service.workspace,bundle,receipt)
    with zipfile.ZipFile(service.workspace/delivery['model_package']) as archive:
        assert {'model/predict.py','model/weights.json','model/train.py','MODEL_CARD.md','test-report.json','checksums.json'}<=set(archive.namelist())
        assert 'model/eval.json' not in archive.namelist()
        assert json.loads(archive.read('test-report.json'))['final_test']['training_reproduced']
    library=tmp_path/'private_capabilities'
    entry=publish_proposals([bundle],'fixture',library)[0]
    output=await run_capability(entry['id'],{'x':2.5},library=library)
    assert output['value']==pytest.approx(6.25)


@pytest.mark.asyncio
async def test_changed_runtime_data_or_fake_metrics_cannot_replace_host_receipt(tmp_path):
    service,baseline,candidate=await measured(tmp_path)
    assert candidate['validation']['score']!=123456
    await service.write_research_file('experiments/'+candidate['id']+'.json','{"score":123456}')
    assert service.development.record(candidate['id'])['validation']['score']<1e-10
    with pytest.raises(ValueError,match='must change'):
        await service.evaluate_research_model('renamed','predict.py',['weights.json'],'eval.json','value','Rename only','custom',candidate['id'])
    modified=data();modified['splits']['validation'][0]['target']+=100
    await service.write_research_file('eval.json',json.dumps(modified))
    with pytest.raises(ValueError,match='identical data'):
        await service.evaluate_research_model('bad split','predict.py',['weights.json'],'eval.json','value','Change evaluation','custom',candidate['id'])
    with pytest.raises(ValueError,match='differs'):
        service.development.selected(candidate['id'],{'source':'predict.py','files':{}},data())


@pytest.mark.asyncio
async def test_baseline_can_be_retained_after_measured_revisions(tmp_path):
    service,baseline,candidate=await measured(tmp_path)
    selected=service.development.selected(baseline['id'],{'source':baseline['source'],'files':baseline['files']},data())
    assert selected['selected_experiment_id']==baseline['id']
    assert [r['id'] for r in selected['generations']]==[baseline['id'],candidate['id']]


@pytest.mark.asyncio
async def test_measured_generations_are_visible_before_the_episode_finishes(tmp_path):
    from types import SimpleNamespace
    from agent.harness.autonomous_research import AutonomousResearch
    service,baseline,candidate=await measured(tmp_path)
    state={'status':'active','progress':{}}
    runner=AutonomousResearch(SimpleNamespace(mutate=lambda update:update(state)),'fixture',{},tools=[])
    runner.record_development_progress({'tool':'evaluate_research_model','status':'running'},service.development)
    assert 'development' not in state['progress']
    runner.record_development_progress({'tool':'mcp__venusfactory__evaluate_research_model','status':'completed'},service.development)
    assert state['status']=='active'
    rows=state['progress']['development']['generations']
    assert [row['id'] for row in rows]==[baseline['id'],candidate['id']]
    assert rows[1]['validation']['score']==candidate['validation']['score']


@pytest.mark.asyncio
async def test_model_goal_cannot_submit_an_unmeasured_tool(tmp_path):
    service=ResearchWorkspace(tmp_path/'w',tmp_path/'c',{'steps':[],'deliverable':'model'})
    with pytest.raises(ValueError,match='fitted model'):
        await service.propose_research_capability('tool','description','a.py',[],[{}])


@pytest.mark.parametrize('fault',['groups','inputs','unlabeled','tasks'])
def test_advanced_strategy_prerequisites_are_checked(fault):
    value=data();strategy='supervised'
    if fault=='groups':value['splits']['test'][0]['group']=value['splits']['train'][0]['group']
    if fault=='inputs':value['splits']['validation'][0]['inputs']=value['splits']['train'][0]['inputs']
    if fault=='unlabeled':strategy='semi_supervised'
    if fault=='tasks':strategy='meta_learning'
    with pytest.raises(ValueError):validate_data(value,strategy)


@pytest.mark.asyncio
async def test_real_semisupervised_training_and_host_scoring(tmp_path):
    service=ResearchWorkspace(tmp_path/'w',tmp_path/'c',{'steps':[]})
    value=data();value['task']='classification'
    for split,rows in value['splits'].items():
        for i,row in enumerate(rows):
            row['inputs']={'x':(-1 if i%2 else 1)*(1+int(row['id'])/100)};row['target']=i%2
    value['unlabeled']=[{'id':'u'+str(i),'inputs':{'x':(-1 if i%2 else 1)*(.3+i/40)}} for i in range(30)]
    await service.write_research_file('eval.json',json.dumps(value))
    await service.write_research_file('predict.py','def run(inputs):\n return {"value":0}\n')
    first=await service.evaluate_research_model('majority','predict.py',[],'eval.json','value','Labeled-only constant reference','supervised',variant='baseline')
    await service.write_research_file('train.py','''import json,joblib,numpy as np
from sklearn.semi_supervised import LabelSpreading
d=json.load(open('eval.json'));l=d['splits']['train'];u=d['unlabeled']
x=np.array([[r['inputs']['x']] for r in l+u]);y=np.array([r['target'] for r in l]+[-1]*len(u))
m=LabelSpreading(kernel='rbf',gamma=5).fit(x,y);joblib.dump(m,'weights.pkl')
''')
    await service.write_research_file('predict.py',"import joblib\ndef run(inputs):\n return {'value':int(joblib.load('weights.pkl').predict([[inputs['x']]])[0])}\n")
    result=await service.run_research_python('train.py',30);assert result['exit_code']==0,result
    child=await service.evaluate_research_model('label spreading','predict.py',['weights.pkl'],'eval.json','value','Use unlabeled neighborhood structure','semi_supervised',first['id'])
    assert child['validation']['score']==1


@pytest.mark.asyncio
async def test_real_meta_initialization_and_heldout_task_adaptation(tmp_path):
    service=ResearchWorkspace(tmp_path/'w',tmp_path/'c',{'steps':[]})
    def rows(task,offset,count):
        return [{'id':task+str(i),'group':task,'task_id':task,'inputs':{'x':i/10,'support':[{'x':-.5,'target':offset-1},{'x':.5,'target':offset+1}]},'target':offset+2*i/10} for i in range(1,count+1)]
    value={'task':'regression','provenance':'Synthetic task family, software test only','grouping':'Entire tasks held out',
        'splits':{'train':rows('a',.1,5)+rows('b',.3,5),'validation':rows('c',.5,3),'test':rows('d',.7,3)},'claimed_metric':0}
    await service.write_research_file('eval.json',json.dumps(value))
    await service.write_research_file('predict.py',"def run(inputs):\n return {'value':0.0}\n")
    baseline=await service.evaluate_research_model('no adaptation','predict.py',[],'eval.json','value','No learned prior and no support adaptation','supervised',variant='baseline')
    await service.write_research_file('train.py','''import json,numpy as np
d=json.load(open('eval.json'));rows=d['splits']['train'];theta=np.zeros(2)
tasks=sorted(set(r['task_id'] for r in rows))
for episode in range(100):
 t=tasks[episode%len(tasks)];batch=[r for r in rows if r['task_id']==t]
 x=np.array([[1,r['inputs']['x']] for r in batch]);y=np.array([r['target'] for r in batch]);local=theta.copy()
 for step in range(5):local-=.3*(x.T@(x@local-y)/len(y))
 theta+=.2*(local-theta)
json.dump(theta.tolist(),open('weights.json','w'))
''')
    await service.write_research_file('predict.py','''import json,numpy as np
def run(inputs):
 theta=np.array(json.load(open('weights.json')));support=inputs['support']
 x=np.array([[1,r['x']] for r in support]);y=np.array([r['target'] for r in support])
 for step in range(20):theta-=.4*(x.T@(x@theta-y)/len(y))
 return {'value':float(np.array([1,inputs['x']])@theta)}
''')
    result=await service.run_research_python('train.py',20);assert result['exit_code']==0,result
    candidate=await service.evaluate_research_model('meta initialization','predict.py',['weights.json'],'eval.json','value','Learn an initialization across training tasks and adapt to separate-task support observations','meta_learning',baseline['id'])
    assert candidate['comparison']['improved_on_validation']
    assert candidate['validation']['score']<.2


def test_model_intent_does_not_force_training_for_every_analysis():
    assert model_goal('开发一个蛋白溶解度模型') and model_goal('Build a new predictor')
    assert not model_goal('分析已有模型的结果') and not model_goal('Compute peptide lengths')


def test_replanning_remaining_work_preserves_completed_requirements():
    from agent.harness.autonomous_research import parse_plan
    old={'summary':'Fit and deliver','steps':[{'id':'fit','title':'Fit','acceptance':'Real weights','status':'completed','evidence':[{'path':'weights.json'}]},
        {'id':'deliver','title':'Deliver','acceptance':'Reviewed model','status':'pending','evidence':[]}],'needs_input':[],'deliverable':'model'}
    remaining={'summary':'Finish delivery','steps':[{'id':'deliver','title':'Deliver','acceptance':'Just a script'}],'needs_input':[]}
    revised=parse_plan(json.dumps(remaining),old)
    assert revised['steps'][0]==old['steps'][0]
    assert revised['steps'][1]['acceptance']=='Reviewed model'
    assert revised['deliverable']=='model'


@pytest.mark.asyncio
async def test_model_goal_cannot_complete_with_only_a_script_and_friendly_critic(tmp_path,monkeypatch):
    from agent.harness import autonomous_research as ar, Profile
    class Relay:
        calls=0;base_url='unused';token='fixture'
        def __init__(self,**kw):Path(kw['output']).touch()
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
    monkeypatch.setattr(ar,'EvidenceGateway',Relay)
    monkeypatch.setattr(ar,'audit_requests',lambda p:{'passed':True,'model_calls':4})
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})
    store=ar.create(tmp_path/'goal',profile,'Develop a new protein prediction model',dict(max_cycles=1,model_calls=12,tool_calls=12,wall_seconds=120))
    async def consult(role,context):
        if role=='PI':return 'Model development'
        if role=='CB':return json.dumps({'summary':'Train a model','steps':[{'id':'fit','title':'Train','acceptance':'Deliver a fitted predictor'}],'needs_input':[]})
        return json.dumps({'decision':'ready_for_host_validation','summary':'Looks complete','issues':[]})
    async def executor(request,service,events):
        await service.write_research_file('train.py','print("not a model")')
        await service.submit_research_checkpoint('Training script',['fit'],['train.py'],[])
        events.append({'kind':'completed','text':'Script ready','event_audit':{'passed':True}})
    state=await ar.AutonomousResearch(store,'fixture',{},consult=consult,executor=executor,tools=[]).run()
    assert state['status']!='completed'
    assert not state['progress']['research_completed']
    assert state['progress']['delivery_pending']


@pytest.mark.asyncio
@pytest.mark.parametrize('statuses',[[502,200],[502,502],[403]])
async def test_role_transport_retries_once_without_hiding_auth_failures(tmp_path,monkeypatch,statuses):
    from types import SimpleNamespace
    import httpx
    from agent.harness import autonomous_research as ar,Profile
    original=httpx.AsyncClient;calls=[]
    def handler(request):
        status=statuses[len(calls)];calls.append(status)
        return httpx.Response(status,json={'choices':[{'finish_reason':'stop','message':{'content':'Direction checked'}}]})
    monkeypatch.setattr(ar.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})
    runner=ar.AutonomousResearch(None,'fixture',{})
    if statuses[-1]==200:
        assert await runner.consult('PI',{},profile,SimpleNamespace(base_url='http://fixture',token='fixture',calls=0),tmp_path)=='Direction checked'
    else:
        with pytest.raises(httpx.HTTPStatusError):await runner.consult('PI',{},profile,SimpleNamespace(base_url='http://fixture',token='fixture',calls=0),tmp_path)
    assert calls==statuses
def test_plan_ignores_model_claimed_completion_and_preserves_host_evidence():
    from agent.harness.autonomous_research import parse_plan
    previous={'summary':'verified','needs_input':[],'steps':[
        {'id':'old','title':'Fit','acceptance':'actual fitted weights','status':'completed','evidence':['host-receipt']}]}
    raw={'summary':'continued','needs_input':[],'steps':[
        {'id':'old','title':'Fit','acceptance':'weaker claim','status':'pending','evidence':['invented']},
        {'id':'new','title':'Review','acceptance':'independent tests','status':'completed','evidence':['invented']}]}
    result=parse_plan(json.dumps(raw),previous)
    assert result['steps'][0]['status']=='completed'
    assert result['steps'][0]['evidence']==['host-receipt']
    assert result['steps'][0]['acceptance']=='actual fitted weights'
    assert result['steps'][1]['status']=='pending'
    assert result['steps'][1]['evidence']==[]


@pytest.mark.asyncio
async def test_free_json_mode_fallback_keeps_strict_plan_validation(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import httpx
    from agent.harness import autonomous_research as ar,Profile
    original=httpx.AsyncClient;payloads=[]
    def handler(request):
        payloads.append(json.loads(request.content))
        if len(payloads)==1:return httpx.Response(404,json={'error':'No JSON-mode endpoint'})
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'{"steps":[]}'}}]})
    monkeypatch.setattr(ar.httpx,'AsyncClient',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))
    profile=Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})
    raw=await ar.AutonomousResearch(None,'fixture',{}).consult('CB',{},profile,SimpleNamespace(base_url='http://fixture',token='fixture',calls=0),tmp_path)
    assert payloads[0]['response_format']=={'type':'json_object'}
    assert 'response_format' not in payloads[1]
    assert payloads[0]['model']==payloads[1]['model']
    with pytest.raises(ValueError,match='Invalid research plan'):ar.parse_plan(raw)
