import json
import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform, read
from backend.web_v2 import model_platform_api as api
from backend.web_v2 import workflow_tools


@pytest.fixture
def setup(tmp_path):
    p = Platform(tmp_path)
    owner = p.session(None)[1]
    p._cache, p._cache_time = (p.official(), []), time.monotonic()
    d = p.upload(owner, 'train.csv', ('x,target,sequence\n' + '\n'.join(f'{i},{i*2+3},ACDEFGHIK' for i in range(40))).encode())
    predict = p.upload(owner, 'predict.csv', b'x\n61\n62\n63')
    return p, owner, d, predict


def training_graph(d, predict):
    return {'nodes': [
        {'id': 'data', 'kind': 'data', 'dataset_id': d['id']},
        {'id': 'tool', 'kind': 'tool', 'model_id': 'tool:read_fasta'},
        {'id': 'train', 'kind': 'train', 'label': 'Workflow trained model', 'config': {'target': 'target', 'features': ['x'], 'problem': 'regression', 'method': 'linear'}},
        {'id': 'new_data', 'kind': 'data', 'dataset_id': predict['id']},
        {'id': 'predict', 'kind': 'predict'},
        {'id': 'model_result', 'kind': 'output'},
        {'id': 'predictions', 'kind': 'output'},
    ], 'edges': [{'source': a, 'target': b} for a,b in [('data','tool'),('tool','train'),('train','predict'),('new_data','predict'),('train','model_result'),('predict','predictions')]]}


def test_real_registered_tool_train_and_predict_chain(setup):
    p, owner, d, prediction = setup
    j = p.start(owner, 'workflow', {'graph': training_graph(d,prediction)}, launch=False)
    p.execute(j['id'])
    done = p.job(owner,j['id'])
    assert done['status']=='completed', done
    assert all(s['status']=='completed' for s in done['steps'])
    trained = done['result']['models'][0]
    assert done['result']['rows']==3
    assert trained['visibility']=='private' and trained['metrics']['R2'] > .99
    with pytest.raises(FileNotFoundError): p.model('another-owner',trained['id'])
    result = read(p.root/'results'/(j['id']+'.json'))
    assert result['outputs']['model_result']['kind']=='model'
    assert result['outputs']['model_result']['model']['id']==trained['id']
    assert len(result['outputs']['predictions']['rows'])==3
    assert abs(result['outputs']['predictions']['rows'][0]['target_prediction']-125)<5
    assert any(m['id']==trained['id'] for m in p.workflow_components(owner,source='community')['items'])
    step=next(s for s in done['steps'] if s['id']=='train')
    assert step['model']['id']==trained['id'] and step['progress']==100


def test_tool_catalog_category_source_search_and_pagination(setup):
    p, owner, _, _=setup
    a=p.workflow_components(owner,limit=10)
    b=p.workflow_components(owner,offset=10,limit=10)
    assert a['total']>50
    assert not {c['id'] for c in a['items']} & {c['id'] for c in b['items']}
    tools=p.workflow_components(owner,kind='tool',category='processing')
    assert tools['items'] and all(c['node_kind']=='tool' and c['category']=='processing' for c in tools['items'])
    found=p.workflow_components(owner,query='PubMed',kind='tool')
    assert [c['id'] for c in found['items']]==['tool:query_pubmed']
    assert not found['unavailable_tools']


def test_registered_tool_reads_managed_input_and_computes_properties(setup):
    p,owner,d,_=setup
    data=p.owned('datasets',d['id'],owner);data={**data,'rows':data['rows'][:2]}
    result=workflow_tools.run('calculate_physchem_from_fasta',data,{},p.root/'work')
    assert len(result['rows'])==2
    assert result['rows'][0]['x']=='0'
    assert 'file_info' not in result['rows'][0]
    assert any(isinstance(v,(int,float)) for v in result['rows'][0].values())


@pytest.mark.parametrize('config',[{'parameters':{'file_path':'/etc/passwd'}},{'columns':{'output_file':'x'}},{'parameters':{'api_key':'not-a-real-key'}}])
def test_managed_tool_parameters_cannot_be_overridden(setup,config):
    p,owner,d,_=setup
    with pytest.raises(ValueError,match='documented tool parameters'):
        workflow_tools.run('read_fasta',p.owned('datasets',d['id'],owner),config,p.root/'work')


def test_tool_error_not_reported_as_success():
    with pytest.raises(ValueError,match='missing'):
        workflow_tools.result_rows(json.dumps({'status':'error','error':{'message':'missing'}}))
    with pytest.raises(ValueError):workflow_tools.card('agent_generated_code')
    with pytest.raises(ValueError):workflow_tools.card('query_tavily')


def test_prediction_requires_distinct_model_and_table_inputs(setup):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    graph['edges']=[e for e in graph['edges'] if not (e['source']=='train' and e['target']=='predict')]
    with pytest.raises(ValueError,match='trained model and prediction data'):p.validate_graph(owner,graph)


def test_training_fails_without_publishing_and_marks_downstream_skipped(setup):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    graph['nodes'][2]['config']['target']='missing'
    j=p.start(owner,'workflow',{'graph':graph},launch=False);p.execute(j['id'])
    done=p.job(owner,j['id']);assert done['status']=='failed'
    assert next(s for s in done['steps'] if s['id']=='train')['status']=='failed'
    assert next(s for s in done['steps'] if s['id']=='predict')['status']=='skipped'
    assert not list((p.root/'models').glob('*.json'))


def test_workflow_component_http_contract_and_private_visibility(setup,monkeypatch):
    p,owner,d,pred=setup;monkeypatch.setattr(api,'service',p)
    app=FastAPI();app.include_router(api.router)
    with TestClient(app) as client:
        response=client.get('/api/model-platform/workflow-components?kind=tool&category=processing')
        assert response.status_code==200 and response.json()['items']
        tool=client.get('/api/model-platform/workflow-components/tool:query_pubmed').json()
        assert 'query' in tool['contract']['properties'] and tool['standalone']
        assert client.get('/api/model-platform/workflow-components/tool:query_tavily').status_code==422
        assert client.get('/api/model-platform/workflow-components?limit=101').status_code==422
        assert client.post('/api/model-platform/workflows/run',json={'graph':training_graph(d,pred)}).status_code==404


def test_http_training_workflow_completes_in_separate_worker(tmp_path,monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    p=Platform(tmp_path);monkeypatch.setattr(api,'service',p)
    app=FastAPI();app.include_router(api.router)
    with TestClient(app) as client:
        content=('x,target,sequence\n'+'\n'.join(f'{i},{i*2+3},ACDEFGHIK' for i in range(24))).encode()
        d=client.post('/api/model-platform/datasets',files={'file':('train.csv',content)}).json()
        pred=client.post('/api/model-platform/datasets',files={'file':('predict.csv',b'x\n50\n51')}).json()
        response=client.post('/api/model-platform/workflows/run',json={'graph':training_graph(d,pred)})
        assert response.status_code==200,response.text
        job=response.json()
        deadline=time.monotonic()+45
        while job['status'] in {'queued','running'} and time.monotonic()<deadline:
            time.sleep(.2)
            job=client.get('/api/model-platform/jobs/'+job['id']).json()
        assert job['status']=='completed',job
        assert job['result']['rows']==2 and job['result']['models']
        result=client.get(job['result']['download'])
        assert result.status_code==200
        assert len(result.json()['outputs']['predictions']['rows'])==2


def test_trace_has_dependency_order_actual_data_and_model_metrics(setup):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    graph['nodes'].reverse()  # UI position / array order is not execution order.
    job=p.start(owner,'workflow',{'graph':graph},launch=False);p.execute(job['id'])
    done=p.job(owner,job['id']);assert done['status']=='completed',done
    steps={s['id']:s for s in done['steps']}
    for edge in graph['edges']:
        assert steps[edge['source']]['finished'] <= steps[edge['target']]['started']
    assert steps['predict']['output']['rows']==3
    assert len(steps['predict']['inputs'])==2
    assert steps['train']['output']['model']['metrics']['R2']>.99
    assert steps['tool']['output']['preview'] and len(steps['tool']['output']['preview'])==3
    assert any(e['status']=='progress' and e['step_id']=='tool' for e in done['events'])
    assert [e['seq'] for e in done['events']]==sorted(set(e['seq'] for e in done['events']))
    with pytest.raises(FileNotFoundError):p.job('other-user',job['id'])


@pytest.mark.parametrize('source,port', [('train','data'), ('new_data','model'), ('new_data','unknown')])
def test_rejects_wrong_prediction_ports_before_enqueue(setup,source,port):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    next(e for e in graph['edges'] if e['source']==source and e['target']=='predict')['target_port']=port
    with pytest.raises(ValueError,match='port'):p.start(owner,'workflow',{'graph':graph},launch=False)
    assert not list((p.root/'jobs').glob('*.json'))


def test_direct_input_columns_checked_before_enqueue(setup):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    graph['nodes'][1]['config']={'sequence_column':'not_sequence'}
    with pytest.raises(ValueError,match='Missing input columns'):p.start(owner,'workflow',{'graph':graph},launch=False)


def test_preview_is_bounded_and_excludes_storage_credentials():
    from backend.web_v2.workflow_flow import summarize
    value={'kind':'table','owner':'private-owner','path':'/private/path','rows':[{'sequence':'A'*1000,'api_key':'private-key','path':'/private/path','payload':{'secret':'x'}}]*1000}
    result=summarize(value)
    assert result['rows']==1000 and len(result['preview'])==3
    assert len(result['preview'][0]['sequence'])==181
    assert not any(x in json.dumps(result) for x in ['private-owner','private-key','/private/path'])


def test_job_restores_its_submitted_graph_even_after_workflow_edit(setup):
    p,owner,d,pred=setup;graph=training_graph(d,pred)
    saved=p.save_workflow(owner,{'name':'Original workflow','graph':graph})
    job=p.start(owner,'workflow',{'workflow_id':saved['id'],'name':saved['name'],'graph':graph},launch=False)
    p.save_workflow(owner,{**saved,'name':'Edited workflow','graph':{'nodes':[],'edges':[]}})
    restored=p.job(owner,job['id'])['workflow']
    assert restored['name']=='Original workflow' and restored['graph']==graph
    assert restored['id']==saved['id']
