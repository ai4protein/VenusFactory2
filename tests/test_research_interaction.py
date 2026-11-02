"""Real file/revision boundaries, with no external model requests."""
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from agent.harness import Profile
from agent.science_goal import GoalStore, GoalSpec, run_goal
from agent.harness.contracts import file_hash
from agent.harness.research_updates import queue_revision, apply_revisions
from backend.web_v2.research_files import save_inputs, resolve_inputs, stage_inputs, list_files, resolve_file, archive_files


def goal(tmp_path):
    profile = Profile.from_dict({'mode':'science_expert','harness':'expert-deepseek-harness','model':{'provider':'dmx','model':'glm-4-flash','base_url':'https://www.dmxapi.cn/v1'},'budget':{'model_calls':12,'tool_calls':12,'wall_seconds':120,'output_tokens':4096,'thinking':'disabled'}})
    return GoalStore.create(tmp_path/'goal', GoalSpec('Develop a model from supplied data', ('accepted',), max_cycles=3, model_calls=36, tool_calls=36, wall_seconds=360), profile)


def test_multiple_inputs_ownership_and_integrity(tmp_path):
    records = save_inputs(tmp_path/'a', [('data.csv', b'x,y\n1,2\n'), ('ligand.sdf', b'ligand\n$$$$\n')])
    ids = [r['id'] for r in records]
    with pytest.raises(FileNotFoundError): resolve_inputs(tmp_path/'b', ids)
    staged = stage_inputs(tmp_path/'a', tmp_path/'work', ids)
    assert len(staged) == 2
    assert (tmp_path/'work'/staged[0]['path']).read_bytes() == b'x,y\n1,2\n'
    assert stage_inputs(tmp_path/'a', tmp_path/'work', ids) == staged
    (tmp_path/'work'/staged[0]['path']).write_text('changed')
    with pytest.raises(ValueError): stage_inputs(tmp_path/'a', tmp_path/'work', ids)
    for name in ['.env','../data.csv','folder\\data.csv','payload.exe']:
        with pytest.raises(ValueError): save_inputs(tmp_path/'a', [(name,b'x')])


def test_files_scope_filter_and_zip(tmp_path):
    a=tmp_path/'a';a.mkdir();b=tmp_path/'b';b.mkdir()
    (a/'data.csv').write_text('x,y\n1,2');(a/'.env').write_text('hidden');(b/'report.pdf').write_bytes(b'%PDF-test')
    (a/'escape').symlink_to(b, target_is_directory=True)
    roots={'agent':a,'research-test':b}
    rows=list_files(roots)['files']
    assert {f['key'] for f in rows} == {'agent/data.csv','research-test/report.pdf'}
    for key in ['agent/../b/report.pdf','agent/escape/report.pdf','agent/.env','other/data.csv']:
        with pytest.raises((ValueError,FileNotFoundError)): resolve_file(roots,key)
    with archive_files(roots,[r['key'] for r in rows]) as stream:
        with zipfile.ZipFile(stream) as archive:
            assert set(archive.namelist()) == {r['key'] for r in rows}
            assert archive.read('agent/data.csv') == b'x,y\n1,2'


def test_queued_plan_survives_episode_and_invalidates_previous_evidence(tmp_path):
    store=goal(tmp_path)
    old={'summary':'old','steps':[{'id':'train','title':'Train','acceptance':'score > 0','status':'completed','evidence':[{'path':'old.csv'}]}]}
    store.mutate(lambda s:s['progress'].update(plan=old,reviewed_episode=0))
    changed={'summary':'group split','steps':[{'id':'train','title':'Retrain','acceptance':'Group holdout score'}]}
    queue_revision(store,'Use protein groups',changed,[{'path':'inputs/new.csv'}])
    # A running episode commits its captured progress after the user submits.
    store.mutate(lambda s:s.update(progress={'plan':old,'reviewed_episode':0}))
    state=apply_revisions(store)
    assert not state['pending_research_updates']
    step=state['progress']['plan']['steps'][0]
    assert step['acceptance']=='Group holdout score' and step['status']=='pending' and not step['evidence']
    assert 'reviewed_episode' not in state['progress']
    assert state['progress']['revision_history'][0]['previous_plan']['steps'][0]['status']=='completed'
    assert state['progress']['inputs'][0]['path']=='inputs/new.csv'
    assert apply_revisions(store)['progress']==state['progress']


@pytest.mark.asyncio
async def test_revision_arriving_during_assessment_cannot_complete_stale_plan(tmp_path):
    store=goal(tmp_path);evidence=store.root/'result.json';evidence.write_text('{}')
    visits=[]
    def evaluate(state):
        if not visits:
            visits.append('assessment')
            queue_revision(store,'Repeat with new conditions')
        return {'criteria':{'accepted':{'passed':True,'evidence':[{'path':'result.json','sha256':file_hash(evidence)}]}}}
    async def cycle(*args):
        state=apply_revisions(store);visits.append('cycle')
        return {'status':'completed','summary':'Revalidated','progress':state['progress'],'usage':{'model_calls':0,'tool_calls':0,'wall_seconds':0.01}}
    state=await run_goal(store,cycle,evaluate)
    assert state['status']=='completed' and visits==['assessment','cycle']
    assert state['progress']['user_updates']==['Repeat with new conditions']


def test_invalid_plan_does_not_queue(tmp_path):
    store=goal(tmp_path)
    with pytest.raises(ValueError): queue_revision(store,plan={'summary':'missing steps','steps':[]})
    store.mutate(lambda s:s.update(status='completed'))
    with pytest.raises(ValueError): queue_revision(store,'New request')
    assert not store.read().get('pending_research_updates')


def test_legacy_modes_return_unified_catalog_with_scientist_models(tmp_path,monkeypatch):
    from backend.web_v2.model_platform import Platform
    platform=Platform(tmp_path/'platform')
    cards=[{'id':'builtin:first','name':'First','task':'Solubility','category':'function','source':'platform','adapter':'function'}, {'id':'builtin:second','name':'Second','task':'Solubility','category':'function','source':'platform','adapter':'function'}, {'id':'backbone','name':'Backbone','category':'foundation','source':'platform','adapter':'foundation'}, {'id':'scientist:new','name':'Protein-specific solubility','category':'function','source':'scientist','adapter':'scientist'}]
    monkeypatch.setattr(platform,'shared',lambda:(cards,[]))
    quick=platform.catalog('owner',mode='quick')['items']
    advanced=platform.catalog('owner',mode='advanced')['items']
    assert {m['id'] for m in quick}=={m['id'] for m in cards}
    assert len(advanced)==4


def test_authenticated_file_routes(tmp_path,monkeypatch):
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    import backend.web_v2.chat_api.session_files as module
    work=tmp_path/'work';work.mkdir();(work/'model.json').write_text('{"ok":true}')
    async def get(sid):
        if sid!='owned':raise HTTPException(404)
        return {'owner':'token'}
    def access(state,request):
        if request.headers.get('X-Test-Session')!=state['owner']:raise HTTPException(403)
    monkeypatch.setattr(module,'_get_session_or_404',get)
    monkeypatch.setattr(module,'_assert_session_access',access)
    monkeypatch.setattr(module,'session_roots',lambda *a:{'agent':work})
    app=FastAPI();app.include_router(module.router);client=TestClient(app)
    url='/sessions/owned/files/content?path=agent/model.json'
    assert client.get(url).status_code==403
    headers={'X-Test-Session':'token'}
    assert client.get(url,headers=headers).json()=={'ok':True}
    assert client.get('/sessions/owned/files/content?path=agent/../model.json',headers=headers).status_code==404
    response=client.post('/sessions/owned/files/archive',json={'paths':['agent/model.json']},headers=headers)
    assert response.status_code==200
    assert zipfile.ZipFile(io.BytesIO(response.content)).read('agent/model.json')==b'{"ok":true}'


@pytest.mark.asyncio
async def test_research_reads_pdf(tmp_path):
    from agent.harness.research_workspace import ResearchWorkspace
    try:
        from pypdf import PdfWriter
    except ImportError:
        from PyPDF2 import PdfWriter
    service=ResearchWorkspace(tmp_path/'work',tmp_path/'control',{'steps':[]})
    writer=PdfWriter();writer.add_blank_page(width=100,height=100)
    with (service.workspace/'paper.pdf').open('wb') as file:writer.write(file)
    result=await service.read_research_file('paper.pdf')
    assert result['pages']==1 and result['sha256']


def test_answer_preserves_verified_work_but_requires_another_review(tmp_path):
    store = goal(tmp_path)
    plan = {'summary': 'Measured data', 'steps': [{'id': 'collect', 'title': 'Collect', 'acceptance': 'Curated data', 'status': 'completed', 'evidence': [{'path': 'data.csv'}]}]}
    store.mutate(lambda s: s['progress'].update(plan=plan, reviewed_episode=0, questions=['Preferred variant?']))
    queue_revision(store, 'No preferred variant; use public PETase data')
    state = apply_revisions(store)
    assert state['progress']['plan'] == plan
    assert state['progress']['user_updates'] == ['No preferred variant; use public PETase data']
    assert state['progress']['questions'] == []
    assert 'reviewed_episode' not in state['progress']
    assert state['assessment'] is None
