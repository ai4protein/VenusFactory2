from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.web_v2.model_platform import Platform
from backend.web_v2 import model_platform_api as api


def test_incomplete_workflow_save_reopen_update_and_owner_isolation(tmp_path, monkeypatch):
    service=Platform(tmp_path);monkeypatch.setattr(api,'service',service)
    app=FastAPI();app.include_router(api.router)
    graph={'nodes':[{'id':'input','kind':'data','label':'Input','x':40,'y':80}],'edges':[]}
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.get('/api/model-platform/session');bob.get('/api/model-platform/session')
        saved=alice.post('/api/model-platform/workflows',json={'name':'Draft','graph':graph})
        assert saved.status_code==200,saved.text
        draft=saved.json();identity=draft['id'];assert 'owner' not in draft
        route='/api/model-platform/workflows/'+identity
        assert alice.get(route).json()==draft
        assert bob.get(route).status_code==404
        assert bob.post('/api/model-platform/workflows',json={**draft,'name':'Other'}).status_code==404
        updated=alice.post('/api/model-platform/workflows',json={**draft,'name':'Renamed'})
        assert updated.status_code==200
        assert alice.get(route).json()['name']=='Renamed'
        assert len(alice.get('/api/model-platform/workflows').json())==1
        assert bob.get('/api/model-platform/workflows').json()==[]
        # Saving a partial document does not authorize executing it.
        assert alice.post('/api/model-platform/workflows/run',json={'graph':graph}).status_code==422
        assert not list((service.root/'jobs').glob('*.json'))
