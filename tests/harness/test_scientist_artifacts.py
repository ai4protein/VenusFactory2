"""Release behavior is checked with controlled code, never an external model."""
import copy
import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from agent.rsi.artifacts.package import build_delivery
from agent.rsi.artifacts.contract import family_id, selected_deliveries
from agent.harness.capability_review import review_proposal
from agent.harness.research_workspace import run_python
from agent.harness.capability_catalog import register, cards, search, require_registered
from backend.web_v2.research_delivery import export_project
from test_capability_admission import candidate, spec, reviewer


@pytest.mark.asyncio
async def test_method_release_is_runnable_and_preserves_only_selected_runtime(tmp_path):
    service = await candidate(tmp_path, spec())
    receipt = await review_proposal(service.proposals[0], reviewer=reviewer, control=tmp_path/'review', runner=run_python)
    assert receipt['status'] == 'approved'
    await service.write_research_file('experiments/abandoned.py', 'raise RuntimeError("old trial")')
    await service.write_research_file('raw_measurements.csv', 'private,measurement\n')
    release = build_delivery(service.workspace, service.proposals[0], receipt, goal_id='topic-one')
    assert release['kind'] == 'method'
    dest = tmp_path/'released'
    with zipfile.ZipFile(service.workspace/release['package']) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert {'README.md','METHOD_CARD.md','capability.json','predict.py','runtime/length.py','evaluation/test-report.json','tests/test_interface.py'} <= set(archive.namelist())
        assert not any('abandoned' in p or 'raw_measurements' in p for p in archive.namelist())
        checksums = json.loads(archive.read('checksums.json'))
        for name, sha in checksums.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
        archive.extractall(dest)
    result = subprocess.run([sys.executable, 'predict.py', 'examples/input.json'], cwd=dest, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'value': 2}
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], cwd=dest, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    state = {'status':'completed','definition':{'goal':{'objective':'Develop a length method'}},
             'progress':{'deliveries':[release],'research_completed':True},'cycles':[]}
    folder = tmp_path
    service.workspace.rename(folder/'workspace')
    with export_project(folder,state) as stream, zipfile.ZipFile(stream) as archive:
        names = archive.namelist()
        assert any(name.endswith('/runtime/length.py') for name in names)
        assert not any(name.endswith('artifact.zip') or 'abandoned.py' in name or 'raw_measurements.csv' in name for name in names)
    with export_project(folder,state,include_working_files=True) as stream, zipfile.ZipFile(stream) as archive:
        assert 'project/experiments/abandoned.py' in archive.namelist()
    (folder/'workspace'/release['package']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='changed'):
        export_project(folder,state)


def manifest(identity, version='1.0', protein='P1', name='Predictor'):
    contract=spec();contract['version']=version;contract['scope']['proteins']=[protein]
    return {'id':identity,'name':name,'description':'Test fixture','validation':{},'review_spec':contract}


def test_discovery_selects_one_version_per_topic_and_scope_without_erasing_history(tmp_path):
    receipt={'receipt_sha256':'a'*64}
    first=register(tmp_path,manifest('1'*64),receipt,'topic-one')
    second=register(tmp_path,manifest('2'*64,'1.1',name='Renamed predictor'),receipt,'topic-one')
    assert first['family_id']==second['family_id']
    assert [card['id'] for card in cards(tmp_path)]==['2'*64]
    assert search(tmp_path)['total']==1
    require_registered(tmp_path,'1'*64,receipt['receipt_sha256'])
    register(tmp_path,manifest('3'*64,protein='P2',name='P2 predictor'),receipt,'topic-one')
    register(tmp_path,manifest('4'*64),receipt,'topic-two')
    assert len(cards(tmp_path))==3
    # Re-selecting the simpler approved version is legitimate and does not mutate it.
    register(tmp_path,manifest('1'*64),receipt,'topic-one')
    assert {'1'*64,'3'*64,'4'*64}=={card['id'] for card in cards(tmp_path)}
    with pytest.raises(ValueError,match='new version'):
        register(tmp_path,manifest('5'*64),receipt,'topic-one')


def test_delivery_selection_retains_distinct_scopes_and_legacy_records():
    rows=[{'candidate_id':'old','family_id':'f'}, {'candidate_id':'other','family_id':'g'},
          {'candidate_id':'new','family_id':'f'}, {'candidate_id':'legacy'}]
    assert {x['candidate_id'] for x in selected_deliveries(rows)}=={'other','new','legacy'}
    a=spec()['scope'];b=copy.deepcopy(a);b['limitations']=['New limitation']
    assert family_id('topic','tool',a)==family_id('topic','tool',b)
    assert family_id('topic','tool',a)!=family_id('other topic','tool',a)
