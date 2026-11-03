import json
import sys
import zipfile
from pathlib import Path
import pytest
from backend.web_v2 import external_models as external
from backend.web_v2.model_platform import Platform,write
from backend.web_v2.model_parameters import parameter_schema,validated_parameters


@pytest.fixture(autouse=True)
def mode_policy(monkeypatch):
    import os
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setattr(runtime_tool_policy,'is_online_mode',lambda:os.getenv('WEBUI_V2_MODE')=='online')


def fake_runtime(tmp_path,monkeypatch,key='vina'):
    executable=tmp_path/'vina';executable.write_text('fixture');executable.chmod(0o700)
    config=tmp_path/'runtimes.json';config.write_text(json.dumps({key:{'enabled':True,'executable':str(executable)}}))
    monkeypatch.setenv('VENUS_MODEL_RUNTIMES_CONFIG',str(config));monkeypatch.setenv('WEBUI_V2_MODE','local')
    return config


def test_typed_models_are_disabled_until_configured(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_MODEL_RUNTIMES_CONFIG',str(tmp_path/'absent'))
    cards=external.cards();assert len(cards)==7
    assert all(not c['available'] and c['disabled_reason']=='not_configured' for c in cards)
    assert all(parameter_schema(c['adapter'])['properties'] for c in cards)
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    assert all(Platform.availability(c)['disabled_reason']==('local_only' if c['id'] in {'external:bindcraft','external:boltzgen'} else 'not_configured') for c in cards)


def test_catalog_does_not_override_missing_runtime_or_online_policy(tmp_path,monkeypatch):
    fake_runtime(tmp_path,monkeypatch)
    platform=Platform(tmp_path/'platform')
    assert platform.model('owner','external:vina')['available']
    assert not platform.model('owner','external:ligandmpnn')['available']
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    assert platform.model('owner','external:vina')['available']
    with pytest.raises((ValueError, FileNotFoundError)):platform.start('owner','predict',{'model_id':'external:vina','dataset_id':'missing','config':{}},launch=False)


def test_parameters_reject_paths_nan_and_unbounded_counts():
    for payload in [{'center':[0,0,float('nan')]},{'center':[0,0,0],'executable':'sh'},{'center':[0,0,0],'size':[500,20,20]}]:
        with pytest.raises(ValueError):validated_parameters('external:vina',{'parameters':payload})
    with pytest.raises(ValueError):external.validate('ligandmpnn',{'parameters':{'batch_size':32,'number_of_batches':512}})
    with pytest.raises(ValueError):external.validate('antibmpnn',{'parameters':{'design_positions':'1 2','num_sequences':5,'batch_size':2}})


def test_vina_secondary_dataset_is_owned_and_coordinator_holds_no_compute_slot(tmp_path,monkeypatch):
    fake_runtime(tmp_path,monkeypatch)
    p=Platform(tmp_path/'platform');receptor=p.upload('a','receptor.pdbqt',b'ATOM  sample');ligand=p.upload('b','ligand.pdbqt',b'ATOM  ligand')
    config={'parameters':{'center':[0,0,0]},'ligand_dataset_id':ligand['id']}
    with pytest.raises(FileNotFoundError):p.start('a','predict',{'model_id':'external:vina','dataset_id':receptor['id'],'config':config},launch=False)
    ligand=p.upload('a','ligand.pdbqt',b'ATOM  ligand');config['ligand_dataset_id']=ligand['id']
    payload={'model_id':'external:vina','dataset_id':receptor['id'],'config':config}
    job=p.start('a','predict',payload,launch=False)
    assert job['status']=='queued' and p.resource_for('a','predict',payload)=='api:orchestration'
    graph={'nodes':[{'id':'input','kind':'data','dataset_id':receptor['id']},{'id':'dock','kind':'model','model_id':'external:vina','config':config},{'id':'result','kind':'output'}], 'edges':[{'source':'input','target':'dock'},{'source':'dock','target':'result'}]}
    assert p.validate_graph('a',graph)[0]==['input','dock','result']


def test_all_upstream_commands_and_output_paths(tmp_path):
    source=tmp_path/'input.pdb';source.write_text('ATOM  x');out=tmp_path/'out';out.mkdir()
    cfg={'python':sys.executable,'root':str(tmp_path),'weights':str(tmp_path/'weights'),'executable':'/configured/executable', 'filters':'/configured/filters.json','advanced':'/configured/advanced.json','cache':'/configured/cache'}
    defaults={'ligandmpnn':{},'antibmpnn':{'design_positions':'102 103'},'bindcraft':{},'boltzgen':{},'vina':{'center':[1,2,3]}}
    for key,values in defaults.items():
        p=external.validate(key,{'parameters':values});cmds=external.commands(key,p,cfg,source,tmp_path/'ligand.pdbqt',tmp_path,out)
        assert all(isinstance(cmd,list) and all(isinstance(arg,str) for arg in cmd) for cmd in cmds)
        assert str(out) in ' '.join(cmds[-1]) or key=='bindcraft'
        if key=='vina':assert cmds[0][cmds[0].index('--cpu')+1]=='2'
        if key=='boltzgen':
            assert cmds[0][cmds[0].index('--devices')+1]=='1'
            assert cmds[0][cmds[0].index('--cache')+1]=='/configured/cache'
            assert json.loads((tmp_path/'design.yaml').read_text())['entities'][1]['file']['path']==str(source)
        if key=='antibmpnn':assert len(cmds)==4 and '--specify_non_fixed' in cmds[2]
    config=tmp_path/'base.yaml';config.write_text('inference_steps: 99\nactual_steps: 98\nprotein_path: old.pdb\n')
    cfg.update(config=str(config),model_dir='/configured/score',confidence_model_dir='/configured/confidence')
    cmds=external.commands('diffdock',external.validate('diffdock',{'parameters':{'ligand_smiles':'CCO','inference_steps':12}}),cfg,source,None,tmp_path,out)
    final=json.loads((tmp_path/'inference.yaml').read_text())
    assert final['actual_steps']==12
    assert final['inference_steps']==12 and final['protein_path']==str(source)
    assert final['ligand_description']==str(tmp_path/'ligand.sdf')
    assert cmds[0][-2]=='CCO' and len(cmds)==2


def test_model_outputs_and_download_ownership(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'poses.pdbqt').write_text('MODEL 1\nREMARK VINA RESULT: -7.1 0.0 0.0\nENDMDL\n')
    assert external.collect('vina',out)[0]['affinity_kcal_mol']==-7.1
    (out/'design.fa').write_text('>native\nACDE\n>design\nACDF\n')
    assert len(external.collect('ligandmpnn',out))==2
    p=Platform(tmp_path/'platform');identity='a'*32;work=p.root/'work'/identity;work.mkdir(parents=True)
    artifact=work/'poses.pdbqt';artifact.write_text('ATOM  x')
    write(p.path('jobs',identity),{'id':identity,'owner':'a','status':'completed'})
    write(p.root/'results'/(identity+'.json'),{'artifacts':[str(artifact)]})
    with pytest.raises(FileNotFoundError):p.artifact_archive('b',identity)
    archive=p.artifact_archive('a',identity)
    with zipfile.ZipFile(archive) as z:assert z.namelist()==['poses.pdbqt']
    write(p.root/'results'/(identity+'.json'),{'artifacts':[str(out/'poses.pdbqt')]})
    with pytest.raises(ValueError):p.artifact_archive('a',identity)


def test_runtime_timeout_stops_child_processes(tmp_path):
    import os,psutil
    pidfile=tmp_path/'child.pid'
    script="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
    with (tmp_path/'runtime.log').open('wb') as log,pytest.raises(ValueError,match='time budget'):
        external.execute_command([sys.executable,'-c',script,str(pidfile)],cwd=tmp_path,env=dict(os.environ),log=log,timeout=1)
    pid=int(pidfile.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE


def test_design_outputs_do_not_promote_input_visualization_or_rejected_binders(tmp_path):
    (tmp_path/'design.cif').write_text('input visualization')
    with pytest.raises(ValueError,match='without model output'):
        external.collect('boltzgen',tmp_path)
    (tmp_path/'design.cif').unlink()
    rejected=tmp_path/'Trajectory/Clashing';rejected.mkdir(parents=True)
    (rejected/'candidate.pdb').write_text('ATOM')
    row=external.collect('bindcraft',tmp_path)[0]
    assert not row['accepted'] and row['outcome']=='candidate_not_accepted'


def test_gpu_runtime_requires_inference_receipt(tmp_path,monkeypatch):
    executable=tmp_path/'boltzgen';executable.write_text('fixture')
    cache=tmp_path/'cache';cache.mkdir()
    receipt=tmp_path/'installation.json';receipt.write_text('{"runtime":"boltzgen","verified":false}')
    cfg=tmp_path/'runtimes.json';cfg.write_text(json.dumps({'boltzgen':{'enabled':True,'executable':str(executable),'cache':str(cache),'installation':str(receipt)}}))
    monkeypatch.setenv('VENUS_MODEL_RUNTIMES_CONFIG',str(cfg))
    assert external.availability('boltzgen')==(False,'not_configured')
    receipt.write_text('{"runtime":"boltzgen","verified":true}')
    assert external.availability('boltzgen')==(True,None)
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    assert external.availability('boltzgen')==(False,'local_only')
