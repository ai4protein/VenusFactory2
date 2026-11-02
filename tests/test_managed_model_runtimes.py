import argparse
import json
import sys
from pathlib import Path

import pytest
import mcp_tools.models.execution as base
import mcp_tools.models.pretrained.runners.proteinmpnn as proteinmpnn
import mcp_tools.models.pretrained.runners.mutation as mutation
import mcp_tools.models.factory.runners.finetuned as finetuned
import mcp_tools.models.factory.runners.checkpoint as checkpoint


@pytest.fixture(autouse=True)
def profiles(monkeypatch):
    monkeypatch.setattr(base,'load_runtime_settings',lambda:{})


@pytest.mark.parametrize('variant',list(proteinmpnn.VARIANTS))
def test_mpnn_variant_preserves_every_prediction_parameter(tmp_path,variant):
    weights=tmp_path/'weights';folder=weights/proteinmpnn.VARIANTS[variant];folder.mkdir(parents=True)
    (folder/'v_48_020.pt').write_bytes(b'fixture')
    args=argparse.Namespace(ca_only=variant=='ca',use_soluble_model=variant=='soluble',model_name='v_48_020',
        path_to_model_weights='legacy-wrong-path',out_folder=str(tmp_path/'output'),pdb_path=str(tmp_path/'input.pdb'),
        num_seq_per_target=8,batch_size=2,sampling_temp='0.1 0.2',seed=23,score_only=1,save_probs=1,
        conditional_probs_only=0,unconditional_probs_only=0,backbone_noise=0.02,omit_AAs='CX',
        fixed_positions_jsonl=str(tmp_path/'fixed.jsonl'),tied_positions_jsonl=str(tmp_path/'tied.jsonl'))
    actual=proteinmpnn.prepare(args,{'proteinmpnn':{'weights':str(weights)}})
    assert actual['path_to_model_weights']==str(folder)
    assert all(actual[k]==v for k,v in vars(args).items() if k!='path_to_model_weights')


def test_mpnn_disabled_or_incomplete_never_uses_legacy_weights(tmp_path):
    args=argparse.Namespace(model_name='v_48_020',out_folder=str(tmp_path))
    with pytest.raises(ValueError,match='disabled'):proteinmpnn.prepare(args,{'proteinmpnn':{'enabled':False}})
    with pytest.raises(ValueError,match='checkpoint is not installed'):
        proteinmpnn.prepare(args,{'proteinmpnn':{'weights':str(tmp_path)}})
    args.ca_only=True;args.use_soluble_model=True
    with pytest.raises(ValueError,match='cannot be combined'):proteinmpnn.prepare(args,{})


def test_mpnn_design_and_score_share_dispatch(monkeypatch):
    from mcp_tools.models.pretrained.design.proteinmpnn import protein_mpnn_run
    calls=[];monkeypatch.setattr(proteinmpnn,'run',lambda args:calls.append(args.score_only))
    protein_mpnn_run.proteinmpnn_run(argparse.Namespace(score_only=0))
    protein_mpnn_run.proteinmpnn_run(argparse.Namespace(score_only=1))
    assert calls==[0,1]


def test_mcp_gateway_imports_canonical_model_package():
    import os,subprocess
    root=Path(__file__).resolve().parents[1]
    code='from mcp_tools import mcp_server; from mcp_tools.models.pretrained.design import tools_mcp; assert tools_mcp.proteinmpnn_design.__module__ == "mcp_tools.models.pretrained.design.proteinmpnn.protein_mpnn_function"'
    result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True,
        env={**os.environ,'PYTHONPATH':str(root)+os.pathsep+str(root/'src')},timeout=90)
    assert result.returncode==0,result.stderr[-4000:]


def test_protssn_profile_and_prosst_catalog_alias(tmp_path):
    source=tmp_path/'protein.pdb';source.write_text('END\n')
    cfg={'protssn':{'weights':str(tmp_path),'model_id':'custom/backbone','use_ensemble':False}}
    spec=mutation.build_command('ProtSSN','structure',source,tmp_path/'scores.csv',settings=cfg)
    assert spec.profile=='protssn' and '--single_model' in spec.argv
    assert spec.argv[spec.argv.index('--plm_model')+1]=='custom/backbone'
    assert spec.argv[spec.argv.index('--gnn_model_path')+1]==str(tmp_path)
    spec=mutation.build_command('ProSST-2048','structure',source,tmp_path/'scores.csv',settings={})
    assert spec.profile=='prosst'


def test_prosst_catalog_name_routes_through_runtime(monkeypatch,tmp_path):
    from mcp_tools.common import prediction_runners
    source=tmp_path/'protein.pdb';source.write_text('END\n');calls=[]
    def fake(spec):
        calls.append(spec.profile)
        Path(spec.argv[spec.argv.index('--output_csv')+1]).write_text('mutant,prosst_score\nM1A,0.1\n')
    monkeypatch.setattr(mutation,'execute',fake);monkeypatch.setattr(mutation,'load_runtime_settings',lambda:{})
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path))
    message,table=prediction_runners.run_zero_shot_prediction('structure','ProSST-2048',str(source))
    assert calls==['prosst'] and len(table)==1,message


def test_finetuned_uses_artifact_arguments_not_runtime_weights(monkeypatch,tmp_path):
    calls=[];monkeypatch.setattr(finetuned,'execute',lambda spec:calls.append(spec))
    loader=finetuned.PROJECT/'mcp_tools/models/factory/prediction/esm2.py'
    for dataset in ['DeepSol','SoluE']:
        finetuned.run_cli([sys.executable,str(loader),'--adapter_path',str(tmp_path/dataset)])
    assert all(c.profile=='finetuned' for c in calls)
    assert calls[0].argv[-1]!=calls[1].argv[-1]
    monkeypatch.setattr(base,'load_runtime_settings',lambda:{'finetuned':{'weights':'one-global-checkpoint'}})
    with pytest.raises(ValueError,match='artifact'):finetuned.run_cli([sys.executable,str(loader)])
    with pytest.raises(ValueError,match='Unknown'):finetuned.run_cli([sys.executable,'/tmp/untrusted.py'])


def test_checkpoint_loader_policy_and_environment(monkeypatch):
    monkeypatch.setenv('DMXAPI_API_KEY','not-forwarded');monkeypatch.setenv('TAVILY_API_KEY','not-forwarded')
    spec=checkpoint.prepare(['python','src/predict.py','--model_path','ckpt/my-model/best.pt'])
    assert spec.profile=='plm_checkpoint' and spec.argv[0]==sys.executable
    assert spec.argv[-1]=='ckpt/my-model/best.pt'
    assert 'DMXAPI_API_KEY' not in spec.env and 'TAVILY_API_KEY' not in spec.env
    with pytest.raises(ValueError,match='Unknown'):checkpoint.prepare(['python','src/train.py'])


def test_protssn_explicit_missing_weights_does_not_download(tmp_path,monkeypatch):
    from mcp_tools.models.pretrained.runners.protssn_weights import model_directory
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',lambda **kw:pytest.fail('Should not download configured weights'))
    with pytest.raises((ValueError,FileNotFoundError)):model_directory(tmp_path,False,10)
