import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from mcp_tools.models.factory.training import session_training as training


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_TOOL_WORKSPACE',str(tmp_path))
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path/'outputs'))
    from mcp_tools.tools import runtime_tool_policy as policy
    monkeypatch.setattr(policy,'is_online_mode',lambda:True)
    return tmp_path


def test_scoped_training_rejects_other_sessions_and_large_online_models(scoped):
    config={'plm_model':'ESM2-8M','output_dir':'models','dataset_custom':'data'}
    checked=training.validate_config(config)
    assert checked['plm_model']==training.ESM8M
    assert checked['output_dir']==str(scoped/'models')
    for changes in ({'plm_model':'ESM2-650M'},{'output_dir':'../other'},{'dataset_custom':'/etc'}, {'output_model_name':'../model.pt'}, {'batch_size':100000}):
        with pytest.raises(ValueError): training.validate_config({**config,**changes})
    (scoped/'link').symlink_to(scoped.parent)
    with pytest.raises(ValueError):training.scoped_path('link/private.csv')


def test_online_training_requires_trusted_scope(monkeypatch):
    monkeypatch.delenv('VENUS_TOOL_WORKSPACE',raising=False)
    from mcp_tools.tools import runtime_tool_policy as policy
    monkeypatch.setattr(policy,'is_online_mode',lambda:True)
    with pytest.raises(ValueError):training.scoped_path('/tmp/config.json')


def test_session_checkpoint_loader_cannot_execute_pickle(scoped):
    from mcp_tools.models.execution import environment
    assert environment({})['TORCH_FORCE_WEIGHTS_ONLY_LOAD']=='1'


def test_config_and_loader_preserve_supplied_validation_and_hold_out_test(scoped):
    import pandas as pd
    from mcp_tools.models.factory.training.train_operations import process_csv_and_generate_config
    from mcp_tools.models.factory.training.data.dataloader import _load_dataset_dict
    alphabet='ACDEFGHIKLMNPQRSTVWY'
    sequences=['M'+alphabet[i//20]+alphabet[i%20]+'ACDEFG' for i in range(40)]
    pd.DataFrame({'aa_seq':sequences,'label':[i%2 for i in range(40)]}).to_csv(scoped/'input.csv',index=False)
    pd.DataFrame({'aa_seq':['WWWWA','WWWWC'],'label':[0,1]}).to_csv(scoped/'validation.csv',index=False)
    result=json.loads(process_csv_and_generate_config(str(scoped/'input.csv'),str(scoped/'validation.csv'),user_overrides={'num_epochs':1,'batch_size':2}))
    assert result['success'],result
    config=json.loads(Path(result['config_path']).read_text())
    assert Path(config['output_dir']).is_relative_to(scoped)
    data=_load_dataset_dict(SimpleNamespace(**config,dataset=config['dataset_custom']))
    assert data['validation']['aa_seq']==['WWWWA','WWWWC']
    assert len(data['train'])==36 and len(data['test'])==4
    assert set(data['train']['aa_seq']).isdisjoint(data['test']['aa_seq'])


def test_training_tool_enabled_but_admin_switch_still_wins(monkeypatch):
    from mcp_tools.tools import runtime_tool_policy as policy
    from mcp_tools.tools.capability_policy import CapabilityStore
    monkeypatch.setattr(policy,'is_online_mode',lambda:True)
    monkeypatch.setattr(CapabilityStore,'tool_enabled',lambda self,name:True)
    assert all(policy.is_tool_allowed(n) for n in ['generate_training_config','train_protein_model','protein_model_predict'])
    assert not policy.is_tool_allowed('query_foldseek_search_by_pdb_file')
    monkeypatch.setattr(CapabilityStore,'tool_enabled',lambda self,name:False)
    assert not policy.is_tool_allowed('train_protein_model')


def test_regression_holdouts_do_not_treat_each_measurement_as_a_class(scoped):
    import pandas as pd
    pd.DataFrame({'sequence':['ACDE'+str(i) for i in range(30)],'value':[i/7 for i in range(30)]}).to_csv(scoped/'values.csv',index=False)
    output=Path(training.stage_dataset(str(scoped/'values.csv'),None,None,'sequence','value','regression'))
    assert [len(pd.read_csv(output/(s+'.csv'))) for s in ['train','validation','test']]==[24,3,3]


def test_live_phases_and_logs_persist_without_exposing_secrets(tmp_path):
    from agent.harness.research_timeline import ResearchTimeline
    folder=tmp_path/'episode';folder.mkdir()
    timeline=ResearchTimeline(folder,2,tmp_path/'workspace')
    timeline.phase('PI');timeline.phase('CB');timeline.phase('MLS')
    timeline.tool({'tool':'train_protein_model','status':'running','progress':{'stdout_tail':'epoch 1/2 /home/other/private api_key=abcdefghijklmnopqrstuvwxyz'}})
    timeline.tool({'tool':'train_protein_model','status':'completed'})
    timeline.phase('SC');timeline.finish('needs_revision')
    rows=json.loads((folder/'stages.json').read_text())
    assert [s['role'] for s in rows]==['PI','CB','MLS','SC']
    assert all(s['status']=='completed' and s['ended_at']>=s['started_at'] for s in rows)
    current=json.loads((folder/'current_tool.json').read_text())
    assert current['progress']['stdout_tail'].startswith('epoch 1/2')
    assert '/home/other' not in json.dumps(current) and 'abcdefghijklmnopqrstuvwxyz' not in json.dumps(current)


def test_readme_starts_from_user_goal_and_preserves_authored_document(tmp_path):
    from agent.harness.research_timeline import ensure_project_readme
    plan={'steps':[{'title':'Evaluate the supplied data'}]}
    ensure_project_readme(tmp_path,'Train a solubility model',plan)
    assert not (tmp_path/'README.md').exists()  # No fabricated inputs or outputs.
    (tmp_path/'data.csv').write_text('sequence,label\nACDE,1\n')
    ensure_project_readme(tmp_path,'Train a solubility model',plan)
    assert 'Train a solubility model' in (tmp_path/'README.md').read_text()
    (tmp_path/'README.md').write_text('Measured results and reproduction commands')
    ensure_project_readme(tmp_path,'Changed objective',plan)
    assert (tmp_path/'README.md').read_text()=='Measured results and reproduction commands'
