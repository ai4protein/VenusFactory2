import json
from pathlib import Path
import pytest
from agent.harness.free_catalog import configured_free
from agent.harness.catalog import ExecutionCatalog
from agent.harness import Profile
from agent.rsi import model_access

def setup_config(tmp_path,monkeypatch):
    binding={'provider':'custom','model':'free-science','base_url':'https://inference.example/v1','credential_env':'TEST_FREE_KEY'}
    data={'models':[{'id':'free-one','label':'Free science','binding':binding,'enabled':True,'pricing':{'input_per_million':0,'output_per_million':0},'max_output_tokens':4096,'protocols':['chat_completions']}],'defaults':{'science_agent':{'harness':'deepseek-harness','model_id':'free-one'}},'harnesses':{'science_agent':['kimi-code','deepseek-harness','codex','claude-code']}}
    path=tmp_path/'catalog.json';path.write_text(json.dumps(data));monkeypatch.setenv('VENUS_EXECUTION_CATALOG',str(path))
    policy=tmp_path/'access.json';policy.write_text(json.dumps({'schema':2,'policy_id':'fixture','catalog':'catalog.json'}));monkeypatch.setattr(model_access,'POLICY_PATH',policy)
    return path,data,binding

def test_routes_are_configurable_without_frontend_or_python_edits(tmp_path,monkeypatch):
    path,data,binding=setup_config(tmp_path,monkeypatch)
    assert configured_free('free-science',binding['base_url'],'chat_completions')
    assert model_access.authorize_model(model='free-science',base_url=binding['base_url'],protocol='chat_completions',output_tokens=100)['policy_id']=='fixture'
    with pytest.raises(model_access.ModelAccessDenied):model_access.authorize_model(model='paid',base_url=binding['base_url'],protocol='chat_completions',output_tokens=100)
    data['models'][0]['enabled']=False;path.write_text(json.dumps(data))
    assert not configured_free('free-science',binding['base_url'],'chat_completions')
    with pytest.raises(model_access.ModelAccessDenied):model_access.authorize_model(model='free-science',base_url=binding['base_url'],protocol='chat_completions',output_tokens=100)

def test_configured_harnesses_and_defaults(tmp_path,monkeypatch):
    path,data,binding=setup_config(tmp_path,monkeypatch)
    cat=ExecutionCatalog(tmp_path,Profile.from_dict({'mode':'science_agent','harness':'langgraph','model':binding}),path=path,free_only=True)
    assert cat.default_profile.harness=='deepseek-harness'
    assert cat.select()[0].harness=='deepseek-harness'
    with pytest.raises(ValueError):cat.select({'harness':'langgraph','model_id':'free-one'})
    assert cat.models['free-one'].public()['free'] is True

def test_paid_pricing_never_enables_a_model(tmp_path,monkeypatch):
    path,data,binding=setup_config(tmp_path,monkeypatch)
    data['models'][0]['pricing']['output_per_million']=0.01;path.write_text(json.dumps(data))
    assert not configured_free('free-science',binding['base_url'],'chat_completions')
