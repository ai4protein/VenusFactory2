import json
from pathlib import Path
import pytest

from mcp_tools.tools.tool_surface import RETIRED_CODE_TOOLS, WORKSPACE_ALIASES
from mcp_tools.tools.capability_policy import CapabilityStore
from agent.harness.session_boundary import session_tools
from agent.harness.research_workspace import ResearchWorkspace


@pytest.fixture(autouse=True)
def private_runtime(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_RUNTIME_ROOT',str(tmp_path/'runtime'))
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('WEBUI_V2_MODE','local')
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode',lambda:False)


def test_no_retired_code_tools_in_any_global_catalog():
    from mcp_tools.tools.tools_agent_hub import ALL_TOOLS,MCP_TOOLS,FASTAPI_TOOLS,TRAIN_TOOLS
    from mcp_tools.models.factory.training.tools_agent import TRAIN_TOOLS as train
    from mcp_tools.tools.skill.tools_agent import get_python_repl_tool
    for tools in [ALL_TOOLS,MCP_TOOLS,FASTAPI_TOOLS,TRAIN_TOOLS,train]:
        assert not RETIRED_CODE_TOOLS.intersection(t.name for t in tools)
    assert get_python_repl_tool() is None


def test_legacy_entry_is_a_migration_receipt_not_a_second_llm(monkeypatch):
    from mcp_tools.models.factory.training.tools_agent import agent_generated_code_tool
    monkeypatch.setattr('mcp_tools.models.factory.training.train_operations.generate_and_execute_code',lambda *a,**k:pytest.fail('retired generation must not run'))
    result=json.loads(agent_generated_code_tool.invoke({'task_description':'write and run code'}))
    assert not result['success'] and result['code']=='tool_retired'
    assert 'write_session_file' in result['error'] and 'run_session_python' in result['error']


def test_agent_scientist_share_one_workspace_contract(tmp_path):
    agent={t.name:t for t in session_tools(tmp_path/'a',tmp_path/'ac')}
    science={t.name:t for t in ResearchWorkspace(tmp_path/'s',tmp_path/'sc',{'steps':[]}).tools()}
    for name in WORKSPACE_ALIASES.values():
        assert name in agent and name in science
        a=agent[name].get_input_schema().model_json_schema();s=science[name].get_input_schema().model_json_schema()
        assert a['properties']==s['properties'] and a.get('required')==s.get('required')
        assert not {'session_id','workspace','mounts','control','backend'} & a['properties'].keys()
    assert not set(WORKSPACE_ALIASES) & set(science)
    assert 'run_session_shell' in agent and 'run_session_shell' not in science
    assert {'evaluate_research_model','get_research_review_requirements','propose_research_capability','submit_research_checkpoint'}<=set(science)


@pytest.mark.parametrize('old,new',list(WORKSPACE_ALIASES.items()))
def test_legacy_admin_switches_cannot_be_bypassed(tmp_path,old,new):
    store=CapabilityStore()
    with store.connect() as c:
        c.execute('INSERT INTO switches VALUES(?,?,?,?,?,?)',('tool:'+old,0,2,'{}',1,'legacy-admin'))
    assert not store.tool_enabled(new) and not store.tool_enabled(old)
    assert store.rows()['tool:'+new]['revision']==2
    from mcp_tools.tools.capability_policy import CapabilityDisabled
    with pytest.raises(CapabilityDisabled):store.assert_call(new)
    revision=store.set('tool:'+new,True,2,{},'admin')
    assert revision==3 and store.tool_enabled(old) and store.tool_enabled(new)
    with store.connect() as c:assert not c.execute('SELECT 1 FROM switches WHERE key=?',('tool:'+old,)).fetchone()


def test_conflicting_switches_keep_denial_until_explicit_admin_update():
    store=CapabilityStore()
    with store.connect() as c:
        c.executemany('INSERT INTO switches VALUES(?,?,?,?,?,?)',[
            ('tool:run_research_python',0,3,'{}',1,'legacy'),
            ('tool:run_session_python',1,2,'{}',2,'admin')])
    assert not store.tool_enabled('run_session_python')
    with pytest.raises(ValueError):store.set('tool:run_session_python',True,2,{},'admin')
    store.set('tool:run_session_python',True,3,{},'admin')
    assert store.tool_enabled('run_session_python')


def test_current_role_prompts_and_skill_catalog_do_not_offer_retired_code_tools():
    root=Path(__file__).resolve().parents[2]/'src/agent'
    for name in ['principal_investigator','computational_biologist','computational_biologist_step_planning','machine_learning_specialist','machine_learning_specialist_self_check']:
        text=(root/'prompts'/f'{name}.md').read_text()
        assert not any(retired in text for retired in RETIRED_CODE_TOOLS)
    for path in (root/'skills').rglob('SKILL.md'):
        assert not any(retired in path.read_text() for retired in RETIRED_CODE_TOOLS),path
