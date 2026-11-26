import copy
import json

import pytest

from agent.session_store import persistable_state
from backend.web_v2.agent_conversation import continuation_messages, record_tool_observation, completed_messages


@pytest.mark.parametrize('harness', ['deepseek-harness', 'kimi-code', 'codex', 'claude-code', 'langgraph'])
def test_unfinished_request_and_tool_evidence_survive_reload(harness):
    state = {'agent_messages': [], 'agent_pending_turn': {'text': 'Count the available tools', 'observations': []}}
    pending = state['agent_pending_turn']
    record_tool_observation(pending, {'kind':'tool_call_start','tool_call_id':'c1','tool_name':'search_research_tools','args':{'kind':'tools'}}, harness)
    record_tool_observation(pending, {'kind':'tool_result','tool_call_id':'c1','output':{'total':137,'items':[]}}, harness)
    restored = json.loads(json.dumps(persistable_state(state)))
    before=copy.deepcopy(restored)
    rows=continuation_messages(restored, harness=harness)
    assert restored==before
    text=json.dumps(rows)
    assert 'Count the available tools' in text and '137' in text and 'interrupted' in text
    assert len(rows)==3
    if harness=='langgraph': assert all('type' in row and 'data' in row for row in rows)


def test_retry_does_not_duplicate_its_own_unfinished_turn():
    state={'agent_retry_base':[{'role':'user','content':'original context'}],
           'agent_pending_turn':{'text':'retry this','observations':[]}}
    rows=continuation_messages(state,retry=True)
    assert rows==state['agent_retry_base'] and rows is not state['agent_retry_base']


def test_failed_tool_is_never_promoted_to_success_and_output_is_bounded():
    pending={}
    record_tool_observation(pending, {'kind':'tool_call_start','tool_call_id':'c1','tool_name':'predict','args':{}}, 'codex')
    record_tool_observation(pending, {'kind':'tool_result','tool_call_id':'c1','is_error':True,'output':'x'*50000}, 'codex')
    content=pending['observations'][0]['content']
    assert '"status": "failed"' in content and len(content)<12200 and 'excerpt' in content


def test_tool_measurement_is_available_on_next_completed_turn():
    observation={'role':'assistant','content':'Recorded tool observation: MW = 4500 Da; source fixture'}
    rows=completed_messages([], 'Calculate molecular weight', [observation], 'Done.', 'deepseek-harness')
    assert rows==continuation_messages({'agent_messages':rows})
    assert rows[1]==observation


def test_legacy_interruption_only_recovers_messages_after_last_committed_answer():
    rows=[{'role':'user','content':'How many tools?'},{'role':'assistant','content':'Checking inventory'}]
    state={'agent_messages':rows,'history':rows+[{'role':'user','content':'The total, please'}, {'role':'assistant','kind':'thinking','content':'thinking'}]}
    recovered=continuation_messages(state)
    assert len(recovered)==4 and recovered[2]['content']=='The total, please'
    assert not any(r['content']=='thinking' for r in recovered)


def test_completed_turn_replaces_pending_checkpoint_without_changing_originals():
    earlier=[{'role':'user','content':'Protein A'}]
    obs=[{'role':'assistant','content':'Tool observed 42'}]
    saved=completed_messages(earlier, 'Measure it', obs, '42 units', 'kimi-code')
    saved[0]['content']='changed'
    assert earlier[0]['content']=='Protein A' and len(obs)==1
