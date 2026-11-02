import asyncio
import copy
import json
from dataclasses import replace

import pytest

from backend.web_v2.agent_context import (
    ContextLimit, ContextPolicy, SETTINGS, cancellable, check_admission,
    estimate, prepare_context, setting_value, turn_count,
)


def transcript(n, long=False, langgraph=False):
    rows=[]
    for i in range(n):
        content=f'Protein P{i:05d}; goal: solubility; source /workspace/data{i}.csv'
        answer='Result: 12.5 mg/mL; predicted, not measured.' + (' 蛋白质'*1000 if long else '')
        if langgraph:
            rows += [{'type':'human','data':{'content':content}},
                     {'type':'ai','data':{'content':'','tool_calls':[{'id':f'call{i}','name':'measure','args':{}}]}},
                     {'type':'tool','data':{'content':answer,'tool_call_id':f'call{i}'}},
                     {'type':'ai','data':{'content':answer}}]
        else:
            rows += [{'role':'user','content':content},{'role':'assistant','content':answer}]
    return rows


def prepare(tmp_path, rows, *, policy=None, memory='', current='Continue', summary=None):
    calls=[]
    async def summarizer(source, limit):
        calls.append((source, limit))
        return summary or 'Solubility goal, P00000; predicted 12.5 mg/mL. Source: /workspace/data0.csv. Next: validate.'
    result=asyncio.run(prepare_context(rows,memory,current,policy or ContextPolicy(),131072,4096,tmp_path,summarizer))
    return result,calls


def test_short_chat_does_not_call_model_or_mutate_history(tmp_path):
    rows=transcript(2);before=copy.deepcopy(rows)
    (kept,memory,info),calls=prepare(tmp_path,rows)
    assert not calls and kept==rows==before and not memory and not info['compacted']


@pytest.mark.parametrize('langgraph',[False,True])
def test_old_turns_compact_as_complete_groups_and_archive_exact_source(tmp_path,langgraph):
    rows=transcript(12,langgraph=langgraph);before=copy.deepcopy(rows)
    (kept,memory,info),calls=prepare(tmp_path,rows,memory='Earlier user constraint: only free APIs.')
    assert len(calls)==1 and info['compacted'] and info['retained_turns']==8
    assert kept==rows[-(32 if langgraph else 16):] and rows==before
    archive=json.loads(next((tmp_path/'.venus-context').glob('*.json')).read_text())
    assert archive['messages']==rows[:(16 if langgraph else 8)]
    assert archive['memory']=='Earlier user constraint: only free APIs.'
    assert 'only free APIs' in calls[0][0] and '12.5 mg/mL' in memory
    assert '/workspace/.venus-context/' in memory
    assert info['estimated_tokens']<=info['budget']


def test_token_pressure_compresses_even_below_turn_threshold(tmp_path):
    rows=transcript(2,long=True)
    (kept,memory,info),calls=prepare(tmp_path,rows,policy=ContextPolicy(context_tokens=4096))
    assert calls and info['compacted'] and info['retained_turns']<2
    assert info['estimated_tokens']<=4096 and estimate(calls[0][0])<4096


def test_summary_carries_forward_and_only_new_old_turns_are_resummarized(tmp_path):
    policy=ContextPolicy(recent_turns=1)
    (kept,memory,_),_=prepare(tmp_path,transcript(3),policy=policy)
    rows=kept+transcript(2)
    (kept2,memory2,_),calls=prepare(tmp_path,rows,memory=memory,policy=policy)
    assert '12.5 mg/mL' in calls[0][0] and '/workspace/.venus-context/' in calls[0][0]
    assert kept2==rows[-2:] and len(list((tmp_path/'.venus-context').glob('*.json')))==2


def test_failure_leaves_working_history_intact(tmp_path):
    rows=transcript(10);before=copy.deepcopy(rows)
    async def fail(*args): raise RuntimeError('provider unavailable')
    with pytest.raises(RuntimeError):
        asyncio.run(prepare_context(rows,'saved memory','go',ContextPolicy(),131072,4096,tmp_path,fail))
    assert rows==before


def test_overlong_summary_is_bounded_and_current_request_is_not_truncated(tmp_path):
    current='Current critical constraint: no wet-lab claims.'
    (_,memory,info),_=prepare(tmp_path,transcript(10),current=current,summary='summary摘要'*30000)
    assert estimate(memory)<2200 and info['estimated_tokens']<=info['budget']
    assert current=='Current critical constraint: no wet-lab claims.'


def test_model_capacity_takes_precedence():
    assert ContextPolicy(context_tokens=100000).budget(16000,4096)<16000-4096


def test_expanded_attachment_prompt_is_also_checked(tmp_path):
    with pytest.raises(ContextLimit,match='too long'):
        prepare(tmp_path,[],current='Attachment metadata: '+'x'*33000)


def test_smaller_new_budget_recompacts_existing_memory(tmp_path):
    (_,memory,info),calls=prepare(tmp_path,[],memory='Historical memory '*1000,policy=ContextPolicy(context_tokens=4096))
    assert calls and info['compacted'] and info['estimated_tokens']<=4096


@pytest.mark.parametrize('value',['bad','1.2','-1','10001'])
def test_invalid_turn_setting(value):
    with pytest.raises(ValueError):setting_value('WEBUI_V2_AGENT_MAX_TURNS',value)


def test_limits_and_defaults(monkeypatch):
    for key in SETTINGS:monkeypatch.delenv(key,raising=False)
    assert ContextPolicy.load()==ContextPolicy()
    monkeypatch.setenv('WEBUI_V2_AGENT_MAX_TURNS','12')
    assert ContextPolicy.load().max_turns==12
    assert setting_value('WEBUI_V2_AGENT_MAX_TURNS','')==0


def test_hard_turn_limit_is_separate_from_context_and_retry():
    state={'agent_turn_count':3}
    policy=ContextPolicy(max_turns=3)
    with pytest.raises(ContextLimit,match='3-turn'):check_admission(state,'hi',policy,131072,4096)
    check_admission(state,'hi',policy,131072,4096,retry=True)
    check_admission(state,'hi',ContextPolicy(),131072,4096)
    assert turn_count({'history':[{'role':'user'},{'role':'assistant'}]})==1


def test_oversized_current_message_rejected_before_any_model_call():
    with pytest.raises(ContextLimit,match='upload'):
        check_admission({},'x'*40000,ContextPolicy(),131072,4096)


def test_cancellation_stops_pending_summary():
    async def scenario():
        finished=asyncio.Event()
        async def slow():
            try:await asyncio.sleep(100)
            finally:finished.set()
        async def cancelled():return True
        with pytest.raises(asyncio.CancelledError):await cancellable(slow(),cancelled)
        # The coroutine can be cancelled before starting, and no task survives.
        assert len([t for t in asyncio.all_tasks() if t is not asyncio.current_task()])==0
    asyncio.run(scenario())


def test_archive_symlink_cannot_escape_workspace(tmp_path):
    outside=tmp_path/'outside';outside.mkdir()
    work=tmp_path/'work';work.mkdir()
    (work/'.venus-context').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='archive directory'):prepare(work,transcript(12))
    assert not list(outside.iterdir())
