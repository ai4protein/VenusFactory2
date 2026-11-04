import json
from agent.harness.provider_usage import reported_usage
from backend.web_v2.usage_store import normalize_usage


def sse(*events):
    return b"".join(b"data: "+json.dumps(e).encode()+b"\n\n" for e in events)


def test_partial_and_repeated_snapshots_are_not_added():
    raw=sse({'usage':{'prompt_tokens':10,'total_tokens':10}},
            {'usage':{'completion_tokens':2}},
            {'usage':{'completion_tokens':2,'reasoning_tokens':1,'cached_tokens':8}})
    assert normalize_usage(reported_usage(raw,True))==(10,2,12)


def test_counts_do_not_come_from_text_and_invalid_tail_does_not_erase_usage():
    assert reported_usage(sse({'choices':[{'delta':{'content':'total_tokens: 1000'}}]}),True) is None
    raw=sse({'usage':{'input_tokens':9,'output_tokens':1,'total_tokens':10}})+b'data: {broken'
    assert normalize_usage(reported_usage(raw,True))==(9,1,10)


def test_missing_is_distinct_from_reported_zero():
    assert reported_usage(b'{}',False) is None
    assert normalize_usage(reported_usage(b'{"usage":{"input_tokens":0,"output_tokens":0}}',False))==(0,0,0)
    assert reported_usage(b'{"usage":{"input_tokens":true,"output_tokens":-2}}',False) is None
