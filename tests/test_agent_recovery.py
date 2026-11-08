import copy
import json
import pytest

from backend.web_v2.agent_recovery import completed_turn_recovery, canonical_recovered_history
from agent.harness.events import audit_requests


def records():
    first={"event":"request","call":1,"model":"fixture","request_sha256":"a"*64,
           "request_bytes":100,"tool_schema_sha256":"b"*64,"stream":True,
           "transport":"web_live_text_validated_tools_v1"}
    return [first,
            {"event":"transport_error","call":1,"error_type":"RemoteProtocolError"},
            {"event":"response","call":1,"status":200,"completed":False,
             "error_type":"RemoteProtocolError","usage":{"total_tokens":10}},
            {**first,"call":2},
            {"event":"response","call":2,"status":200,"completed":True,
             "error_type":None,"usage":{"total_tokens":12}}]


def save(tmp_path, rows):
    path=tmp_path/"model_requests.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
    return path


def test_exact_completed_sdk_retry_keeps_raw_failure_and_both_usage_records(tmp_path):
    path=save(tmp_path,records());before=path.read_bytes()
    strict=audit_requests(path)
    result=completed_turn_recovery(path,2)
    assert result["accepted"] and result["physical_calls"]==2
    assert result["recovered_calls"]==[{"failed_call":1,"successful_call":2,"error_type":"RemoteProtocolError"}]
    assert not strict["passed"] and strict["model_calls"]==2 and len(strict["usage"])==2
    assert path.read_bytes()==before


@pytest.mark.parametrize("fault", [
    "hash","model","schema","budget","request_missing","duplicate_response","order",
    "cancelled","parse_failure","http_error","second_failed","unmatched_error",
    "policy","changed_transport","unbuffered","two_failures","recovery_policy",
    "missing_hash","error_mismatch","successful_error","gap"])
def test_partial_or_unrelated_failures_cannot_be_relabelled(tmp_path,fault):
    rows=records();budget=3
    if fault=="hash": rows[3]["request_sha256"]="c"*64
    elif fault=="model": rows[3]["model"]="another"
    elif fault=="schema": rows[3]["tool_schema_sha256"]="c"*64
    elif fault=="budget": budget=1
    elif fault=="request_missing": rows.pop(0)
    elif fault=="duplicate_response": rows.append(copy.deepcopy(rows[-1]))
    elif fault=="order": rows[2],rows[3]=rows[3],rows[2]
    elif fault in {"cancelled","parse_failure"}:
        rows[1]["error_type"]=rows[2]["error_type"]="CancelledError" if fault=="cancelled" else "ValueError"
    elif fault=="http_error": rows[2]["status"]=403
    elif fault=="second_failed": rows[-1]["completed"]=False
    elif fault=="unmatched_error": rows.append({"event":"transport_error","call":2,"error_type":"ReadError"})
    elif fault=="policy": rows.append({"event":"policy_rejected"})
    elif fault=="changed_transport": rows[3]["transport"]="messages_to_free_chat_buffered_v1"
    elif fault=="unbuffered": rows[0]["transport"]=rows[3]["transport"]="direct_unchecked_stream"
    elif fault=="two_failures":
        rows[-1]["completed"]=False
        rows.extend([{**rows[0],"call":3},{"event":"response","call":3,"status":200,"completed":True}])
    elif fault=="recovery_policy": rows[0]["recovery_policy"]="unknown"
    elif fault=="missing_hash": rows[0].pop("request_sha256")
    elif fault=="error_mismatch": rows[1]["error_type"]="ReadTimeout"
    elif fault=="successful_error": rows[-1]["error_type"]="ReadError"
    elif fault=="gap": rows[3]["call"]=rows[-1]["call"]=3
    assert not completed_turn_recovery(save(tmp_path,rows),budget)["accepted"]


def test_normal_success_does_not_invent_a_recovery(tmp_path):
    rows=records()
    assert not completed_turn_recovery(save(tmp_path,[rows[0],{**rows[-1],"call":1}]),2)["accepted"]


def test_recovered_preview_replacement_preserves_other_turns_and_tools():
    history=[{"role":"assistant","kind":"text","turn_id":"old","content":"old reply"},
             {"role":"user","turn_id":"new","content":"original request"},
             {"role":"assistant","kind":"text","turn_id":"new","content":"partial failed preview"},
             {"role":"assistant","kind":"thinking","turn_id":"new","content":"status"},
             {"role":"assistant","kind":"text","turn_id":"new","content":"duplicate successful preview"}]
    original=copy.deepcopy(history)
    fixed=canonical_recovered_history(history,"new","verified final reply")
    assert history==original and fixed[0]==history[0]
    assert [x["content"] for x in fixed if x.get("turn_id")=="new" and x.get("kind")=="text"]==["verified final reply"]
    assert fixed[-1]["content"]=="status"
