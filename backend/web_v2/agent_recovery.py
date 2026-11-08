"""Conversation recovery from evidenced native SDK retries.

The strict request audit is retained unchanged for scientific evaluation. This
check only decides whether a completed conversational answer can be retained.
It neither makes a request nor executes/replays a tool.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_BUFFERED_TOOLS = {
    "web_live_text_validated_tools_v1",
    "buffered_complete_response_v1",
    "messages_to_free_chat_buffered_v1",
}
_TRANSIENT = {"ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout",
              "RemoteProtocolError", "PoolTimeout"}
_FORBIDDEN = {"policy_rejected", "budget_rejected", "request_deadline_exhausted",
              "upstream_rejection", "retry_scheduled", "response_shape"}


def completed_turn_recovery(path, max_calls):
    result = {"policy":"native_identical_retry_v1", "accepted":False,
              "recovered_calls":[], "physical_calls":0}
    try:
        rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        requests=[row for row in rows if row.get("event")=="request"]
        responses=[row for row in rows if row.get("event")=="response"]
        errors=[row for row in rows if row.get("event")=="transport_error"]
        result["physical_calls"]=len(requests)
        if (not requests or not errors or len(requests)>max_calls or
            any(row.get("event") in _FORBIDDEN or "recovery_policy" in row for row in rows)):
            return result
        ids=[row["call"] for row in requests]
        if (any(type(value) is not int for value in ids) or
            ids!=list(range(1,len(ids)+1)) or sorted(ids)!=sorted(row["call"] for row in responses)):
            return result
        replies={row["call"]:row for row in responses}
        recovered_errors=set()
        index=0
        while index<len(requests):
            first=requests[index]
            reply=replies[first["call"]]
            if rows.index(first)>=rows.index(reply):
                return result
            if reply.get("status")==200 and reply.get("completed") is True and not reply.get("error_type"):
                index+=1
                continue
            if index+1>=len(requests):
                return result
            second=requests[index+1]
            success=replies[second["call"]]
            problem=[row for row in errors if row.get("call")==first["call"]]
            fingerprint=first.get("request_sha256","")
            if not (
                first.get("transport") in _BUFFERED_TOOLS
                and second.get("transport")==first.get("transport")
                and re.fullmatch(r"[0-9a-f]{64}",fingerprint)
                and second.get("request_sha256")==fingerprint
                and all(first.get(key)==second.get(key) for key in
                        ("model","request_bytes","tool_schema_sha256","stream"))
                and reply.get("completed") is False
                and reply.get("status") in (None,200)
                and reply.get("error_type") in _TRANSIENT
                and len(problem)==1
                and problem[0].get("error_type")==reply["error_type"]
                and rows.index(first)<rows.index(problem[0])<rows.index(reply)<rows.index(second)<rows.index(success)
                and success.get("status")==200 and success.get("completed") is True
                and not success.get("error_type")
            ):
                return result
            recovered_errors.add(first["call"])
            result["recovered_calls"].append({"failed_call":first["call"],"successful_call":second["call"],
                                               "error_type":reply["error_type"]})
            index+=2
        if len(errors)!=len(recovered_errors) or {row.get("call") for row in errors}!=recovered_errors:
            return result
        result["accepted"]=bool(result["recovered_calls"])
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return result


def canonical_recovered_history(history, turn_id, text):
    """Replace speculative preview text only in the recovered, completed turn."""
    if not text:
        return history
    result=[];written=False
    for row in history:
        if row.get("turn_id")==turn_id and row.get("role")=="assistant" and row.get("kind","text")=="text":
            if not written:
                result.append({**row,"content":text});written=True
        else:
            result.append(row)
    if not written:
        result.append({"role":"assistant","kind":"text","turn_id":turn_id,"content":text})
    return result
