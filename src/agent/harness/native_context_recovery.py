"""Evidence-bound audit of native overflow recovery; never retries or summarizes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .context_limit_error import is_context_limit
from .contracts import digest

POLICY = "native_context_recovery.v1"


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _rows(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("invalid_evidence_path")
    path = root / name
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("symlinked_evidence")
    path.resolve().relative_to(root.resolve())
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def native_context_for_run(output, result, harness, tools, tool_limit, calls_after):
    """Bind a host result to its native phase; expert roles cannot share recovery."""
    root = Path(output)
    before = 0
    if harness.startswith("expert-"):
        history = json.loads((root / "role_history.json").read_text())
        phases = [row for row in history if row.get("role") == "MLS"]
        if len(phases) != 1 or phases[0].get("status") != "completed":
            raise ValueError("missing_completed_native_phase")
        phase = phases[0]
        root = root / "roles/MLS"
        native = json.loads((root / "native_result.json").read_text())
        if digest(native) != phase.get("native_result_sha256") or native.get("session_id") != phase.get("native_session_id"):
            raise ValueError("native_phase_binding_changed")
        result = native
        before, calls_after = phase["calls_before"], phase["calls_after"]
    return {"root": root, "result": result, "calls_before": before, "calls_after": calls_after,
            "tools": set(tools), "tool_limit": tool_limit}


def verify_native_continuation(native):
    """Verify raw-bound boundary/tool/terminal witnesses; return no inferred success.

    The observation-time gateway high-water marks deliberately err conservatively
    if native stdout arrives late. A notice is not a completed compaction.
    """
    from .events import audit_trace

    root, result = Path(native["root"]), native["result"]
    sid = result.get("session_id")
    if result.get("status") != "completed" or result.get("sandbox_stopped") is not True or not isinstance(sid, str) or not sid:
        raise ValueError("native_terminal_not_successful")
    if not audit_trace(root / "events.jsonl", native["tools"], native["tool_limit"])["passed"]:
        raise ValueError("invalid_native_tool_ledger")
    rows = _rows(root, "events.jsonl")
    selected = [row for row in rows if row.get("kind") in
                {"context_compaction", "tool_call_start", "tool_result", "native_terminal"}]
    if not selected or any(row.get("session_id") != sid for row in selected):
        raise ValueError("native_session_mismatch")
    calls = [row.get("gateway_calls") for row in selected]
    if any(type(call) is not int or call < 0 for call in calls) or calls != sorted(calls):
        raise ValueError("invalid_gateway_call_anchors")
    if type(native["calls_before"]) is not int or type(native["calls_after"]) is not int:
        raise ValueError("invalid_native_call_interval")
    if not all(native["calls_before"] <= call <= native["calls_after"] for call in calls):
        raise ValueError("native_call_anchor_outside_phase")
    cache = {}
    def indexed(name):
        if name not in cache:
            entries = _rows(root, name)
            hashes = {}
            for index, event in enumerate(entries):
                hashes.setdefault(_hash(event), []).append((index, event))
            cache[name] = (entries, hashes)
        return cache[name]

    checked = []
    for row in selected:
        name = row.get("native_event_path")
        raw_rows, hashes = indexed(name)
        matches = hashes.get(row.get("native_event_sha256"), [])
        if len(matches) != 1:
            raise ValueError("native_event_hash_missing_or_ambiguous")
        index, event = matches[0]
        if event.get("type") == "thread.started" or any(e.get("type") == "thread.started" for e in raw_rows):
            sessions = [e.get("thread_id") for e in raw_rows if e.get("type") == "thread.started"]
            if sessions != [sid]:
                raise ValueError("raw_native_session_mismatch")
        elif "session_id" in event:
            if event["session_id"] != sid:
                raise ValueError("raw_native_session_mismatch")
        else:
            _, notification_hashes = indexed(row.get("native_notification_path"))
            found = notification_hashes.get(row.get("native_notification_sha256"), [])
            if len(found) != 1 or found[0][1].get("sessionId") != sid or found[0][1].get("event") != event:
                raise ValueError("native_notification_binding_changed")
        _validate_raw(row, event)
        checked.append((row, name, index))
    terminals = [item for item in checked if item[0]["kind"] == "native_terminal"]
    if len(terminals) != 1 or checked[-1] != terminals[0]:
        raise ValueError("missing_unique_final_native_terminal")
    terminal = terminals[0]
    if terminal[0]["gateway_calls"] != native["calls_after"]:
        raise ValueError("native_terminal_call_interval_mismatch")
    boundaries = [item for item in checked if item[0]["kind"] == "context_compaction"
                  and item[0].get("phase") in {"notice", "completed"}]
    witnesses = []
    for boundary in boundaries:
        b, name, index = boundary
        for start in checked:
            row, source, raw_index = start
            if row["kind"] != "tool_call_start" or source != name or raw_index <= index:
                continue
            ends = [item for item in checked if item[0]["kind"] == "tool_result"
                    and item[0].get("tool_call_id") == row.get("tool_call_id") and item[0].get("is_error") is False]
            if len(ends) != 1:
                continue
            end = ends[0]
            if end[1] != name or terminal[1] != name or not raw_index < end[2] < terminal[2]:
                continue
            lower = b["gateway_calls"]
            delayed = {}
            if lower == row["gateway_calls"] and b.get("native_type") == "compaction/end":
                # Native stdout may arrive after its next HTTP request starts.
                # An earlier, raw-bound summary from THIS compaction supplies a
                # conservative range; equality alone never proves continuation.
                cid = b.get("compaction_id")
                same_cycle = [item for item in checked if item[1] == name
                              and item[0].get("kind") == "context_compaction"
                              and item[0].get("compaction_id") == cid]
                starts = [item for item in same_cycle if item[0].get("phase") == "started"]
                summaries = [item for item in same_cycle if item[0].get("phase") == "summary"]
                if not isinstance(cid, str) or not cid or len(starts) != 1 or len(summaries) != 1:
                    continue
                begin, summary = starts[0], summaries[0]
                if not begin[2] < summary[2] < index:
                    continue
                start_call, summary_call = begin[0]["gateway_calls"], summary[0]["gateway_calls"]
                if not start_call < summary_call < row["gateway_calls"]:
                    continue
                lower = summary_call
                delayed = {"compaction_start_call": start_call, "summary_call": summary_call,
                           "summary_sha256": summary[0]["native_event_sha256"],
                           "boundary_observed_call": b["gateway_calls"], "compaction_id": cid}
            if not lower < row["gateway_calls"] <= end[0]["gateway_calls"] <= terminal[0]["gateway_calls"]:
                continue
            witnesses.append({"session_id": sid, "boundary_call": lower,
                "continuation_call": row["gateway_calls"], "terminal_call": terminal[0]["gateway_calls"],
                "boundary_phase": b["phase"], "boundary_sha256": b["native_event_sha256"],
                "tool_call_id": row["tool_call_id"], "terminal_sha256": terminal[0]["native_event_sha256"],
                **delayed})
    if not witnesses:
        raise ValueError("missing_post_boundary_successful_tool_continuation")
    return witnesses


def _validate_raw(row, event):
    kind, typ = row["kind"], event.get("type")
    item, data = event.get("item", {}), event.get("data", {})
    if kind == "context_compaction":
        phase = row.get("phase")
        if phase == "notice":
            from .codex_adapter import COMPACTION_NOTICE
            valid = typ == "item.completed" and item.get("type") == "error" and item.get("message") == COMPACTION_NOTICE
        elif phase == "completed":
            valid = (typ == "compaction/end" and not data.get("error")) or (
                typ == "system" and event.get("subtype") == "compact_boundary"
                and event.get("compact_metadata", {}).get("trigger") in {"auto", "manual"})
        else:
            # These records do not themselves authorize recovery. Their exact
            # phase and cycle identity may bound a delayed completion witness.
            expected = {"started": "compaction/start", "summary": "compaction/summary",
                        "pruned": "compaction/prune", "failed": "compaction/end"}
            valid = typ == expected.get(phase) or (
                phase == "started" and typ == "system" and event.get("subtype") == "status"
                and event.get("status") == "compacting")
            if phase == "failed":
                valid = valid and bool(data.get("error"))
        if typ.startswith("compaction/"):
            valid = valid and row.get("native_type") == typ and row.get("compaction_id") == data.get("compactionId")
    elif kind == "native_terminal":
        valid = row.get("phase") == "completed" and (typ == "turn.completed" or
            (typ == "result" and event.get("subtype") == "success" and event.get("is_error") is False
             and not event.get("permission_denials") and not event.get("safety_stops")) or
            (typ == "turn/end" and data.get("reason", {}).get("kind") == "completed"))
    elif typ in {"item.started", "item.completed"}:
        valid = (item.get("type") == "mcp_tool_call" and item.get("id") == row.get("tool_call_id")
                 and item.get("server") == "venusfactory")
        if kind == "tool_call_start":
            valid = valid and typ == "item.started" and item.get("status") == "in_progress" and (
                row.get("tool_name") == "mcp__venusfactory__" + str(item.get("tool")) and row.get("args") == item.get("arguments"))
        else:
            result = item.get("result")
            actual_error = item.get("status") != "completed" or item.get("error") is not None or not isinstance(result, dict) or bool(result.get("isError", result.get("is_error", False)))
            valid = valid and typ == "item.completed" and row.get("is_error") is actual_error
    elif typ in {"assistant", "user"}:
        content = event.get("message", {}).get("content", [])
        if kind == "tool_call_start":
            valid = typ == "assistant" and any(part.get("type") == "tool_use" and
                part.get("id") == row.get("tool_call_id") and part.get("name") == row.get("tool_name")
                and part.get("input") == row.get("args") for part in content)
        else:
            valid = typ == "user" and any(part.get("type") == "tool_result" and
                part.get("tool_use_id") == row.get("tool_call_id") and
                part.get("is_error", False) is row.get("is_error") for part in content)
    elif kind == "tool_call_start":
        valid = typ == "tool/call" and data.get("callId") == row.get("tool_call_id") and data.get("name") == row.get("tool_name")
    else:
        valid = typ == "tool/result" and any(part.get("type") == "tool-result" and
            part.get("toolCallId") == row.get("tool_call_id") and part.get("isError", False) is row.get("is_error")
            for part in data.get("message", {}).get("content", []))
    if not valid:
        raise ValueError("raw_native_event_semantics_mismatch")


def audit_native_context_recovery(native, records, requests, responses):
    if native is None:
        return None
    rejected = {row["call"]: row for row in records if row.get("event") == "upstream_rejection"
                and type(row.get("call")) is int and is_context_limit(row, row.get("status"))}
    failed = [row for row in responses if row.get("call") in rejected and
              row.get("status") == rejected[row["call"]]["status"]]
    if not failed:
        return None
    result = {"policy": POLICY, "valid": False, "recovered_calls": [], "problems": []}
    if isinstance(native, list):
        try:
            intervals = sorted((context["calls_before"], context["calls_after"], index)
                               for index, context in enumerate(native))
            if any(type(lo) is not int or type(hi) is not int or lo < 0 or hi <= lo for lo, hi, _ in intervals):
                raise ValueError("invalid_native_call_interval")
            if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
                raise ValueError("overlapping_native_call_intervals")
            for failure in failed:
                scopes = [context for context in native if context["calls_before"] < failure["call"] <= context["calls_after"]]
                if len(scopes) != 1:
                    raise ValueError("unbound_failed_native_call")
            for lo, hi, index in intervals:
                subset = [row for row in responses if lo < row["call"] <= hi]
                review = audit_native_context_recovery(native[index], records, requests, subset)
                if review is not None:
                    if not review["valid"]:
                        result["problems"].extend(review["problems"])
                    result["recovered_calls"].extend(review["recovered_calls"])
            result["valid"] = not result["problems"]
        except (ValueError, KeyError, TypeError) as exc:
            result["problems"].append(str(exc) if type(exc) is ValueError else type(exc).__name__)
        if not result["valid"]:
            result["recovered_calls"] = []
        return result
    try:
        witnesses = verify_native_continuation(native)
        req = {row["call"]: row for row in requests}
        for failure in failed:
            call = failure["call"]
            if (failure.get("completed") is not False or failure.get("error_type") is not None
                    or not native["calls_before"] < call <= native["calls_after"]):
                raise ValueError("failed_call_outside_native_phase_or_relabelled")
            if any(row.get("event") == "retry_scheduled" and row.get("failed_call") == call for row in records):
                raise ValueError("overlapping_recovery_claims")
            if req.get(call, {}).get("retry_of") is not None:
                raise ValueError("overlapping_recovery_claims")
            matches = []
            for witness in witnesses:
                if call > witness.get("compaction_start_call", witness["boundary_call"]):
                    continue
                if "summary_call" in witness and not any(
                    row.get("status") == 200 and row.get("completed") is True
                    and witness["compaction_start_call"] < row["call"] <= witness["summary_call"]
                    for row in responses
                ):
                    continue
                success = [row for row in responses if row.get("status") == 200 and row.get("completed") is True
                           and witness["boundary_call"] < row["call"] <= witness["continuation_call"]]
                if success and all(row["call"] in req for row in success):
                    matches.append((witness, success[-1]["call"]))
            if not matches:
                raise ValueError("missing_successful_request_after_native_boundary")
            witness, successful_call = matches[0]
            result["recovered_calls"].append({"failed_call": call, "successful_call": successful_call, **witness})
        result["valid"] = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["problems"].append(str(exc) if type(exc) is ValueError else type(exc).__name__)
        result["recovered_calls"] = []
    return result
