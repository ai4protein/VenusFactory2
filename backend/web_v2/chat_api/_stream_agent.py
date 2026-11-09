"""Web transport for native Agent adapters, separate from Scientist goal control."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from agent.harness import Profile
from agent.harness.contracts import digest
from agent.harness.events import audit_requests, audit_trace
from agent.harness.gateway import EvidenceGateway
from agent.harness.host import write_json
from agent.kimi_client import KimiEvent
from agent.rsi.engine.model_gateway import GatewayPolicy
from backend.web_v2.agent_execution import ROOT, admit, runtime_paths
from backend.web_v2.agent_conversation import continuation_messages, record_tool_observation, completed_messages, record_model_result
from backend.web_v2.agent_context import ContextPolicy, prepare_context, summarize_via_relay, turn_count, cancellable
from backend.web_v2.chat_api._hooks_runtime import _is_cancelled, _session_store
from backend.web_v2.chat_api._shared import _snapshot, _to_json, clear_expert_gates
from backend.web_v2.chat_api._stream_kimi import _apply_event
from backend.web_v2.chat_api._uploads import _normalize_uploaded_file
from backend.web_v2.redact import redact_obj_for_frontend

EVIDENCE_ROOT = Path(os.getenv("VENUS_AGENT_EVIDENCE_ROOT") or ROOT / "research_runs/rsi/web_agent_sessions").resolve()


async def stop_process(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), 5)
    except TimeoutError:
        pass
    finally:
        # A tool descendant can ignore TERM even after the parent exits.
        # Always reap the remaining process group, including on a normal exit.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()


async def _stream_agent(state, text, attachment_paths, *, retry=False, interaction_response=None):
    binding, credential = admit(state)
    profile = Profile.from_dict(binding["profile"])
    sid, turn_id = state["session_id"], uuid.uuid4().hex
    workspace = Path(state["agent_session_dir"]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    control = EVIDENCE_ROOT / digest(sid) / turn_id
    control.mkdir(parents=True, mode=0o700)
    valid = []
    for path in attachment_paths:
        normalized = await _normalize_uploaded_file(
            path, str(workspace), state.setdefault("temp_files", []), str(state.get("owner_key", ""))
        )
        if normalized:
            valid.append(normalized)
    if interaction_response:
        from agent.harness.interactions import InteractionResponse, response_text
        answer = InteractionResponse.model_validate(interaction_response)
        gate = state.get("pending_interaction")
        if gate:
            response_text(gate, answer)
            state.setdefault("interaction_history", []).append({"request": gate, "response": answer.model_dump(), "run_id": state.get("_agent_run_id")})
            state.pop("pending_interaction", None)
            state["waiting_for"] = ""
    display = (text or "").strip()
    if interaction_response and gate:
        from agent.harness.interactions import response_display
        display = response_display(gate, answer, state.get("user_lang", "en"))
    if valid:
        display += "\nAttached files: " + ", ".join(Path(p).name for p in valid)
    if not display.strip():
        yield "event: done\ndata: {}\n\n"
        return
    request_text = (text if interaction_response else display) + ("\nSession attachment paths:\n" + "\n".join(valid) if valid else "")
    previous = continuation_messages(state, retry=retry, harness=profile.harness)
    memory = state.get("agent_retry_memory", "") if retry else state.get("agent_memory", "")
    context_policy = ContextPolicy.load()
    if not retry and not interaction_response:
        state["agent_turn_count"] = turn_count(state) + 1
    state["agent_retry_base"] = previous
    state["agent_retry_memory"] = memory
    state["agent_pending_turn"] = {"turn_id": turn_id, "text": request_text, "observations": []}
    state.update(
        engine=profile.harness + "-agent",
        chat_mode="science_agent",
        last_user_text=text,
        last_attachment_paths=valid,
        status="started",
    )
    state.pop("error", None)
    clear_expert_gates(state)
    for key in ["kimi_pending_question", "kimi_pending_approval", "kimi_context", "approval_prompt", "plan_markdown"]:
        state.pop(key, None)
    state.setdefault("history", []).append({"role": "user", "content": display, "turn_id": turn_id})
    state.setdefault("conversation_log", []).append(
        {"role": "user", "content": display, "timestamp": datetime.now().isoformat()}
    )
    for frame in _apply_event(state, KimiEvent(kind="turn_started", turn_id=turn_id)):
        yield frame
    await _session_store.save(sid)
    policy = GatewayPolicy(
        model=profile.model.model,
        mode="science_agent",
        scientific_tools=set(),
        max_calls=profile.budget.model_calls,
        temperature=profile.budget.temperature,
        max_tokens=profile.budget.output_tokens,
        thinking=profile.budget.thinking,
    )
    gateway_type = EvidenceGateway
    if profile.model.protocol == "responses":
        from agent.harness.responses_gateway import ResponsesGateway, ResponsesPolicy

        gateway_type = ResponsesGateway
        policy = ResponsesPolicy.from_profile(profile, [])
    elif profile.harness == "codex":
        from agent.harness.responses_bridge import ResponsesChatBridge
        gateway_type = ResponsesChatBridge
    elif profile.harness == "claude-code":
        from agent.harness.messages_bridge import MessagesBridge

        gateway_type = MessagesBridge
    queue = asyncio.Queue(maxsize=128)
    preview_call = None
    reader = None
    preparing_context = True
    preview_text_seen = set()

    async def preview_text(call, text):
        if not preparing_context:
            if call not in preview_text_seen and not text.strip():
                return
            preview_text_seen.add(call)
            await queue.put(("preview", {"kind": "text", "text": text, "call": call}))

    async def read_worker(proc):
        try:
            while line := await proc.stdout.readline():
                await queue.put(("worker", json.loads(line)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await queue.put(("failure", {"error_type": type(exc).__name__}))
        else:
            await queue.put(("eof", {}))

    proc, completed, tools, outcome = None, None, [], "runtime_error"
    (control / "events.jsonl").touch()
    write_json(
        control / "profile.json", {"profile": profile.public(), "binding_sha256": binding["sha256"], "turn_id": turn_id}
    )
    started_at = time.monotonic()
    try:
        async with gateway_type(
            policy=policy,
            upstream_url=profile.model.base_url,
            api_key=credential,
            output=control / "model_requests.jsonl",
            timeout_s=profile.budget.request_seconds,
            on_text=preview_text,
        ) as relay:
            async def summarize(source, limit):
                return await summarize_via_relay(relay, profile, source, limit)

            previous, memory, context_info = await cancellable(
                prepare_context(previous, memory, request_text, context_policy,
                                profile.model.context_tokens, profile.budget.output_tokens, workspace, summarize),
                lambda: _is_cancelled(sid),
            )
            preparing_context = False
            if gateway_type is EvidenceGateway:
                from agent.harness.chat_tool_budget import ChatToolBudget
                relay.chat_tool_budget = ChatToolBudget(profile.budget.tool_calls)
            state["agent_context"] = context_info
            # Persist the exact retry input once preparation succeeds. Keep the
            # original working history untouched until an answer completes.
            state["agent_retry_base"] = previous
            state["agent_retry_memory"] = memory
            await _session_store.save(sid)
            write_json(control / "context.json", {**context_info, "session_id": sid, "turn_id": turn_id,
                       "history_messages": len(previous), "history_sha256": digest(previous), "has_memory": bool(memory)})
            with (control / "runtime.log").open("wb") as err:
                env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
                proc = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-u",
                    "-m",
                    "agent.harness.web_worker",
                    cwd=str(ROOT),
                    env=env,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=err,
                    start_new_session=True,
                    limit=16_000_000,
                )
                reader = asyncio.create_task(read_worker(proc))
                request = {
                    "session_id": sid,
                    "profile": profile.public(),
                    "runtimes": runtime_paths(),
                    "control": str(control),
                    "python_paths": sys.path,
                    "model": profile.model.model,
                    "base_url": relay.base_url,
                    "token": relay.token,
                    "budget": profile.public()["budget"],
                    "workspace": str(workspace),
                    "lang": state.get("user_lang"),
                    "messages": previous,
                    "memory": memory,
                    "text": request_text,
                    "handshake": True,
                }
                proc.stdin.write((json.dumps(request) + "\n").encode())
                await proc.stdin.drain()
                try:
                    async with asyncio.timeout(max(0.01, profile.budget.wall_seconds - (time.monotonic() - started_at))):
                        while True:
                            if await _is_cancelled(sid):
                                outcome = "stopped"
                                break
                            try:
                                origin, row = await asyncio.wait_for(queue.get(), 0.25)
                            except TimeoutError:
                                continue
                            if origin == "eof":
                                break
                            if origin == "failure":
                                raise RuntimeError(row["error_type"])
                            kind = row.get("kind")
                            if origin == "preview":
                                if preview_call is not None and preview_call != row["call"]:
                                    row["text"] = "\n\n" + row["text"]
                                preview_call = row["call"]
                            elif kind == "text" and preview_call is not None:
                                # The gateway has already displayed these exact model deltas.
                                continue
                            if kind == "catalog":
                                if tools or not isinstance(row.get("tools"), list) or not row["tools"]:
                                    raise ValueError("Invalid scientific tool catalog")
                                names = row["tools"]
                                if len(set(names)) != len(names) or any(
                                    not n.startswith("mcp__venusfactory__") for n in names
                                ):
                                    raise ValueError("Invalid scientific tool identity")
                                tools = [n.removeprefix("mcp__venusfactory__") for n in names]
                                if profile.model.protocol == "responses":
                                    policy.allowed = set(names)
                                else:
                                    policy.scientific_tools = frozenset(tools)
                                write_json(control / "tools.json", names)
                                proc.stdin.write(b"ready\n")
                                await proc.stdin.drain()
                            elif kind == "interaction":
                                from agent.harness.interactions import Interaction
                                if state.get("pending_interaction"):
                                    raise ValueError("Duplicate pending interaction")
                                gate = Interaction.model_validate(row["interaction"]).model_dump()
                                state["pending_interaction"] = gate
                                state["agent_pending_turn"]["interaction"] = gate
                                state["status"] = "waiting_for_input"
                                state["waiting_for"] = "user_interaction"
                                outcome = "waiting_for_input"
                                await _session_store.save(sid)
                                yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                            elif kind == "model_result":
                                if record_model_result(state, row, turn_id):
                                    await _session_store.save(sid)
                                    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                            elif kind == "completed":
                                if completed is not None:
                                    raise ValueError("Duplicate completion")
                                completed = row
                            elif kind == "error":
                                state["error"] = row.get("error_type", "Agent execution failed")
                            elif kind in {"text", "tool_call_start", "tool_result"}:
                                if kind != "text":
                                    record_tool_observation(state["agent_pending_turn"], row, profile.harness, workspace=workspace)
                                    if kind == "tool_result":
                                        await _session_store.save(sid)
                                    with (control / "events.jsonl").open("a") as log:
                                        log.write(json.dumps(row, ensure_ascii=False) + "\n")
                                event = KimiEvent(
                                    kind=kind,
                                    turn_id=turn_id,
                                    text=row.get("text", ""),
                                    tool_call_id=row.get("tool_call_id", ""),
                                    tool_name=row.get("tool_name", ""),
                                    tool_args=row.get("args", {}),
                                    tool_output=row.get("output"),
                                    is_error=bool(row.get("is_error")),
                                )
                                for frame in _apply_event(state, event):
                                    yield frame
                        if outcome != "stopped":
                            await proc.wait()
                            if completed and proc.returncode == 0 and outcome != "waiting_for_input":
                                outcome = "completed"
                finally:
                    await stop_process(proc)
    except TimeoutError:
        outcome = "timeout"
    except asyncio.CancelledError:
        outcome = "stopped"
        raise
    except Exception as exc:
        state["error"] = type(exc).__name__
    finally:
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        await stop_process(proc)
        request_audit = audit_requests(control / "model_requests.jsonl")
        event_audit = audit_trace(control / "events.jsonl", tools, profile.budget.tool_calls)
        passed = request_audit["passed"] and event_audit["passed"]
        recovery = None
        if outcome == "completed" and event_audit["passed"] and not request_audit["passed"]:
            from backend.web_v2.agent_recovery import completed_turn_recovery, canonical_recovered_history
            recovery = completed_turn_recovery(control / "model_requests.jsonl", profile.budget.model_calls)
            if recovery["accepted"]:
                passed = True
                state["history"] = canonical_recovered_history(
                    state.get("history", []), turn_id, completed["text"])
        if outcome == "completed" and not passed:
            outcome = "protocol_failed"
        if outcome == "completed":
            state["agent_messages"] = completed["messages"] if profile.harness == "langgraph" else completed_messages(
                previous, request_text, state["agent_pending_turn"]["observations"], completed["text"], profile.harness)
            state.pop("agent_pending_turn", None)
            state["agent_memory"] = memory
            state.setdefault("conversation_log", []).append(
                {
                    "role": "assistant",
                    "content": redact_obj_for_frontend(completed["text"]),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        if outcome == "waiting_for_input":
            from agent.harness.interactions import settle_interaction_tools
            settle_interaction_tools(state.get("tool_executions", []), turn_id, state["pending_interaction"])
        for item in state.get("tool_executions", []):
            if item.get("turn_id") == turn_id and item.get("status") == "running":
                item["status"] = "interrupted"
        state["history"] = [
            h
            for h in state.get("history", [])
            if not (h.get("kind") == "thinking" and not str(h.get("content") or "").strip())
        ]
        state["status"] = outcome if outcome in {"completed", "stopped", "waiting_for_input"} else "error"
        state["agent_last_outcome"] = outcome
        if outcome not in {"completed", "stopped", "waiting_for_input"}:
            from agent.harness.failure_summary import public_failure
            state["error"] = public_failure(control / "model_requests.jsonl", outcome)["message"]
        write_json(
            control / "run.json",
            {"status": outcome, "binding_sha256": binding["sha256"], "requests": request_audit, "events": event_audit,
             **({"conversation_recovery": recovery} if recovery is not None else {})},
        )
        await _session_store.save(sid)
    if outcome not in {"completed", "stopped", "waiting_for_input"}:
        from agent.harness.failure_summary import public_failure
        yield f"event: error\ndata: {_to_json(public_failure(control / 'model_requests.jsonl', outcome))}\n\n"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"
