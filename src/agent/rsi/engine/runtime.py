"""Bounded offline runner using the existing Kimi REST/WebSocket client."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.harness.events import normalize_kimi_tool_result
from agent.kimi_client import KimiClient

from .storage import atomic_json


class TaskBudgetExceeded(RuntimeError):
    pass


class ConversationNeedsInput(RuntimeError):
    def __init__(self, questions):
        self.questions = questions
        super().__init__("conversation requested user input")


def safe_approval(approval: dict, allowed_mcp_tools: frozenset[str] | None = None) -> bool:
    name = str(approval.get("tool_name") or approval.get("toolName") or approval.get("name") or "")
    tool = approval.get("tool")
    if isinstance(tool, dict):
        name = str(tool.get("name") or name)
    if name.startswith("mcp__venusfactory__"):
        return name.split("__")[-1] in (
            allowed_mcp_tools
            if allowed_mcp_tools is not None
            else {
                "sequence_properties",
                "audit_prediction_table",
                "convert_temperature",
                "check_mutation",
                "fit_linear_fixture",
                "validate_linear_fixture",
                "predict_linear_fixture",
            }
        )
    # Native file tools remain subject to the existing online path policy.
    if name.lower() not in {"read", "write", "edit", "glob", "grep"}:
        return False
    from agent.kimi_security import decide

    normalized = dict(approval, tool_name=name)
    return decide(normalized, session_dir="/workspace", mode="online").allowed


def build_task_message(system_prompt, prompt, *, interaction="bounded"):
    if interaction == "chat":
        return (
            system_prompt + "\nAnswer the user's current question or carry out the requested task. "
            "Use the available scientific tools only when useful. Direct answers and clarification questions "
            "are valid; do not invent a persistent research goal or require a file artifact. "
            "If information is missing, ask the user in your reply.\n\n" + prompt
        )
    if interaction == "research_execution":
        return (
            "Scientist execution: continue the assigned research goal.\n" + system_prompt
            + "\nFollow the current plan within the supplied budget using only declared VenusFactory MCP tools "
            "and this session workspace. Preserve the scientific objective, acceptance criteria and data splits. "
            "Save actual work and a research checkpoint before ending the iteration; an ended iteration is not "
            "a completed study or an accepted model. Use ask_user or request_confirmation for essential missing "
            "input or permission, then preserve progress and wait.\n\nCurrent research context:\n" + prompt
        )
    if interaction != "bounded":
        raise ValueError("Unknown task interaction")
    return (
        "Research task instructions (execute in this turn):\n"
        + system_prompt
        + "\nUse only the declared venusfactory MCP tools. Do not enter plan mode, "
        "create a goal, ask questions, or invoke native tools. Complete the task now." + "\n\nTask:\n" + prompt
    )


async def run_task(
    client: KimiClient,
    *,
    prompt: str,
    system_prompt: str,
    model: str,
    output_dir: Path,
    timeout_s: float = 180,
    max_tool_calls: int = 12,
    allowed_mcp_tools: frozenset[str] | None = None,
    request_gateway: bool = False,
    interaction: str = "bounded",
    mcp_startup_timeout_s: float = 25,
) -> dict:
    if (
        type(mcp_startup_timeout_s) not in (int, float)
        or not math.isfinite(mcp_startup_timeout_s)
        or mcp_startup_timeout_s <= 0
    ):
        raise ValueError("Invalid MCP startup timeout")
    submitted_text = build_task_message(system_prompt, prompt, interaction=interaction)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "starting",
        "mcp_startup_timeout_s": min(mcp_startup_timeout_s, timeout_s),
        "interaction": interaction,
        "session_id": None,
        "prompt_id": None,
        "model_requested": model,
        "text": "",
        "tool_calls": 0,
        "tool_errors": 0,
        "approvals": [],
        "usage": None,
        "usage_status": "not_reported",
        "wall_seconds": None,
        "allowed_mcp_tools": sorted(allowed_mcp_tools) if allowed_mcp_tools is not None else "engineering_default",
    }
    events_path = output_dir / "events.jsonl"
    monitor = None
    stream = None
    pending_event = None
    stage = "session_setup"
    try:
        async with asyncio.timeout(timeout_s):
            sid = await client.create_session(
                cwd="/workspace",
                title="VF RSI bounded research session",
                mcp_servers=["venusfactory"],
                thinking="medium",
                permission_mode="manual",
                model=model,
                system_prompt=system_prompt,
            )
            result["session_id"] = sid
            # v0.19.2 ignores create-session agent_config. Profile applies these
            # fields, but not custom system prompts or tool lists. Verify reality.
            await client.update_profile(
                sid,
                agent_config={
                    "model": model,
                    "thinking": "medium",
                    "permission_mode": "manual",
                    "plan_mode": False,
                    "swarm_mode": False,
                },
            )
            status = await client.get_status(sid)
            if status.get("model") != model or status.get("permission") != "manual":
                raise RuntimeError("runtime model or permission binding mismatch")
            if status.get("plan_mode") or status.get("swarm_mode"):
                raise RuntimeError("unexpected runtime orchestration mode")
            result["profile_verified"] = {
                k: status.get(k) for k in ("model", "thinking_level", "permission", "plan_mode", "swarm_mode")
            }
            # The installed runtime has a fixed native system prompt. Do not
            # misrepresent a requested custom prompt as an applied system message.
            result["instruction_binding"] = "task_message_prelude; native system prompt retained"
            result["submitted_text_sha256"] = hashlib.sha256(submitted_text.encode()).hexdigest()
            (output_dir / "submitted_prompt.txt").write_text(submitted_text)
            stage = "mcp_connection"
            async with asyncio.timeout(min(mcp_startup_timeout_s, timeout_s)):
                while True:
                    servers = await client.list_mcp_servers()
                    if any(x.get("id") == "venusfactory" and x.get("status") in {"failed", "error"} for x in servers):
                        result["error_code"] = "mcp_startup_failed"
                        raise RuntimeError("Scientific MCP connection failed before model submission")
                    if any(x.get("id") == "venusfactory" and x.get("status") == "connected" for x in servers):
                        break
                    await asyncio.sleep(0.3)
            stage = "tool_discovery"
            tool_response = await client._http.get("/api/v1/tools", params={"session_id": sid})
            tool_response.raise_for_status()
            visible = [x["name"] for x in tool_response.json().get("data", {}).get("tools", [])]
            result["visible_tools"] = sorted(visible)
            result["tool_visibility_enforced"] = bool(request_gateway)
            result["tool_visibility_scope"] = "outbound API function list" if request_gateway else "not enforced"
            result["native_prompt_retained"] = True
            if allowed_mcp_tools is not None and not {
                "mcp__venusfactory__" + name for name in allowed_mcp_tools
            }.issubset(visible):
                raise RuntimeError("declared MCP tools missing from runtime")
            # Preserve only public runtime metadata; never export full profiles.
            result["initial_status"] = {k: status[k] for k in ("status", "model", "provider") if k in status}

            async def approvals():
                while True:
                    for item in await client.list_pending_approvals(sid):
                        aid = str(item.get("id") or item.get("approval_id") or item.get("approvalId") or "")
                        if not aid:
                            raise RuntimeError("approval response missing id")
                        allowed = safe_approval(item, allowed_mcp_tools)
                        result["approvals"].append(
                            {
                                "id": aid,
                                "allowed": allowed,
                                "tool": item.get("tool_name") or item.get("toolName") or item.get("name"),
                                "schema_keys": sorted(item),
                            }
                        )
                        if allowed:
                            await client.decide_approval(sid, aid, decision="approved")
                        else:
                            await client.reject(
                                sid,
                                aid,
                                feedback="This research session allows only its declared MCP tools and workspace file operations.",
                            )
                    questions = await client.list_pending_questions(sid)
                    if questions:
                        if interaction in {"chat", "research_execution"}:
                            raise ConversationNeedsInput(questions)
                        raise RuntimeError("task requested human input; task is not autonomous")
                    await asyncio.sleep(0.4)

            stage = "event_subscription"
            monitor = asyncio.create_task(approvals())
            stream = client.stream_session_events(sid)
            submitted = False
            with events_path.open("w", encoding="utf-8") as events:
                while True:
                    pending_event = asyncio.create_task(anext(stream))
                    await asyncio.wait({pending_event, monitor}, return_when=asyncio.FIRST_COMPLETED)
                    if monitor.done():
                        monitor.result()
                        raise RuntimeError("approval monitor stopped unexpectedly")
                    try:
                        event = pending_event.result()
                    except StopAsyncIteration as exc:
                        raise RuntimeError("event stream ended without turn completion") from exc
                    item = {
                        "kind": event.kind,
                        "type": event.raw_type,
                        "turn_id": event.turn_id,
                        "tool_name": event.tool_name,
                        "tool_call_id": event.tool_call_id,
                        "is_error": event.is_error,
                    }
                    if event.kind != "thinking":
                        item["text"] = event.text[:20000]
                    if event.kind == "tool_call_start":
                        result["tool_calls"] += 1
                        item["args"] = event.tool_args
                        if result["tool_calls"] > max_tool_calls:
                            raise TaskBudgetExceeded("tool call budget exceeded")
                    if event.kind == "tool_result":
                        result["tool_errors"] += int(event.is_error)
                        # Tool returns are scientific evidence. Downstream review
                        # contexts build bounded previews; the execution ledger must
                        # retain exact source, numerical feedback and receipt text.
                        item.update(normalize_kimi_tool_result(event.tool_output, is_error=event.is_error))
                    if event.raw_type == "turn.step.interrupted" and event.raw.get("reason") == "error":
                        message = str(event.raw.get("message") or "")
                        if "401" in message or "Authentication" in message:
                            result["error_code"] = "provider_authentication_failed"
                            result["error"] = "Provider rejected the configured credential (401)."
                        elif "empty response" in message.lower():
                            result["error_code"] = "provider_empty_response"
                            result["error"] = "Provider returned neither content nor tool calls."
                        else:
                            result["error_code"] = "step_interrupted"
                    for key in ("usage", "token_usage", "tokenUsage"):
                        if isinstance(event.raw.get(key), dict):
                            # Store snapshots instead of incorrectly summing cumulative usage.
                            result["usage"] = event.raw[key]
                            result["usage_status"] = "runtime_reported_unverified_semantics"
                    events.write(json.dumps(item, ensure_ascii=False) + "\n")
                    events.flush()
                    if event.kind == "tool_call_start" and allowed_mcp_tools is not None:
                        if event.tool_name not in {"mcp__venusfactory__" + name for name in allowed_mcp_tools}:
                            raise RuntimeError("undeclared tool invoked: " + event.tool_name)
                    if event.kind == "subscribed" and not submitted:
                        response = await client.submit_prompt(sid, submitted_text)
                        result["prompt_id"] = str(
                            response.get("id") or response.get("prompt_id") or response.get("promptId") or ""
                        )
                        stage = "model_execution"
                        result["status"] = "running"
                        submitted = True
                    elif event.kind == "text":
                        result["text"] += event.text
                    elif event.kind == "error":
                        raise RuntimeError(f"runtime error {event.error_code}: {event.text[:300]}")
                    elif event.kind == "turn_ended" and submitted:
                        result["end_reason"] = event.text
                        result["status"] = (
                            "runtime_error" if event.is_error or result.get("error_code") else "completed"
                        )
                        if interaction in {"chat", "research_execution"} and result["status"] == "completed" and not result["text"].strip():
                            result["status"] = "incomplete"
                        break
    except ConversationNeedsInput as exc:
        result.update(status="needs_input", questions=exc.questions)
    except TimeoutError:
        result.update(status="timeout", timeout_stage=stage)
    except TaskBudgetExceeded as exc:
        result.update(status="budget_exceeded", error=str(exc))
    except Exception as exc:
        result.update(status="runtime_error", error=f"{type(exc).__name__}: {exc}")
    finally:
        if monitor is not None:
            monitor.cancel()
            try:
                await monitor
            except (asyncio.CancelledError, Exception):
                pass
        if pending_event is not None:
            pending_event.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await pending_event
        if stream is not None:
            await stream.aclose()
        if result["status"] != "completed" and result["session_id"] and result["prompt_id"]:
            try:
                async with asyncio.timeout(3):
                    await client.abort_prompt(result["session_id"], result["prompt_id"])
            except Exception:
                pass
        if result["session_id"]:
            try:
                async with asyncio.timeout(3):
                    response = await client._http.get(f"/api/v1/sessions/{result['session_id']}/snapshot")
                    response.raise_for_status()
                    data = response.json().get("data", {})
                    session = data.get("session", data)
                    usage = session.get("usage")
                    if isinstance(usage, dict) and any(usage.get(key, 0) for key in ("input_tokens", "output_tokens")):
                        result["usage"] = {
                            key: usage.get(key)
                            for key in (
                                "input_tokens",
                                "output_tokens",
                                "cache_read_tokens",
                                "cache_creation_tokens",
                                "total_cost_usd",
                                "turn_count",
                            )
                        }
                        result["usage_status"] = "runtime_snapshot_rollup; provider billing not independently verified"
            except Exception:
                pass
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        atomic_json(output_dir / "run.json", result)
    return result
