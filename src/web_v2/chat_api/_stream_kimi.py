"""SSE streaming for the kimi-code engine.

Translates kimi server WebSocket session_event frames into the existing SSE
event shape consumed by the frontend (`event: state`, `event: token`,
`event: done`), plus one new `event: thinking` for kimi's reasoning stream.

Each VenusFactory2 chat session is mapped 1:1 to a kimi session id, stored as
state["kimi_session_id"]. The kimi session is lazy-created on the first
message so an empty chat costs nothing.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from typing import Any, AsyncIterator

from agent.kimi_client import KimiAPIError, KimiClient, KimiEvent
from agent.kimi_daemon import base_url as kimi_base_url
from agent.kimi_mcp_config import VENUSFACTORY_MCP_NAME
from agent.kimi_security import decide as kimi_security_decide
from agent.kimi_session_pool import (
    PoolExhaustedError, SpawnError, get_pool as get_kimi_pool,
)
from config import get_config
from logger import get_logger

from web_v2.chat_api._hooks_runtime import _is_cancelled, _session_store
from web_v2.chat_api._shared import _snapshot, _to_json
from web_v2.chat_api._stream import _STREAM_STATE_KEYS, _finalize_after_stream  # noqa: F401
from web_v2.chat_api._uploads import _normalize_uploaded_file
from web_v2.redact import redact_for_frontend, redact_obj_for_frontend

_logger = get_logger("chat_api.stream_kimi")


def _make_status_evt(message: str) -> str:
    return f"event: status\ndata: {_to_json({'message': message})}\n\n"


def _make_error_evt(message: str, code: str = "") -> str:
    return f"event: error\ndata: {_to_json({'message': message, 'code': code})}\n\n"


_LANG_LABEL = {"en": "English", "zh": "Chinese (中文)"}


def _language_policy_block(lang: str) -> str:
    """Mandatory language directive. The UI locale pins the response language;
    auto-detection from user input is explicitly forbidden so short prompts
    like 'ok' or code snippets don't flip the conversation language."""
    label = _LANG_LABEL.get(lang or "", "")
    if not label:
        return (
            "## Language policy\n\n"
            "Respond AND think in the SAME language as the user's most recent "
            "message. Both the visible reply and the chain-of-thought / "
            "reasoning shown in the Thinking block must match.\n"
        )
    return (
        "## Language policy — MANDATORY, OVERRIDES INPUT DETECTION\n\n"
        f"You MUST respond ONLY in **{label}**, regardless of the language of "
        "the user's input, regardless of the language of file contents or "
        "tool outputs, regardless of any earlier turns. Both your visible "
        f"reply AND your chain-of-thought / reasoning (the Thinking block) "
        f"MUST be in {label}.\n\n"
        "- If the user writes in another language, still reply in "
        f"{label}.\n"
        "- If a tool returns text in another language, summarize it in "
        f"{label}.\n"
        f"- Code, identifiers, command names and quoted error messages stay "
        "verbatim; everything else is in "
        f"{label}.\n"
        "- Do NOT translate user-supplied names, sequences, or PDB IDs.\n"
    )


def _build_system_prompt(lang: str) -> str:
    return f"""\
You are the VenusFactory2 protein-engineering assistant.

## Tool calling — read carefully

You have access to TWO namespaces of tools. They are NOT interchangeable.

1. **Kimi built-in tools** — called by their bare name, NO prefix.
   Examples: `Read`, `Write`, `Edit`, `Bash`, `FetchURL`, `Glob`, `Grep`,
   `ReadMediaFile` (for viewing local images / PDFs), `Skill`, `TaskCreate`,
   `EnterPlanMode`. These come from kimi itself and are always available.

2. **VenusFactory MCP tools** — prefixed `mcp__venusfactory__...`.
   These wrap protein-specific operations (mutation prediction, structure
   download, ESMFold, ProteinMPNN, sequence/structure DB queries, BLAST,
   ClustalO MSA, KEGG/BRENDA/ChEMBL, PubMed/arXiv search, etc.).

**Hard rule**: Only call tools that appear verbatim in your tool catalog.
Do NOT prefix kimi built-ins with `mcp__venusfactory__`. Do NOT invent
plausible-sounding names. If a capability is missing, say so in the answer
instead of fabricating a tool call.

Prefer calling tools over reasoning from memory when the user asks for
protein-specific facts, structures, or computations.

{_language_policy_block(lang)}"""


# Back-compat for any importer that still references the old constant.
_DEFAULT_SYSTEM_PROMPT = _build_system_prompt("")


async def _ensure_kimi_session(client: KimiClient, state: dict[str, Any]) -> str:
    sid = str(state.get("kimi_session_id") or "")
    if sid:
        # Confirm it still exists on the kimi side.
        try:
            await client.get_session(sid)
            return sid
        except KimiAPIError as exc:
            _logger.info("kimi session %s missing (%s); recreating", sid, exc)
            sid = ""
    cwd = state["agent_session_dir"]
    os.makedirs(cwd, exist_ok=True)
    sid = await client.create_session(
        cwd=cwd,
        title=f"VenusFactory chat {state.get('session_id', '')[:8]}",
        mcp_servers=[VENUSFACTORY_MCP_NAME],
        thinking="high",
        # IMPORTANT: use "manual" not "yolo" so every tool call surfaces in the
        # approval queue. `kimi_security.decide` is the only auto-approver and
        # rejects what falls outside the policy. With "yolo" kimi would auto-
        # allow most tools internally and our policy would never see them.
        permission_mode="manual",
        system_prompt=_build_system_prompt(str(state.get("user_lang") or "")),
    )
    state["kimi_session_id"] = sid
    return sid


def _language_pin(lang: str) -> str:
    """Per-turn reminder prepended to the user message. Belt-and-suspenders
    for sessions whose system_prompt was baked with a different language
    (user switched UI locale mid-session — kimi sessions are immutable so we
    can't update the system_prompt after create_session)."""
    label = _LANG_LABEL.get(lang or "", "")
    if not label:
        return ""
    return f"[Reply in {label} only — both visible answer and Thinking block.]\n\n"


def _apply_event(state: dict[str, Any], ev: KimiEvent) -> list[str]:
    """Mutate state in-place. Return zero or more SSE frames to emit now."""
    history: list[dict[str, Any]] = state.setdefault("history", [])
    tool_execs: list[dict[str, Any]] = state.setdefault("tool_executions", [])
    frames: list[str] = []

    if ev.kind == "thinking":
        last = history[-1] if history else None
        if not (last and last.get("role") == "assistant"
                and last.get("kind") == "thinking"
                and last.get("turn_id") == ev.turn_id):
            history.append({
                "role": "assistant",
                "content": "",
                "kind": "thinking",
                "turn_id": ev.turn_id,
            })
            last = history[-1]
        last["content"] += ev.text
        frames.append(f"event: thinking\ndata: {_to_json({'content': ev.text, 'turn_id': ev.turn_id})}\n\n")

    elif ev.kind == "text":
        last = history[-1] if history else None
        if not (last and last.get("role") == "assistant"
                and last.get("kind") in ("text", None)
                and last.get("turn_id") == ev.turn_id):
            history.append({
                "role": "assistant",
                "content": "",
                "kind": "text",
                "turn_id": ev.turn_id,
            })
            last = history[-1]
        last["content"] += ev.text
        frames.append(f"event: token\ndata: {_to_json({'content': ev.text, 'turn_id': ev.turn_id, 'role_id': 'assistant'})}\n\n")

    elif ev.kind == "tool_call_start":
        # Redact args at store time so persisted session state never holds
        # raw key=value tokens / host paths that may have been passed in.
        # _snapshot() also redacts on read, this is defense in depth at rest.
        tool_execs.append({
            "tool_call_id": ev.tool_call_id,
            # Write both `name` (kimi-native) and `tool_name` (legacy graph
            # convention the frontend reads). Avoids a rename across the UI.
            "name": ev.tool_name,
            "tool_name": ev.tool_name,
            "args": redact_obj_for_frontend(ev.tool_args),
            "status": "running",
            "turn_id": ev.turn_id,
            "started_at": datetime.now().isoformat(),
            "timestamp": datetime.now().isoformat(),
        })
        frames.append(f"event: state\ndata: {_to_json(_snapshot(state))}\n\n")

    elif ev.kind == "tool_result":
        # Same: redact tool output before it lands in state. MCP tool results
        # are often JSON dicts with paths and occasional traceback text that
        # may leak env vars; this catches the most common cases.
        redacted_output = redact_obj_for_frontend(ev.tool_output)
        for te in tool_execs:
            if te.get("tool_call_id") == ev.tool_call_id:
                te["output"] = redacted_output
                te["status"] = "error" if ev.is_error else "ok"
                te["finished_at"] = datetime.now().isoformat()
                break
        else:
            tool_execs.append({
                "tool_call_id": ev.tool_call_id,
                "output": redacted_output,
                "status": "error" if ev.is_error else "ok",
                "turn_id": ev.turn_id,
                "finished_at": datetime.now().isoformat(),
                "timestamp": datetime.now().isoformat(),
            })
        frames.append(f"event: state\ndata: {_to_json(_snapshot(state))}\n\n")

    elif ev.kind == "turn_started":
        # Use "chat_mode" so PipelineProgress shows the "Responding…" pulse
        # dot. The legacy "running" status isn't in STATUS_TO_STAGE and
        # renders as nothing — invisible to the user.
        state["status"] = "chat_mode"
        frames.append(f"event: stream_start\ndata: {_to_json({'role_id': 'assistant', 'turn_id': ev.turn_id})}\n\n")

    elif ev.kind == "turn_ended":
        state["status"] = "completed" if not ev.is_error else "error"
        frames.append(f"event: state\ndata: {_to_json(_snapshot(state))}\n\n")

    elif ev.kind == "error":
        state["status"] = "error"
        state["error"] = ev.text or ev.error_code
        frames.append(_make_error_evt(ev.text or "kimi error", code=ev.error_code))

    return frames


async def _stream_kimi(
    state: dict[str, Any],
    text: str,
    attachment_paths: list[str],
) -> AsyncIterator[str]:
    """Stream a single user turn through kimi-code."""
    session_id = state["session_id"]

    if await _is_cancelled(session_id):
        state["status"] = "stopped"
        await _session_store.save(session_id)
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    agent_session_dir = state.get("agent_session_dir")
    if not agent_session_dir:
        raise RuntimeError("Agent session directory is missing.")
    os.makedirs(agent_session_dir, exist_ok=True)

    # Normalize uploads into the session dir so kimi (whose cwd is that dir)
    # can see them.
    valid_attachments: list[str] = []
    for p in attachment_paths or []:
        normalized = await _normalize_uploaded_file(
            p,
            agent_session_dir,
            state.setdefault("temp_files", []),
            str(state.get("owner_key", "")),
        )
        if normalized:
            valid_attachments.append(normalized)

    display_text = (text or "").strip()
    if valid_attachments:
        names = ", ".join(os.path.basename(p) for p in valid_attachments)
        display_text = (display_text + f"\n📎 Attached: {names}").strip()

    if not display_text:
        await _session_store.save(session_id)
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    state.setdefault("history", []).append({"role": "user", "content": display_text})
    state.setdefault("conversation_log", []).append({
        "role": "user", "content": display_text, "timestamp": datetime.now().isoformat()
    })
    state["last_user_text"] = text
    state["last_attachment_paths"] = valid_attachments
    state["status"] = "started"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    # In ONLINE mode, every chat session gets its own sandboxed kimi instance
    # via the per-session pool (UID isolation, bind-mounted /workspace, capped
    # concurrency). In LOCAL mode we use the single shared daemon spawned at
    # api_server startup — simpler for dev, same daemon already running.
    mode = get_config().server.mode or "local"
    if mode == "online":
        pool = get_kimi_pool()
        try:
            inst = await pool.acquire(session_id)
        except PoolExhaustedError as exc:
            yield _make_error_evt(
                f"Server at capacity: {exc}. Please retry in a few minutes.",
                code="pool_exhausted",
            )
            state["status"] = "error"
            state["error"] = "pool_exhausted"
            await _session_store.save(session_id)
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        except SpawnError as exc:
            yield _make_error_evt(
                f"Failed to spawn isolated kimi instance: {exc}",
                code="spawn_failed",
            )
            state["status"] = "error"
            state["error"] = "spawn_failed"
            await _session_store.save(session_id)
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return
        # If pool GC stopped the previous instance, kimi_session_id from
        # state points at a dead kimi server. Clear it so _ensure_kimi_session
        # creates a fresh kimi-side session in the new instance.
        if state.get("_pool_scope") and state.get("_pool_scope") != inst.scope_name:
            state["kimi_session_id"] = ""
        state["_pool_scope"] = inst.scope_name
        client_base_url = inst.base_url
    else:
        client_base_url = kimi_base_url()

    client = KimiClient(base_url=client_base_url)
    try:
        # Lazy create / verify kimi session
        try:
            kimi_sid = await _ensure_kimi_session(client, state)
        except KimiAPIError as exc:
            msg = (
                "Kimi server is not ready. Run `kimi provider catalog add moonshot` "
                "and `kimi login` in a terminal, then retry."
                if exc.code in (None,) or "auth" in str(exc).lower()
                else f"kimi error: {exc}"
            )
            yield _make_error_evt(msg, code=str(exc.code or ""))
            state["status"] = "error"
            state["error"] = msg
            await _session_store.save(session_id)
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        # Subscribe via WS, then submit prompt and translate events. Direct
        # async-for over the client generator — no pump task, no queue (the
        # earlier queue+pump variant racily lost frames after the first one).
        prompt_submitted = False

        # Mode-aware security policy (loose for local, strict for online).
        mode = get_config().server.mode or "local"

        async def _auto_approve_all() -> None:
            """Drain pending approvals through the security policy.

            For each pending approval ask `kimi_security.decide`. If allowed,
            POST `approved` (scope=session so we don't re-prompt for the same
            tool). If denied, POST `rejected` with the human-readable reason
            as `feedback` so the agent knows what got blocked and can plan
            around it. Every decision is audit-logged; denials are also
            appended to `state["security_events"]` for later review and
            emitted as an `event: security_denied` SSE frame (no UI consumer
            yet, but it's there).
            """
            try:
                pending = await client.list_pending_approvals(kimi_sid)
            except Exception:
                _logger.exception("list_pending_approvals failed")
                return

            for ap in pending:
                aid = ap.get("approval_id")
                if not aid:
                    continue
                tool = str(ap.get("tool_name") or "")
                action = str(ap.get("action") or "")
                decision = kimi_security_decide(
                    ap, session_dir=agent_session_dir, mode=mode
                )
                if decision.allowed:
                    try:
                        await client.approve(kimi_sid, aid, scope="session")
                        _logger.info(
                            "kimi-security ALLOW tool=%s action=%s reason=%s",
                            tool, action, decision.reason,
                        )
                    except Exception:
                        _logger.exception("approve POST failed for %s", aid)
                else:
                    try:
                        await client.reject(
                            kimi_sid, aid,
                            feedback=f"Blocked by VenusFactory security policy: {decision.reason}",
                        )
                    except Exception:
                        _logger.exception("reject POST failed for %s", aid)
                    _logger.warning(
                        "kimi-security DENY tool=%s action=%s reason=%s input=%r",
                        tool, action, decision.reason,
                        ap.get("tool_input_display"),
                    )
                    sec_events = state.setdefault("security_events", [])
                    sec_events.append({
                        "ts": datetime.now().isoformat(),
                        "tool": tool,
                        "action": action,
                        "reason": decision.reason,
                        "input_preview": str(ap.get("tool_input_display"))[:500],
                    })
                    # bound the list so a noisy run doesn't bloat the session
                    if len(sec_events) > 50:
                        del sec_events[:-50]

        try:
            async for ev in client.stream_session_events(kimi_sid):
                if await _is_cancelled(session_id):
                    state["status"] = "stopped"
                    state["history"].append({
                        "role": "assistant",
                        "content": "Run stopped by user.",
                        "role_id": "principal_investigator",
                    })
                    await _session_store.save(session_id)
                    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                    yield "event: done\ndata: {}\n\n"
                    return

                if ev.kind == "subscribed":
                    # Fire submit_prompt in the background so we don't block
                    # the WS iterator while waiting on the HTTP round-trip.
                    # Blocking here lets the websockets reader buffer fill up
                    # past internal limits and silently drop later messages.
                    if not prompt_submitted:
                        prompt_submitted = True
                        # Per-turn language pin overrides the session-level
                        # system prompt if the user switched UI locale after
                        # session creation (kimi sessions are immutable).
                        pin = _language_pin(str(state.get("user_lang") or ""))
                        asyncio.create_task(
                            client.submit_prompt(kimi_sid, pin + display_text)
                        )
                    continue

                # Auto-approve any pending kimi tool-call approvals
                # (MCP / Bash / Edit don't respect permission_mode=yolo).
                # Trigger off the session.status_changed event; also fall
                # back to scanning when any tool_call_start arrives, just
                # in case kimi raises approvals without flipping status.
                if ev.raw_type == "event.session.status_changed" and \
                        ev.raw.get("status") == "awaiting_approval":
                    asyncio.create_task(_auto_approve_all())
                elif ev.kind == "tool_call_start":
                    asyncio.create_task(_auto_approve_all())

                for frame in _apply_event(state, ev):
                    yield frame
                if ev.kind == "turn_ended" or ev.kind == "error":
                    break
        except Exception as exc:  # noqa: BLE001
            _logger.exception("kimi WS stream crashed")
            yield _make_error_evt(f"kimi stream error: {exc}")
            state["status"] = "error"
    finally:
        await client.aclose()

    # Final bookkeeping mirroring the graph engine's _finalize_after_stream
    # (memory + conversation_log + archive). status is set above.
    try:
        final = ""
        for item in reversed(state.get("history", [])):
            if item.get("role") == "assistant" and item.get("kind") != "thinking":
                final = item.get("content", "") or ""
                break
        if final:
            state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": final, "timestamp": datetime.now().isoformat(),
            })
            memory = state.get("memory")
            if memory is not None:
                try:
                    memory.save_context({"input": display_text}, {"output": final})
                except Exception:
                    pass
    except Exception:
        _logger.exception("kimi stream: finalization bookkeeping failed")

    await _session_store.save(session_id)
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"
