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
from web_v2.chat_api._shared import _snapshot, _to_json, clear_expert_gates
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

When requirements or preferences are ambiguous, use **AskUserQuestion**
(structured multi-choice). VenusFactory surfaces it in the chat UI; do not
ask the same clarification as free-form prose when AskUserQuestion fits.
For complex multi-step work, EnterPlanMode / ExitPlanMode is supported —
ExitPlanMode pauses for the user to approve (or pick an approach).

## Paths (online sandbox)

VenusFactory MCP tools write artifacts into this chat session directory.
That directory is your cwd and is also mounted at `/workspace`. Prefer
**relative paths** (or paths under `/workspace/...`) when calling `Read` /
`Glob` / `Bash`. Absolute host paths under the session directory also work;
paths outside the session are blocked.

{_language_policy_block(lang)}"""


# Back-compat for any importer that still references the old constant.
_DEFAULT_SYSTEM_PROMPT = _build_system_prompt("")


async def _ensure_kimi_session(client: KimiClient, state: dict[str, Any]) -> str:
    desired_model = str(state.get("kimi_model") or "").strip() or None
    sid = str(state.get("kimi_session_id") or "")
    if sid:
        bound = str(state.get("_kimi_bound_model") or "")
        # Local picker may change the underlying LLM; kimi sessions bake model
        # into agent_config at create time, so recreate when it diverges.
        if (desired_model or "") != bound:
            _logger.info(
                "kimi model changed (%r → %r); recreating session",
                bound,
                desired_model or "",
            )
            sid = ""
            state["kimi_session_id"] = ""
        else:
            # Confirm it still exists on the kimi side.
            try:
                await client.get_session(sid)
                return sid
            except KimiAPIError as exc:
                _logger.info("kimi session %s missing (%s); recreating", sid, exc)
                sid = ""
    # Online bwrap mounts sessions/<sid> at /workspace. Use that pool root as
    # kimi cwd so relative Read/Glob paths match the sandbox layout (and the
    # host dual-bind of the same directory).
    mode = get_config().server.mode or "local"
    if mode == "online":
        from agent.kimi_session_pool import _project_session_dir
        cwd = _project_session_dir(str(state.get("session_id") or ""))
    else:
        cwd = str(state.get("agent_session_dir") or "")
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
        model=desired_model,
        system_prompt=_build_system_prompt(str(state.get("user_lang") or "")),
    )
    state["kimi_session_id"] = sid
    state["_kimi_bound_model"] = desired_model or ""
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


def map_kimi_questions_to_ui(item: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map one kimi pending-question item → AskUserCard shape + reverse map."""
    ui_qs: list[dict[str, Any]] = []
    rev_qs: list[dict[str, Any]] = []
    for q in item.get("questions") or []:
        if not isinstance(q, dict):
            continue
        options = list(q.get("options") or [])
        labels: list[str] = []
        opt_ids: list[str] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            oid = str(opt.get("id") or "")
            label = str(opt.get("label") or "")
            if not oid or not label:
                continue
            labels.append(label)
            opt_ids.append(oid)
        allow_other = bool(q.get("allow_other"))
        other_label = str(q.get("other_label") or "Other")
        if allow_other:
            labels.append(other_label)
            opt_ids.append("__other__")
        # kimi requires ≥2 options; if truncated, still surface the question.
        if len(labels) < 2:
            labels = labels + ["Other", "Skip"][: max(0, 2 - len(labels))]
            while len(opt_ids) < len(labels):
                opt_ids.append("__skip__" if opt_ids.count("__skip__") == 0 else f"__pad{len(opt_ids)}")
            if "Other" in labels and not allow_other:
                allow_other = True
                other_label = "Other"
        qid = str(q.get("id") or "")
        question_text = str(
            q.get("question") or q.get("header") or q.get("body") or ""
        ).strip()
        header = str(q.get("header") or "").strip()
        ui_qs.append({
            "question": question_text,
            "header": header if header and header != question_text else "",
            "options": labels,
            "allow_multiple": bool(q.get("multi_select")),
            "allow_other": allow_other,
            "other_label": other_label,
        })
        rev_qs.append({
            "id": qid,
            "option_ids": opt_ids,
            "allow_other": allow_other,
            "multi_select": bool(q.get("multi_select")),
        })
    rev = {
        "question_id": str(item.get("question_id") or ""),
        "questions": rev_qs,
    }
    return ui_qs, rev


def ui_answers_to_kimi(answers_data: list[dict[str, Any]], rev: dict[str, Any]) -> dict[str, Any]:
    """Convert ClarificationForm answers → kimi `/questions/{id}` body.answers."""
    out: dict[str, Any] = {}
    rev_qs = list(rev.get("questions") or [])
    for a in answers_data:
        qi = int(a.get("question_index") or 0)
        if qi < 0 or qi >= len(rev_qs):
            continue
        meta = rev_qs[qi]
        qid = str(meta.get("id") or "")
        if not qid:
            continue
        selected = list(a.get("selected_options") or [])
        custom = str(a.get("custom_text") or "").strip()
        opt_ids = list(meta.get("option_ids") or [])
        chosen: list[str] = []
        for idx in selected:
            try:
                i = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(opt_ids):
                chosen.append(opt_ids[i])
        has_other = "__other__" in chosen
        has_skip = "__skip__" in chosen
        real_ids = [x for x in chosen if x not in ("__other__", "__skip__", "__pad")]
        real_ids = [x for x in real_ids if not str(x).startswith("__pad")]
        multi = bool(meta.get("multi_select"))
        if has_skip and not real_ids and not has_other:
            out[qid] = {"kind": "skipped"}
        elif has_other and multi:
            out[qid] = {
                "kind": "multi_with_other",
                "option_ids": real_ids,
                "other_text": custom,
            }
        elif has_other:
            out[qid] = {"kind": "other", "text": custom}
        elif multi:
            out[qid] = {"kind": "multi", "option_ids": real_ids or chosen}
        elif real_ids:
            out[qid] = {"kind": "single", "option_id": real_ids[0]}
        elif chosen:
            out[qid] = {"kind": "single", "option_id": chosen[0]}
        else:
            out[qid] = {"kind": "skipped"}
    return out


def _pause_for_kimi_questions(state: dict[str, Any], item: dict[str, Any]) -> None:
    ui_qs, rev = map_kimi_questions_to_ui(item)
    state["engine"] = "kimi-code"
    state["chat_mode"] = "science_agent"
    state["kimi_pending_question"] = rev
    state["clarification_questions"] = ui_qs
    state["clarification_answers"] = []
    state.pop("kimi_pending_approval", None)
    state.pop("approval_prompt", None)
    state.pop("plan_markdown", None)
    state["waiting_for"] = "kimi_question"
    # Dedicated Agent status — never reuse Expert waiting_for_clarification.
    state["status"] = "waiting_for_kimi_question"


def _pause_for_kimi_approval(state: dict[str, Any], approval: dict[str, Any]) -> None:
    """Surface ExitPlanMode (or other needs_user) approval for ApprovalCard."""
    tool = str(approval.get("tool_name") or "")
    args = approval.get("tool_input_display") or {}
    if not isinstance(args, dict):
        args = {}
    options_raw = args.get("options") or []
    labels: list[str] = []
    if isinstance(options_raw, list):
        for opt in options_raw:
            if isinstance(opt, dict) and opt.get("label"):
                labels.append(str(opt["label"]))
            elif isinstance(opt, str) and opt.strip():
                labels.append(opt.strip())
    if not labels:
        labels = ["Approve", "Reject"]
    elif "Reject" not in labels and "拒绝" not in labels:
        labels = labels + ["Reject"]

    plan_text = str(
        args.get("plan")
        or args.get("content")
        or args.get("plan_content")
        or args.get("summary")
        or ""
    ).strip()
    if not plan_text:
        # Fall back to a compact dump of display args (already redacted-ish).
        try:
            import json as _json
            plan_text = _json.dumps(args, ensure_ascii=False, indent=2)[:8000]
        except Exception:
            plan_text = str(args)[:8000]

    approval_prompt = f"Agent requests approval for `{tool}`." if tool else "Agent requests your approval."
    # Keep a short radio fallback for older clients; full plan lives in plan_markdown.
    fallback_question = approval_prompt
    if plan_text and len(plan_text) <= 240:
        fallback_question = f"{approval_prompt}\n\n{plan_text}"

    state["engine"] = "kimi-code"
    state["chat_mode"] = "science_agent"
    state["kimi_pending_approval"] = {
        "approval_id": str(approval.get("approval_id") or ""),
        "tool_name": tool,
        "option_labels": labels,
        "plan_markdown": plan_text,
        "approval_prompt": approval_prompt,
    }
    state["approval_prompt"] = approval_prompt
    state["plan_markdown"] = plan_text
    state.pop("kimi_pending_question", None)
    state["clarification_questions"] = [{
        "question": fallback_question,
        "options": labels,
        "allow_multiple": False,
    }]
    state["clarification_answers"] = []
    state["waiting_for"] = "kimi_approval"
    # Dedicated Agent status — never reuse Expert waiting_for_clarification.
    state["status"] = "waiting_for_kimi_approval"


async def acquire_kimi_client_for_state(
    state: dict[str, Any],
) -> tuple[KimiClient, str]:
    """Return (client, kimi_session_id) for an existing VF chat session.

    Used by interactive endpoints after AskUser / ExitPlanMode pauses.
    """
    mode = get_config().server.mode or "local"
    if mode == "online":
        pool = get_kimi_pool()
        inst = await pool.acquire(state["session_id"])
        if state.get("_pool_scope") and state.get("_pool_scope") != inst.scope_name:
            state["kimi_session_id"] = ""
        state["_pool_scope"] = inst.scope_name
        client = KimiClient(base_url=inst.base_url)
    else:
        client = KimiClient(base_url=kimi_base_url())
    kimi_sid = await _ensure_kimi_session(client, state)
    return client, kimi_sid


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
        history = state.setdefault("history", [])
        last = history[-1] if history else None
        if not (
            last
            and last.get("role") == "assistant"
            and last.get("kind") == "thinking"
        ):
            history.append({
                "role": "assistant",
                "content": "",
                "kind": "thinking",
                "phase": "thinking",
                "turn_id": ev.turn_id,
            })
        elif ev.turn_id and not last.get("turn_id"):
            last["turn_id"] = ev.turn_id
        # Emit stream_start for frontend streamingIdx, but history already has
        # a thinking placeholder — the UI must NOT create a blank text bubble.
        frames.append(f"event: stream_start\ndata: {_to_json({'role_id': 'assistant', 'turn_id': ev.turn_id, 'kind': 'thinking'})}\n\n")
        frames.append(f"event: state\ndata: {_to_json(_snapshot(state))}\n\n")

    elif ev.kind == "turn_ended":
        # Drop empty Thinking placeholders (warm-up stubs that never received
        # thinking.delta) so completed history doesn't keep a hollow block.
        history = state.get("history") or []
        state["history"] = [
            h for h in history
            if not (
                h.get("kind") == "thinking"
                and not str(h.get("content") or "").strip()
            )
        ]
        state["status"] = "completed" if not ev.is_error else "error"
        frames.append(f"event: state\ndata: {_to_json(_snapshot(state))}\n\n")

    elif ev.kind == "error":
        state["status"] = "error"
        state["error"] = ev.text or ev.error_code
        frames.append(_make_error_evt(ev.text or "kimi error", code=ev.error_code))

    return frames


async def _drain_approvals_and_questions(
    client: KimiClient,
    state: dict[str, Any],
    kimi_sid: str,
    *,
    agent_session_dir: str,
    mode: str,
    human_gate: asyncio.Queue | None,
) -> None:
    """Auto-decide safe approvals; push needs_user / questions to human_gate.

    `human_gate` items are `("question", item)` or `("approval", approval)`.
    Only the first human interaction is queued (queue maxsize=1).
    """
    # Approvals first — MCP tools expire at 60s.
    try:
        pending = await client.list_pending_approvals(kimi_sid)
    except Exception:
        _logger.exception("list_pending_approvals failed")
        pending = []

    for ap in pending:
        aid = ap.get("approval_id")
        if not aid:
            continue
        tool = str(ap.get("tool_name") or "")
        action = str(ap.get("action") or "")
        decision = kimi_security_decide(ap, session_dir=agent_session_dir, mode=mode)
        if decision.needs_user:
            if human_gate is not None and human_gate.empty():
                try:
                    human_gate.put_nowait(("approval", ap))
                except asyncio.QueueFull:
                    pass
            continue
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
            if len(sec_events) > 50:
                del sec_events[:-50]

    # AskUserQuestion pending items.
    try:
        questions = await client.list_pending_questions(kimi_sid)
    except Exception:
        _logger.debug("list_pending_questions failed", exc_info=True)
        questions = []
    if questions and human_gate is not None and human_gate.empty():
        try:
            human_gate.put_nowait(("question", questions[0]))
        except asyncio.QueueFull:
            pass


async def _kimi_event_loop(
    client: KimiClient,
    state: dict[str, Any],
    kimi_sid: str,
    *,
    submit_text: str | None,
    after_subscribe=None,
) -> AsyncIterator[str]:
    """Subscribe to kimi WS, optionally submit a prompt, stream until pause/end.

    On AskUserQuestion / ExitPlanMode, mutates state into a waiting_* status
    and returns (caller should yield done without finalizing as completed).

    `after_subscribe`: optional async callable fired once (via create_task)
    after the WS subscribe ack — used to POST question answers / approvals
    only after we're listening, so continuation events are not missed.
    """
    session_id = state["session_id"]
    agent_session_dir = str(state.get("agent_session_dir") or "")
    mode = get_config().server.mode or "local"
    prompt_submitted = False
    after_subscribe_fired = False
    human_gate: asyncio.Queue = asyncio.Queue(maxsize=1)
    poll_stop = asyncio.Event()

    async def _poller() -> None:
        while not poll_stop.is_set():
            await _drain_approvals_and_questions(
                client, state, kimi_sid,
                agent_session_dir=agent_session_dir,
                mode=mode,
                human_gate=human_gate,
            )
            try:
                await asyncio.wait_for(poll_stop.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                pass

    poll_task = asyncio.create_task(
        _poller(), name=f"kimi-interact-poll-{session_id[:8]}"
    )
    paused = False
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

            # Drain human gate before processing more events so we pause
            # inside the 60s approval / question TTL.
            if not human_gate.empty():
                kind, payload = human_gate.get_nowait()
                if kind == "question":
                    _pause_for_kimi_questions(state, payload)
                else:
                    _pause_for_kimi_approval(state, payload)
                paused = True
                break

            if ev.kind == "subscribed":
                if submit_text is not None and not prompt_submitted:
                    prompt_submitted = True
                    pin = _language_pin(str(state.get("user_lang") or ""))
                    asyncio.create_task(
                        client.submit_prompt(kimi_sid, pin + submit_text)
                    )
                if after_subscribe is not None and not after_subscribe_fired:
                    after_subscribe_fired = True
                    # Await so failures can restore waiting gates and surface
                    # an SSE error (fire-and-forget previously swallowed ACK errors).
                    try:
                        await after_subscribe()
                        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                    except Exception as exc:
                        _logger.exception("after_subscribe failed")
                        await _session_store.save(session_id)
                        yield _make_error_evt(str(exc) or "Failed to resume Agent turn.")
                        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
                        yield "event: done\ndata: {}\n\n"
                        return
                continue

            raw_type = (ev.raw_type or "").lower()
            status = str(ev.raw.get("status") or "")
            if (
                status in ("awaiting_approval", "awaiting_question")
                or ev.kind == "tool_call_start"
                or "approval" in raw_type
                or "question" in raw_type
            ):
                asyncio.create_task(
                    _drain_approvals_and_questions(
                        client, state, kimi_sid,
                        agent_session_dir=agent_session_dir,
                        mode=mode,
                        human_gate=human_gate,
                    )
                )

            for frame in _apply_event(state, ev):
                yield frame
            if ev.kind == "turn_ended" or ev.kind == "error":
                break
    except Exception as exc:  # noqa: BLE001
        _logger.exception("kimi WS stream crashed")
        yield _make_error_evt(f"kimi stream error: {exc}")
        state["status"] = "error"
    finally:
        poll_stop.set()
        poll_task.cancel()
        try:
            await poll_task
        except (asyncio.CancelledError, Exception):
            pass

    # Catch a late question/approval that arrived with the final status beat.
    if (
        not paused
        and state.get("status") not in ("stopped", "error")
    ):
        try:
            await _drain_approvals_and_questions(
                client, state, kimi_sid,
                agent_session_dir=agent_session_dir,
                mode=mode,
                human_gate=human_gate,
            )
            if not human_gate.empty():
                kind, payload = human_gate.get_nowait()
                if kind == "question":
                    _pause_for_kimi_questions(state, payload)
                else:
                    _pause_for_kimi_approval(state, payload)
                paused = True
        except Exception:
            _logger.debug("final interaction drain failed", exc_info=True)

    if paused or state.get("status") in (
        "waiting_for_kimi_question",
        "waiting_for_kimi_approval",
        # Legacy sessions paused before Agent/Expert status split.
        "waiting_for_clarification",
    ):
        await _session_store.save(session_id)
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    if state.get("status") == "stopped":
        await _session_store.save(session_id)
        yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    # Completed turn bookkeeping
    try:
        final = ""
        for item in reversed(state.get("history", [])):
            if item.get("role") == "assistant" and item.get("kind") != "thinking":
                final = item.get("content", "") or ""
                break
        if final:
            state.setdefault("conversation_log", []).append({
                "role": "assistant", "content": final,
                "timestamp": datetime.now().isoformat(),
            })
            memory = state.get("memory")
            display = str(state.get("last_user_text") or "")
            if memory is not None and display:
                try:
                    memory.save_context({"input": display}, {"output": final})
                except Exception:
                    pass
    except Exception:
        _logger.exception("kimi stream: finalization bookkeeping failed")

    await _session_store.save(session_id)
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
    yield "event: done\ndata: {}\n\n"


async def _stream_kimi(
    state: dict[str, Any],
    text: str,
    attachment_paths: list[str],
) -> AsyncIterator[str]:
    """Stream a single user turn through kimi-code."""
    session_id = state["session_id"]
    state["engine"] = "kimi-code"
    state["chat_mode"] = "science_agent"

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
    state["engine"] = "kimi-code"
    state["chat_mode"] = "science_agent"
    # Agent path is kimi-code only — never inherit Expert PI/CB checkpoints.
    clear_expert_gates(state)
    state["waiting_for"] = ""
    state["clarification_questions"] = []
    state["clarification_answers"] = []
    state.pop("kimi_pending_question", None)
    state.pop("kimi_pending_approval", None)
    state.pop("approval_prompt", None)
    state.pop("plan_markdown", None)
    # Surface a Thinking placeholder immediately so the UI doesn't sit on a
    # blank assistant bubble while the pool/WS/model warm up (can be seconds).
    state.setdefault("history", []).append({
        "role": "assistant",
        "content": "",
        "kind": "thinking",
        "phase": "thinking",
    })
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

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
        if state.get("_pool_scope") and state.get("_pool_scope") != inst.scope_name:
            state["kimi_session_id"] = ""
        state["_pool_scope"] = inst.scope_name
        client_base_url = inst.base_url
    else:
        client_base_url = kimi_base_url()

    client = KimiClient(base_url=client_base_url)
    try:
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

        async for frame in _kimi_event_loop(
            client, state, kimi_sid, submit_text=display_text,
        ):
            yield frame
    finally:
        await client.aclose()


async def _stream_kimi_resume(
    state: dict[str, Any],
    *,
    after_subscribe=None,
) -> AsyncIterator[str]:
    """Resume a kimi turn after AskUserQuestion / ExitPlanMode was answered.

    Does NOT submit a new prompt — kimi continues the in-flight turn once the
    pending question/approval is resolved. Prefer passing that resolve step as
    `after_subscribe` so the WS is listening before the REST decision lands.
    """
    session_id = state["session_id"]
    state["engine"] = "kimi-code"
    state["chat_mode"] = "science_agent"
    # Do NOT clear waiting_for / kimi_pending_* here — after_subscribe clears
    # them only after the kimi API ACK. Clearing early made failed Approve/
    # AskUser submissions look successful and left the turn stuck.
    if not state.get("status") or state.get("status") in (
        "waiting_for_kimi_question",
        "waiting_for_kimi_approval",
        "waiting_for_clarification",  # legacy
    ):
        state["status"] = "started"
    yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"

    client: KimiClient | None = None
    try:
        try:
            client, kimi_sid = await acquire_kimi_client_for_state(state)
        except (PoolExhaustedError, SpawnError, KimiAPIError) as exc:
            yield _make_error_evt(f"Failed to resume kimi session: {exc}")
            state["status"] = "error"
            await _session_store.save(session_id)
            yield f"event: state\ndata: {_to_json(_snapshot(state))}\n\n"
            yield "event: done\ndata: {}\n\n"
            return

        async for frame in _kimi_event_loop(
            client, state, kimi_sid,
            submit_text=None,
            after_subscribe=after_subscribe,
        ):
            yield frame
    finally:
        if client is not None:
            await client.aclose()
