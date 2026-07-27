"""Interactive checkpoint endpoints (clarification, plan, iteration, step, sub-report)."""
import asyncio

from fastapi import APIRouter, HTTPException, Request

from agent.chat_graph import _format_clarification_answers
from logger import get_logger

from web_v2.chat_api._hooks_runtime import (
    _get_lock,
    _session_store,
    _set_cancel,
)
from web_v2.chat_api._models import (
    ApprovalDecideRequest,
    ClarificationResponseRequest,
    IterationDecideRequest,
    PlanConfirmRequest,
    StepDecideRequest,
    SubReportDecideRequest,
)
from web_v2.chat_api._shared import (
    _append_dialogue_memory,
    _assert_session_access,
    _get_session_or_404,
    _is_agent_session,
    _is_expert_session,
    _is_zh_text,
    _record_access_event,
    _should_skip_research,
    _sse_response,
)
from web_v2.chat_api._uploads import _archive_conversation
from web_v2.chat_api._stream_kimi import (
    _stream_kimi_resume,
    acquire_kimi_client_for_state,
    ui_answers_to_kimi,
)
from web_v2.chat_api._stream_resume import _stream_graph_resume

router = APIRouter()
_logger = get_logger("chat_api.interactive")


_APPROVE_LABELS = frozenset({
    "approve", "批准", "确认", "yes", "ok", "allow", "允许",
})


def _require_expert_session(state: dict) -> None:
    if _is_agent_session(state):
        raise HTTPException(
            status_code=400,
            detail="This checkpoint belongs to Science Expert (LangGraph), not Science Agent.",
        )


async def _resolve_kimi_question_and_resume(
    state: dict,
    answers_data: list[dict],
):
    """Resume WS first, then POST structured answers to kimi /questions."""
    rev = dict(state.get("kimi_pending_question") or {})
    question_id = str(rev.get("question_id") or "")
    if not question_id:
        raise HTTPException(status_code=400, detail="Missing kimi pending question id.")
    kimi_answers = ui_answers_to_kimi(answers_data, rev)

    questions = list(state.get("clarification_questions", []) or [])
    answers_summary = _format_clarification_answers(questions, answers_data)
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    state["history"].append({
        "role": "user",
        "content": ("📝 **回答 Agent：**\n\n" if is_zh else "📝 **Answered Agent:**\n\n") + answers_summary,
    })
    state["clarification_answers"] = answers_data
    # Keep kimi_pending_question until the kimi API ACK succeeds (see after_subscribe).

    async def after_subscribe() -> None:
        client = None
        try:
            client, kimi_sid = await acquire_kimi_client_for_state(state)
            await client.answer_question(kimi_sid, question_id, answers=kimi_answers)
            state.pop("kimi_pending_question", None)
            state["waiting_for"] = ""
            state["clarification_questions"] = []
            state["status"] = "chat_mode"
        except Exception as exc:
            _logger.exception("kimi answer_question failed after subscribe")
            state["kimi_pending_question"] = rev
            state["waiting_for"] = "kimi_question"
            state["clarification_questions"] = questions
            state["status"] = "waiting_for_kimi_question"
            raise RuntimeError(f"Failed to submit AskUser answers: {exc}") from exc
        finally:
            if client is not None:
                await client.aclose()

    async for chunk in _stream_kimi_resume(state, after_subscribe=after_subscribe):
        yield chunk


async def _resolve_kimi_approval_decision_and_resume(
    state: dict,
    *,
    approved: bool,
    selected_label: str = "",
    feedback: str = "",
):
    """Resume WS first, then POST an explicit approve/reject to kimi."""
    pending = dict(state.get("kimi_pending_approval") or {})
    approval_id = str(pending.get("approval_id") or "")
    if not approval_id:
        raise HTTPException(status_code=400, detail="Missing kimi pending approval id.")

    tool = str(pending.get("tool_name") or "")
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    label = (selected_label or "").strip()
    saved_prompt = state.get("approval_prompt")
    saved_plan = state.get("plan_markdown")
    saved_questions = list(state.get("clarification_questions", []) or [])
    if approved:
        summary = label if label and label.lower() not in _APPROVE_LABELS else (
            "批准" if is_zh else "Approved"
        )
        content = (
            f"✅ **已批准**{f' `{tool}`' if tool else ''}：{summary}"
            if is_zh
            else f"✅ **Approved**{f' `{tool}`' if tool else ''}: {summary}"
        )
    else:
        reason = (feedback or "").strip()
        base = f"❌ **已拒绝**{f' `{tool}`' if tool else ''}" if is_zh else f"❌ **Rejected**{f' `{tool}`' if tool else ''}"
        content = f"{base} — {reason}" if reason else base
    state["history"].append({"role": "user", "content": content})
    state["clarification_answers"] = []
    # Keep kimi_pending_approval until the kimi API ACK succeeds.

    reject_feedback = (feedback or "").strip() or "Rejected by user via VenusFactory UI"

    async def after_subscribe() -> None:
        client = None
        try:
            client, kimi_sid = await acquire_kimi_client_for_state(state)
            if approved:
                kwargs: dict = {"decision": "approved", "scope": "session"}
                if label and label.lower() not in _APPROVE_LABELS:
                    kwargs["selected_label"] = label
                await client.decide_approval(kimi_sid, approval_id, **kwargs)
            else:
                await client.decide_approval(
                    kimi_sid, approval_id,
                    decision="rejected",
                    feedback=reject_feedback,
                )
            state.pop("kimi_pending_approval", None)
            state.pop("approval_prompt", None)
            state.pop("plan_markdown", None)
            state["waiting_for"] = ""
            state["clarification_questions"] = []
            state["status"] = "chat_mode"
        except Exception as exc:
            _logger.exception("kimi approval decision failed after subscribe")
            state["kimi_pending_approval"] = pending
            if saved_prompt is not None:
                state["approval_prompt"] = saved_prompt
            if saved_plan is not None:
                state["plan_markdown"] = saved_plan
            state["waiting_for"] = "kimi_approval"
            state["clarification_questions"] = saved_questions
            state["status"] = "waiting_for_kimi_approval"
            raise RuntimeError(f"Failed to submit approval decision: {exc}") from exc
        finally:
            if client is not None:
                await client.aclose()

    async for chunk in _stream_kimi_resume(state, after_subscribe=after_subscribe):
        yield chunk


@router.post("/sessions/{session_id}/clarification/respond/stream")
async def stream_clarification_response(
    session_id: str,
    payload: ClarificationResponseRequest,
    request: Request,
):
    """Science Expert only — PI clarification → LangGraph resume.

    Science Agent AskUser/Approve must use ``/ask-user`` / ``/approval`` —
    this endpoint never routes into kimi-code.
    """
    _record_access_event(request, "/api/chat/sessions/{id}/clarification/respond/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if _is_agent_session(state) or state.get("waiting_for") in ("kimi_question", "kimi_approval"):
        raise HTTPException(
            status_code=400,
            detail="Science Agent interactive gates use /ask-user or /approval, not /clarification.",
        )
    waiting = state.get("waiting_for")
    if waiting != "clarification" and state.get("status") != "waiting_for_clarification":
        raise HTTPException(status_code=400, detail="Session is not waiting for Expert clarification.")

    answers_data = [a.model_dump() for a in payload.answers]
    lock = await _get_lock(session_id)

    questions = state.get("clarification_questions", [])
    state["clarification_answers"] = answers_data
    state["engine"] = "graph"
    state["chat_mode"] = "science_expert"

    answers_summary = _format_clarification_answers(questions, answers_data)
    is_zh = _is_zh_text(state.get("last_user_text", ""))
    state["history"].append({
        "role": "user",
        "content": ("📝 **需求补充说明：**\n\n" if is_zh else "📝 **Clarification Details:**\n\n") + answers_summary,
    })

    original_text = state.get("last_user_text", "")
    enriched_text = f"{original_text}\n\n[Clarification Details]\n{answers_summary}"
    state["last_user_text"] = enriched_text
    skip_research = _should_skip_research(questions, answers_data)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for="skip_to_plan" if skip_research else "clarification_answered",
            ):
                yield chunk

    return _sse_response(event_gen())


@router.post("/sessions/{session_id}/ask-user/respond/stream")
async def stream_ask_user_response(
    session_id: str,
    payload: ClarificationResponseRequest,
    request: Request,
):
    """Science Agent only — kimi AskUserQuestion respond."""
    _record_access_event(request, "/api/chat/sessions/{id}/ask-user/respond/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if _is_expert_session(state):
        raise HTTPException(
            status_code=400,
            detail="Science Expert clarification uses /clarification, not /ask-user.",
        )
    waiting = state.get("waiting_for")
    status = state.get("status")
    if waiting != "kimi_question" and status not in (
        "waiting_for_kimi_question",
        "waiting_for_clarification",  # legacy Agent pause
    ) and not state.get("kimi_pending_question"):
        raise HTTPException(status_code=400, detail="Session is not waiting for AskUser answers.")

    answers_data = [a.model_dump() for a in payload.answers]
    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _resolve_kimi_question_and_resume(state, answers_data):
                yield chunk

    return _sse_response(event_gen())


@router.post("/sessions/{session_id}/approval/decide/stream")
async def stream_approval_decide(
    session_id: str,
    payload: ApprovalDecideRequest,
    request: Request,
):
    """Science Agent only — kimi Approve/Reject."""
    _record_access_event(request, "/api/chat/sessions/{id}/approval/decide/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)

    if _is_expert_session(state):
        raise HTTPException(
            status_code=400,
            detail="Science Expert does not use Agent approval gates.",
        )
    waiting = state.get("waiting_for")
    status = state.get("status")
    if waiting != "kimi_approval" and status not in (
        "waiting_for_kimi_approval",
        "waiting_for_clarification",  # legacy Agent pause
    ) and not state.get("kimi_pending_approval"):
        raise HTTPException(status_code=400, detail="Session is not waiting for approval.")

    decision = (payload.decision or "").strip().lower()
    if decision not in ("approved", "rejected", "approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be approved or rejected.")
    approved = decision in ("approved", "approve")
    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _resolve_kimi_approval_decision_and_resume(
                state,
                approved=approved,
                selected_label=payload.selected_label or "",
                feedback=payload.feedback or "",
            ):
                yield chunk

    return _sse_response(event_gen())


@router.post("/sessions/{session_id}/plan/confirm/stream")
async def stream_plan_confirmation(
    session_id: str,
    payload: PlanConfirmRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/plan/confirm/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)
    _require_expert_session(state)

    if state.get("waiting_for") != "plan_confirmation" and state.get("status") != "waiting_for_plan_confirmation":
        raise HTTPException(status_code=400, detail="Session is not waiting for plan confirmation.")

    confirmed_plan = payload.plan
    auto_execute = payload.auto_execute
    state["plan"] = confirmed_plan
    state["auto_execute"] = auto_execute

    is_zh = _is_zh_text(state.get("last_user_text", ""))
    if auto_execute:
        state["history"].append({
            "role": "user",
            "content": "✅ 已确认执行计划，自动执行所有步骤。" if is_zh else "✅ Plan confirmed. Auto-executing all steps.",
        })
    else:
        state["history"].append({
            "role": "user",
            "content": "✅ 已确认执行计划。" if is_zh else "✅ Plan confirmed.",
        })

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for="plan_confirmed",
                extra_state={
                    "plan": confirmed_plan,
                    "current_step_index": 0,
                    "step_results": {},
                    "auto_execute": auto_execute,
                },
            ):
                yield chunk

    return _sse_response(event_gen())


@router.post("/sessions/{session_id}/iteration/decide")
async def iteration_decide(
    session_id: str,
    payload: IterationDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/iteration/decide")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    _require_expert_session(state)

    if state.get("waiting_for") != "iteration" and state.get("status") != "waiting_for_iteration":
        raise HTTPException(status_code=400, detail="Session is not waiting for iteration decision.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "modify_plan":
        state["status"] = "waiting_for_plan_confirmation"
        state["waiting_for"] = "plan_confirmation"
        state["history"].append({
            "role": "user",
            "content": "🔄 希望修改计划并重新执行。" if is_zh else "🔄 I'd like to modify the plan and re-execute.",
        })
        await _session_store.save(session_id)
        return {
            "success": True,
            "status": "waiting_for_plan_confirmation",
            "plan": list(state.get("plan", [])),
        }

    if action == "continue":
        user_msg = "➕ 继续分析，我有新的指令。" if is_zh else "➕ Continue analysis with new instructions."
        state["has_prior_research"] = True
    else:
        user_msg = "✅ 对结果满意，任务完成。" if is_zh else "✅ Satisfied with the results. Task complete."
        state["has_prior_research"] = False

    state["status"] = "completed"
    state["waiting_for"] = None
    state["history"].append({"role": "user", "content": user_msg})

    original_text = state.get("last_user_text", "")
    final_content = ""
    skip_markers = ("iteration_prompt", "请选择下一步", "Please choose")
    for item in reversed(state.get("history", [])):
        if item.get("role") == "assistant" and item.get("role_id") == "principal_investigator":
            content = item.get("content", "")
            if not any(m in content for m in skip_markers) and len(content) > 10:
                final_content = content
                break
    _append_dialogue_memory(state, original_text, final_content)
    try:
        state["memory"].save_context({"input": original_text}, {"output": final_content})
    except Exception:
        pass
    await _session_store.save(session_id)
    asyncio.create_task(_archive_conversation(state))
    return {"success": True, "status": "completed"}


@router.post("/sessions/{session_id}/step/decide/stream")
async def stream_step_decide(
    session_id: str,
    payload: StepDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/step/decide/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)
    _require_expert_session(state)

    if state.get("waiting_for") != "step_review" and state.get("status") != "waiting_for_step_review":
        raise HTTPException(status_code=400, detail="Session is not waiting for step review.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "abort":
        state["history"].append({
            "role": "user",
            "content": "⏹️ 跳过剩余步骤，直接汇总。" if is_zh else "⏹️ Skip remaining steps and go to summary.",
        })
        waiting_for = "step_abort"
    else:
        state["history"].append({
            "role": "user",
            "content": "▶️ 继续执行下一步。" if is_zh else "▶️ Continue to the next step.",
        })
        waiting_for = "step_continue"

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for=waiting_for,
            ):
                yield chunk

    return _sse_response(event_gen())


@router.post("/sessions/{session_id}/sub-report/decide/stream")
async def stream_sub_report_decide(
    session_id: str,
    payload: SubReportDecideRequest,
    request: Request,
):
    _record_access_event(request, "/api/chat/sessions/{id}/sub-report/decide/stream")
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    await _set_cancel(session_id, False)
    _require_expert_session(state)

    if state.get("waiting_for") != "sub_report_review" and state.get("status") != "waiting_for_sub_report_review":
        raise HTTPException(status_code=400, detail="Session is not waiting for sub-report review.")

    action = payload.action
    is_zh = _is_zh_text(state.get("last_user_text", ""))

    if action == "rewrite":
        comment_text = (payload.comment or "").strip()
        if not comment_text:
            raise HTTPException(status_code=400, detail="Comment is required for rewrite action.")
        current_idx = state.get("research_idx", 1)
        state["research_idx"] = max(0, current_idx - 1)
        sub_reports = list(state.get("research_sub_reports", []))
        if sub_reports:
            sub_reports.pop()
        state["research_sub_reports"] = sub_reports
        state["sub_report_rewrite_comment"] = comment_text
        state["history"].append({
            "role": "user",
            "content": f"✏️ 修改意见：{comment_text}" if is_zh else f"✏️ Revision feedback: {comment_text}",
        })
        waiting_for = "sub_report_rewrite"
    elif action == "skip":
        state["history"].append({
            "role": "user",
            "content": "⏭️ 跳过剩余小节，直接生成报告。" if is_zh else "⏭️ Skip remaining sections and generate report.",
        })
        waiting_for = "sub_report_skip"
    else:
        state["history"].append({
            "role": "user",
            "content": "▶️ 继续调研下一个小节。" if is_zh else "▶️ Continue to the next section.",
        })
        waiting_for = "sub_report_continue"

    extra = {}
    if action == "rewrite":
        extra = {
            "sub_report_rewrite_comment": comment_text,
            "research_idx": state["research_idx"],
            "research_sub_reports": list(state.get("research_sub_reports", [])),
        }

    lock = await _get_lock(session_id)

    async def event_gen():
        async with lock:
            async for chunk in _stream_graph_resume(
                state,
                waiting_for=waiting_for,
                extra_state=extra if extra else None,
            ):
                yield chunk

    return _sse_response(event_gen())
