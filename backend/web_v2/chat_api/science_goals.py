"""Authenticated session controls for persistent expert-team goals."""

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response, UploadFile, File
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from backend.web_v2.chat_api._hooks_runtime import _get_lock, _session_store
from backend.web_v2.chat_api._shared import _assert_session_access, _get_session_or_404, _redact_obj, _runtime_mode
from backend.web_v2.science_goals import GoalConflict, service
from backend.web_v2.online_access import SCIENCE_LIMITS, AccessStore, identities, admit
from backend.web_v2.job_access import require_same_origin

router = APIRouter()


class GoalBudget(BaseModel):
    # Transport ceiling; ScienceGoalService additionally enforces each driver's limits.
    model_config = ConfigDict(extra="forbid")
    max_cycles: StrictInt = Field(default=4, ge=1, le=32)
    model_calls: StrictInt = Field(default=72, ge=1, le=768)
    tool_calls: StrictInt = Field(default=36, ge=1, le=384)
    wall_seconds: StrictInt = Field(default=2400, ge=1, le=28 * 86400)


class ExecutionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    harness: str = Field(min_length=1, max_length=80)
    model_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,80}$")


class CreateGoal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{8,80}$")
    driver: str = "autonomous_research.v1"
    execution: ExecutionSelection | None = None
    parameters: dict[str, str] = Field(default_factory=dict)
    objective: str = Field(min_length=5, max_length=8000)
    budget: GoalBudget = Field(default_factory=GoalBudget)
    attachments: list[str] = Field(default_factory=list, max_length=32)


async def access(session_id, request):
    state = await _get_session_or_404(session_id)
    _assert_session_access(state, request)
    require_same_origin(request)
    return state


async def call(fn, *args):
    try:
        return _redact_obj(await asyncio.to_thread(fn, *args))
    except GoalConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, "Goal or configured research resource not found.") from exc
    except ValueError as exc:
        raise HTTPException(400, "Invalid goal configuration or changed evidence.") from exc
    except RuntimeError as exc:
        raise HTTPException(503, "Goal could not start; diagnostics retained on the research host.") from exc


@router.get("/science-goals/catalog")
async def catalog(response: Response, view: Literal["full", "composer"] = "full"):
    response.headers["Cache-Control"] = "no-store"
    data = await call(service.catalog) if view == "full" else await call(service.catalog, False)
    if _runtime_mode() != "local":
        data.update(default_budget=SCIENCE_LIMITS, maximum_budget=SCIENCE_LIMITS, online_limited=True)
        data["drivers"] = [d for d in data["drivers"] if d["id"] == "autonomous_research.v1"]
        data["available"] = bool(data["available"] and data["drivers"])
        for d in data["drivers"]: d["maximum_budget"] = SCIENCE_LIMITS
    return data


@router.get("/sessions/{session_id}/science-goals")
async def list_goals(session_id: str, request: Request, response: Response):
    await access(session_id, request)
    response.headers["Cache-Control"] = "no-store"
    goals = await call(service.list, session_id)
    if _runtime_mode() != "local" and goals and all(g["status"] in {"completed","cancelled","budget_exhausted","failed"} and not g["worker_alive"] for g in goals):
        AccessStore().release_science("chat:" + session_id)
    return {"goals": goals}


@router.get("/sessions/{session_id}/science-goals/generated-steps")
async def generated_steps(session_id: str, case: str, request: Request, response: Response):
    await access(session_id, request)
    response.headers["Cache-Control"] = "no-store"
    return {"entries": await call(service.generated_steps, session_id, case)}


@router.post("/sessions/{session_id}/science-goals/uploads")
async def upload_research_inputs(session_id: str, request: Request, files: list[UploadFile] = File(...)):
    await access(session_id, request)
    from backend.web_v2.research_files import save_inputs, MAX_BATCH
    from backend.web_v2.upload_limits import read_upload, preflight_uploads, upload_limit
    try:
        if not 1 <= len(files) <= 32:
            raise HTTPException(400, "Select up to 32 files")
        preflight_uploads(files)
        batch = []; total = 0
        for file in files:
            data = await read_upload(file); total += len(data)
            if upload_limit() is not None and total > MAX_BATCH:
                raise HTTPException(413, "Upload up to 80 MB per batch")
            batch.append((file.filename or "", data))
        return {"files": await call(save_inputs, service.session_folder(session_id), batch)}
    finally:
        for file in files:
            await file.close()


@router.post("/sessions/{session_id}/science-goals")
async def create_goal(session_id: str, payload: CreateGoal, request: Request):
    await access(session_id, request)
    if _runtime_mode() != "local":
        if payload.driver != "autonomous_research.v1" or payload.parameters:
            raise HTTPException(400, "Online Scientist supports autonomous research only")
        if any(payload.budget.model_dump()[k] > v for k,v in SCIENCE_LIMITS.items()):
            raise HTTPException(400, "Online Scientist allows 2 iterations and 15 minutes")
    lock = await _get_lock(session_id)
    if lock.locked():
        raise HTTPException(409, "This session is running a conversation turn.")
    async with lock:
        state = await access(session_id, request)
        from functools import partial
        def before_start():
            if _runtime_mode() == "local": return
            store = AccessStore()
            scope = "chat:" + session_id
            store.science_slot(scope, *identities(request))
            try: admit(request, scope, "scientist")
            except BaseException:
                store.release_science(scope)
                raise
        creator = service.create if _runtime_mode() == "local" else partial(
            service.create, before_start=before_start, online_owner=identities(request)[0])
        result = await call(
            creator,
            session_id,
            payload.model_dump(exclude={"request_id"}, exclude_none=True),
            payload.request_id,
        )
        state["chat_mode"] = "science_expert"
        # Persist a human-readable goal in session history so it can be reopened.
        history = state.setdefault("history", [])
        if not any(x.get("science_goal_id") == result["id"] for x in history if isinstance(x, dict)):
            history.append({"role": "user", "content": payload.objective, "science_goal_id": result["id"]})
        await _session_store.save(session_id)
        return result


from agent.harness.interactions import InteractionResponse


class ResearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    interaction: InteractionResponse | None = None
    text: str = Field(default="", max_length=8000)
    plan: dict | None = None
    attachments: list[str] = Field(default_factory=list, max_length=32)


@router.post("/sessions/{session_id}/science-goals/{goal_id}/input")
async def provide_input(session_id: str, goal_id: str, payload: ResearchInput, request: Request):
    await access(session_id, request)
    lock = await _get_lock(session_id)
    if lock.locked():
        raise HTTPException(409, "This session is busy.")
    async with lock:
        return await call(service.provide_input, session_id, goal_id, payload.text, payload.plan, payload.attachments, payload.interaction.model_dump() if payload.interaction else None)


@router.post("/sessions/{session_id}/science-goals/{goal_id}/{action}")
async def control_goal(
    session_id: str, goal_id: str, action: Literal["pause", "resume", "cancel", "publish"], request: Request
):
    await access(session_id, request)
    lock = await _get_lock(session_id)
    if lock.locked():
        raise HTTPException(409, "This session is busy; retry after the current operation.")
    async with lock:
        await access(session_id, request)
        if _runtime_mode() != "local" and action == "resume":
            goal = await call(service.public, session_id, goal_id)
            if goal["driver"] != "autonomous_research.v1" or any(goal["budget"].get(k, v+1) > v for k,v in SCIENCE_LIMITS.items()):
                raise HTTPException(403, "This study exceeds online run limits")
            AccessStore().science_slot("chat:" + session_id, *identities(request))
        return await call(service.control, session_id, goal_id, action)


@router.get("/sessions/{session_id}/science-goals/{goal_id}/project")
async def download_project(session_id: str, goal_id: str, request: Request, format: Literal["zip", "report"] = "zip", include_working_files: bool = False):
    await access(session_id, request)
    from agent.science_goal import GoalStore
    from backend.web_v2.research_delivery import export_project, report
    from starlette.responses import StreamingResponse
    from utils.scoped_files import chunks
    try:
        folder, _, _ = service.metadata(session_id, goal_id)
        if format == "zip" and service.live(folder):
            raise HTTPException(409, "Pause the research before exporting its project")
        state = GoalStore(folder).read()
        headers = {"Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="research-{goal_id}.{ "zip" if format == "zip" else "md" }"'}
        if format == "report":
            return Response(report(state), media_type="text/markdown", headers=headers)
        archive = await asyncio.to_thread(export_project, folder, state, include_working_files=include_working_files)
        return StreamingResponse(chunks(archive), media_type="application/zip", headers=headers)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Research project not found") from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(409, "Project files changed or are unavailable; retry after research stops") from exc
