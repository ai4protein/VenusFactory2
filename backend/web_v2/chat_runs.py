"""Agent turns are durable queue jobs; HTTP only submits and subscribes."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import uuid
from fastapi import HTTPException
from starlette.responses import StreamingResponse
from .job_queue import JobQueue, TERMINAL, launch
from .task_store import SubmissionConflict


def request_identity(request):
    key=(request.headers.get('idempotency-key') or '').strip() or uuid.uuid4().hex
    if not re.fullmatch(r'[a-zA-Z0-9_-]{8,120}',key):raise HTTPException(422,'Invalid request identity')
    return key


def cursor_value(value, identity):
    if not value:return 0
    if ':' in value:
        run,seq=value.rsplit(':',1)
        if run!=identity:raise HTTPException(409,'Event cursor belongs to another run')
    else:seq=value
    try:return max(0,int(seq))
    except ValueError:raise HTTPException(422,'Invalid event cursor') from None


async def event_stream(q, identity, owner, after=0):
    cursor=after;heartbeat=time.monotonic()
    while True:
        rows=await asyncio.to_thread(q.events,identity,owner,cursor)
        for row in rows:
            cursor=row['seq']
            yield f'id: {identity}:{cursor}\n'+row['frame']
        state=await asyncio.to_thread(q.public,identity,owner)
        if state['status'] in TERMINAL:
            # Re-read after terminal state: final events may commit between reads.
            rows=await asyncio.to_thread(q.events,identity,owner,cursor)
            if rows:continue
            yield 'event: done\ndata: {}\n\n';return
        if time.monotonic()-heartbeat>10:
            yield ': heartbeat\n\n';heartbeat=time.monotonic()
        await asyncio.sleep(.08)


def response(q, identity, owner, sid, cursor=''):
    after=cursor_value(cursor,identity)
    return StreamingResponse(event_stream(q,identity,owner,after),media_type='text/event-stream',headers={
        'Cache-Control':'no-cache, no-transform','X-Accel-Buffering':'no',
        'X-Run-ID':identity,'X-Event-Stream':f'/api/chat/sessions/{sid}/runs/{identity}/events',
    })


async def submit(state, request, payload=None, *, retry=False, interaction=None):
    from .chat_api._hooks_runtime import _get_lock, _session_store, _set_cancel
    from .chat_api._shared import _consume_online_chat_quota_or_429, _extract_client_ip, _session_owner_key_for_request
    from .agent_execution import admit
    from .agent_context import ContextPolicy, ContextLimit, check_admission
    from .science_goals import service
    if payload and payload.chat_mode not in (None,'science_agent'):raise HTTPException(422,'Agent execution cannot be selected for Scientist mode')
    if payload and (payload.custom_model_config or payload.custom_model_id):raise HTTPException(422,'Select an admitted model identity')
    sid=state['session_id'];owner=_session_owner_key_for_request(request);q=JobQueue()
    selection=payload.execution.model_dump() if payload and payload.execution else None
    text=state.get('last_user_text','') if retry else (payload.text or '') if payload else ''
    paths=state.get('last_attachment_paths',[]) if retry else payload.attachment_paths if payload else []
    lang=(payload.lang if payload else state.get('user_lang')) or 'en'
    data={'text':text,'paths':paths,'retry':retry,'selection':selection,'lang':lang}
    fingerprint=hashlib.sha256(json.dumps(interaction.model_dump() if interaction else data,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    key=('interaction:'+sid+':'+interaction.interaction_id) if interaction else 'chat:'+sid+':'+request_identity(request)
    try:previous=await asyncio.to_thread(q.request,owner,key,fingerprint)
    except SubmissionConflict as exc:raise HTTPException(409,str(exc)) from None
    if previous:return response(q,previous['id'],owner,sid,request.headers.get('last-event-id',''))
    lock=await _get_lock(sid)
    if lock.locked():raise HTTPException(409,'This conversation already has an active turn')
    async with lock:
        # Refresh after the shared lock, including after another API process submitted.
        state=await _session_store.get(sid)
        if state is None:
            raise HTTPException(404, 'Conversation not found')
        try:
            previous=await asyncio.to_thread(q.request,owner,key,fingerprint)
            if previous:return response(q,previous['id'],owner,sid,request.headers.get('last-event-id',''))
            active=await asyncio.to_thread(q.latest_run,'chat:'+sid)
            if active and active['status'] in {'queued','running'}:raise HTTPException(409,'This conversation already has an active turn')
            if await asyncio.to_thread(service.has_running_goal,sid):raise HTTPException(409,'Pause the running Scientist goal first')
            pending = state.get('pending_interaction')
            if interaction:
                from agent.harness.interactions import response_text
                if not pending:raise HTTPException(409,'This question is no longer pending')
                try:text=response_text(pending,interaction)
                except ValueError as exc:raise HTTPException(422,str(exc)) from None
                data.update(text=text,interaction_response=interaction.model_dump())
            elif pending:raise HTTPException(409,'Respond to the pending question first')
            if not text.strip() and not paths:raise HTTPException(400,'Enter a message or attach a file')
            binding,_=admit(state,selection)
            profile=binding['profile']
            check_admission(state,text,ContextPolicy.load(),profile['model']['context_tokens'],profile['budget']['output_tokens'],retry=retry or bool(interaction),lang=lang)
            await _consume_online_chat_quota_or_429(request,sid)
            await _set_cancel(sid,False)
            state.update(agent_execution=binding,client_ip=_extract_client_ip(request),owner_key=owner,user_lang=lang,status='queued')
            if not retry:state.update(last_user_text=text,last_attachment_paths=paths)
            await _session_store.save(sid)
            identity,_=await asyncio.to_thread(launch,[sys.executable,'-u','-m','backend.web_v2.chat_runs','worker',sid],owner=owner,
                kind='agent',resource='api:agent',timeout=int(profile['budget']['wall_seconds'])+60,queue=q,
                metadata={'mode':'agent','session_id':sid,'harness':profile['harness'],'agent_input':data},session_key='chat:'+sid,request_key=key,request_hash=fingerprint)
        except ContextLimit as exc:raise HTTPException(422,str(exc)) from None
        except (ValueError,OSError,KeyError) as exc:raise HTTPException(409,str(exc)) from None
    return response(q,identity,owner,sid,request.headers.get('last-event-id',''))


async def worker(sid):
    from .chat_api._hooks_runtime import _session_store, _get_lock
    from .chat_api._stream_agent import _stream_agent
    q=JobQueue();identity=os.environ['VENUS_QUEUE_JOB_ID'];lease=os.environ['VENUS_QUEUE_LEASE_TOKEN']
    row=q.get(identity);data=row['metadata']['agent_input']
    lock=await _get_lock(sid)
    async with lock:
        state=await _session_store.get(sid)
        if not state or state.get('owner_key','local')!=row['owner']:raise RuntimeError('Conversation unavailable')
        saved=0
        state["_agent_run_id"]=identity;state["_agent_event_seq"]=0
        async for frame in _stream_agent(state,data['text'],data['paths'],retry=data['retry'],interaction_response=data.get('interaction_response')):
            if frame.startswith('event: done\n'):continue
            # Every delivered frame is committed before it becomes observable.
            state["_agent_event_seq"] = await asyncio.to_thread(q.append_frame,identity,frame,lease)
            if time.monotonic()-saved>=.25:
                await _session_store.save(sid);saved=time.monotonic()
        await _session_store.save(sid)
        return 0 if state.get('status') in {'completed','stopped','waiting_for_input'} else 1


if __name__=='__main__':sys.exit(asyncio.run(worker(sys.argv[2])))
