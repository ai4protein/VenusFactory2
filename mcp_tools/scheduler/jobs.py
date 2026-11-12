"""Durable tool submissions. Arguments/results are private; queue rows hold metadata."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from backend.web_v2.job_queue import JobQueue, TERMINAL, launch
from backend.web_v2.job_records import JobRecords
from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
from mcp_tools.scheduler.catalog import classify, native_tools
from mcp_tools.common.tool_arguments import validate_tool_arguments
from mcp_tools.scheduler.receipt_binding import value_digest

MAX_PAYLOAD = 4_000_000
MAX_RESULT = 16_000_000


def save_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    raw=json.dumps(value,ensure_ascii=False,default=str).encode()
    if len(raw)>MAX_RESULT:raise ValueError('Tool result exceeds the size limit')
    tmp=path.with_suffix('.tmp');tmp.write_bytes(raw);tmp.chmod(0o600);os.replace(tmp,path)


class ToolJobs:
    def __init__(self, queue=None):
        self.queue=queue or JobQueue()
        self.records=JobRecords(self.queue)

    def submit(self, name, arguments, *, owner, route='native', session_id=None, parent_job=None):
        assert_tool_call_allowed(name, arguments)
        if route not in {'native','mcp','public_service'} or not isinstance(arguments,dict):raise ValueError('Invalid tool submission')
        if len(json.dumps(arguments,ensure_ascii=False).encode())>MAX_PAYLOAD:raise ValueError('Tool input is too large')
        if not owner or len(owner)>200:raise ValueError('Missing task owner')
        tools=native_tools()
        if route=='native':
            if name not in tools:raise ValueError('Unknown registered tool')
            validate_tool_arguments(tools[name], arguments)
        if route=='public_service':
            from mcp_tools.tools.research.services import SERVICE_IDS
            if name != 'query_public_protein_service' or arguments.get('service') not in SERVICE_IDS:
                raise ValueError('Unknown protein service')
        policy=classify(name,getattr(tools.get(name),'metadata',None))
        if parent_job:
            parent=self.queue.get(parent_job,owner)
            if parent['status'] in TERMINAL or parent['cancel_requested']:raise ValueError('Parent job has stopped')
        workspace = None
        if parent_job and parent['kind'] == 'scientist':
            meta = parent['metadata']
            folder = Path(meta['goal_folder'])
            session = meta.get('session_id')
            if (not isinstance(session, str) or not session
                    or folder.name != meta.get('goal_id')
                    or folder.parent.name != hashlib.sha256(session.encode()).hexdigest()
                    or folder.is_symlink() or folder.parent.is_symlink()
                    or (folder / 'workspace').is_symlink()):
                raise ValueError('Invalid research workspace')
            workspace = str((folder / 'workspace').resolve(strict=True))
        spec_id=uuid.uuid4().hex
        spec=self.queue.root/'mcp-inputs'/(spec_id+'.json')
        save_json(spec,{'name':name,'arguments':arguments,'route':route,'session_id':session_id,'workspace':workspace})
        try:
            identity,_=launch([sys.executable,'-u','-m','mcp_tools.scheduler.worker',str(spec)],owner=owner,kind='mcp:'+policy['category'],
                              resource=policy['resource'],timeout=policy['timeout_seconds'],queue=self.queue,
                               metadata={'tool':name,'category':policy['category'],'route':route,'parent_job':parent_job,'session_id':session_id,'arguments_sha256':value_digest(arguments)})
        except BaseException:
            spec.unlink(missing_ok=True);raise
        self.records.register('mcp',identity,owner,'mcp:'+policy['category'],name,self.queue.get(identity,owner)['created'])
        return self.get(identity,owner)

    def get(self, identity, owner):
        if identity.startswith('VF-'):identity=self.records.resolve('mcp',identity,owner)
        row=self.queue.get(identity,owner)
        if not row['kind'].startswith('mcp:'):raise FileNotFoundError('Task not found')
        public=self.queue.public(identity,owner)
        public.update(self.records.register('mcp',identity,owner,row['kind'],row['metadata']['tool'],row['created']))
        public.update(tool=row['metadata']['tool'],category=row['metadata']['category'])
        public.update({key:row['metadata'][key] for key in ('arguments_sha256','parent_job') if key in row['metadata']})
        result=self.queue.files(identity)/'mcp-result.json'
        progress=self.queue.files(identity)/'mcp-progress.json'
        if progress.is_file():public['progress']=json.loads(progress.read_text())
        # The worker can finish between the metadata read and public(). Bind
        # result delivery to the status actually returned to the caller.
        if public['status'] in TERMINAL and result.is_file():
            public['result']=json.loads(result.read_text())
        elif public['status'] == 'completed':
            # Never manufacture success or rerun a completed computation when
            # its worker failed to persist a usable result.
            public['result']={'is_error':True,'message':'Completed tool task has no stored result'}
        return public

    def list(self, owner):
        rows=[]
        for row in self.queue.list(owner):
            if row['kind'].startswith('mcp:'):
                item=self.get(row['id'],owner);item.pop('result',None);rows.append(item)
        return rows

    def cancel(self, identity, owner):
        job=self.get(identity,owner)
        self.queue.cancel(job['id'],owner)
        return self.get(job['id'],owner)

    async def wait(self, identity, owner, on_status=None):
        previous=None
        try:
            while True:
                job=await asyncio.to_thread(self.get,identity,owner)
                state=(job['status'],job['position'])
                if state!=previous and on_status:on_status({k:v for k,v in job.items() if k!='result'})
                previous=state
                if job['status'] in TERMINAL:return job
                await asyncio.sleep(.3)
        except asyncio.CancelledError:
            await asyncio.to_thread(self.cancel,identity,owner)
            raise
