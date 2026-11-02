"""Compact progress events; large artifacts stay in their owner-scoped storage."""
import json
import os
from .job_queue import JobQueue

def publish_progress(state):
    identity=os.getenv('VENUS_QUEUE_JOB_ID');lease=os.getenv('VENUS_QUEUE_LEASE_TOKEN')
    if not identity or not lease:return
    data={key:state[key] for key in ('status','phase','steps','plan','chapters','activity','model_calls','minimum_met') if key in state}
    # Chapter bodies belong to report artifacts, not repeated progress events.
    if 'chapters' in data:data['chapters']=[{k:v for k,v in ch.items() if k not in {'draft','content','text'}} for ch in data['chapters']]
    JobQueue().append_frame(identity,'event: progress\ndata: '+json.dumps(data,ensure_ascii=False,default=str)+'\n\n',lease)
