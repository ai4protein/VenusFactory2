"""Detached queue waiter and process-group watchdog."""
import json
import os
import signal
import subprocess
import sys
import time
from .job_queue import JobQueue, QueueCancelled, process_token, stop_process_tree, TERMINAL

def run(identity):
    q=JobQueue();q.bind(identity)
    spec=json.loads((q.files(identity)/'command.json').read_text())
    stopped=False;proc=None
    row=q.get(identity)
    parent=row['metadata'].get('parent_job')
    def parent_stopped():
        if not parent:return False
        try:
            state=q.get(parent,row['owner'])
            return state['status'] in TERMINAL or state['cancel_requested']
        except FileNotFoundError:return True
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    status,message,code='failed','Worker stopped',None
    queued_at=time.monotonic()
    try:
        while not q.claim(identity):
            wait_limit=int(os.getenv("VENUS_QUEUE_WAIT_TIMEOUT","0"))
            if wait_limit and time.monotonic()-queued_at > wait_limit:raise TimeoutError("Queue waiting limit reached")
            if stopped or parent_stopped():raise QueueCancelled('Cancelled')
            time.sleep(.2)
        if stopped or parent_stopped() or q.get(identity)['cancel_requested']:raise QueueCancelled('Cancelled')
        from mcp_tools.tools.capability_policy import CapabilityStore
        if row['metadata'].get('capability_tool'):
            CapabilityStore().assert_call(row['metadata']['capability_tool'],row['metadata'].get('capability_arguments',{}))
        proc=subprocess.Popen(spec['command'],cwd=spec['cwd'],env=os.environ.copy(),stdin=subprocess.DEVNULL,start_new_session=True,pass_fds=tuple(spec['pass_fds']))
        child_token=process_token(proc.pid)
        q.child(identity,proc.pid)
        started=time.monotonic()
        while proc.poll() is None:
            if stopped or parent_stopped() or q.get(identity)['cancel_requested']:raise QueueCancelled('Cancelled')
            if time.monotonic()-started > spec['timeout']:raise TimeoutError('Execution time limit reached')
            time.sleep(.25)
        code=proc.returncode
        status,message=('completed','Completed') if code==0 else ('failed','Process exited unsuccessfully')
    except QueueCancelled:status,message='cancelled','Cancelled'
    except TimeoutError as e:status,message='failed',str(e)
    finally:
        if proc:
            # Clean up descendants before releasing the resource slot.
            stop_process_tree(proc.pid,child_token)
            try:os.killpg(proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:pass
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
        q.finish(identity,status,message,code)
        if row["metadata"].get("quota_scope"):
            from backend.web_v2.online_access import AccessStore
            AccessStore().release_science(row["metadata"]["quota_scope"])
    return 0 if status=='completed' else 1

if __name__=='__main__':sys.exit(run(sys.argv[1]))
