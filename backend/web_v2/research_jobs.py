"""Private, durable research jobs managed by the shared queue."""
from __future__ import annotations
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

from backend.web_v2.job_queue import JobQueue, launch, PROJECT, TERMINAL
from backend.web_v2.job_records import JobRecords
from backend.web_v2.research_report import ACCESSION, clean_sequence, model_option, run_research, validate_scopes


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    tmp.chmod(0o600)
    os.replace(tmp, path)


class ResearchJobs:
    def __init__(self, root=None, queue=None):
        default = Path.home()/'.local/share/venusfactory/reports'/hashlib.sha256(str(PROJECT).encode()).hexdigest()[:16]
        self.root = Path(root or os.getenv('VENUS_REPORT_ROOT') or default).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.queue = queue or JobQueue()
        self.records = JobRecords(self.queue)
        self.records.protect(self.root)

    def folder(self, identity):
        if not re.fullmatch('[a-f0-9]{32}', identity):
            raise FileNotFoundError('Research not found')
        return self.root/identity

    def read(self, identity):
        return json.loads((self.folder(identity)/'request.json').read_text())

    def owned(self, owner, identity):
        identity = self.records.resolve('report', identity, owner)
        from backend.web_v2.online_visibility import OnlineVisibility, is_online
        if is_online() and OnlineVisibility().is_hidden('report', identity):
            raise FileNotFoundError('Research not found')
        row = self.read(identity)
        if row['owner'] != owner:
            raise FileNotFoundError('Research not found')
        return identity, row

    def start(self, owner, payload, *, admit=None):
        from agent.rsi.engine.storage import file_lock
        request_id = payload.pop("request_id", None) or uuid.uuid4().hex
        payload=dict(payload)
        payload.pop('material_sources',None)  # Only owner-checked server snapshots can supply evidence.
        submitted = dict(payload)
        self.validate(payload)
        identity = hashlib.sha256((owner + ":" + request_id).encode()).hexdigest()[:32]
        locks = self.root / ".requests"
        locks.mkdir(exist_ok=True)
        with file_lock(locks / (identity + ".lock")):
            if (self.folder(identity)/"request.json").exists():
                old = self.read(identity)
                if old.get("submitted_payload", old["payload"]) != submitted:
                    raise ValueError("Request identity belongs to a different report")
                return self.get(owner, identity)
            from backend.web_v2.research_materials import sources
            payload['material_sources']=sources(self.root,owner,payload.get('material_ids',[]),identity)
            return self._start(owner, payload, identity, submitted, admit=admit)

    def validate(self, payload):
        validate_scopes(payload['analyses'])
        model_option(payload['model_id'])  # credential readiness and free route before admission
        if payload.get('accession') and not ACCESSION.fullmatch(payload['accession']):
            raise ValueError('Choose a UniProt entry')
        if payload.get('sequence'):
            payload['sequence'] = clean_sequence(payload['sequence'])
        if payload.get('structure_pdb'):
            from mcp_tools.tools.research.services import structure_chain
            pdb, seq, chain = structure_chain(payload['structure_pdb'], payload.get('structure_chain', ''))
            if payload.get('sequence') and seq != payload['sequence']:
                raise ValueError('The sequence must match the selected structure chain')
            if payload.get('accession'):
                raise ValueError('Choose either a UniProt entry or an uploaded structure')
            payload.update(structure_pdb=pdb, structure_chain=chain, sequence=seq)
        if not payload.get('accession') and not payload.get('sequence'):
            raise ValueError('Choose a protein or supply a sequence')

    def _start(self, owner, payload, identity, submitted, *, admit=None):
        from backend.web_v2.research_materials import snapshot
        snapshot(self.root,owner,identity,payload.get('material_sources',[]))
        if admit: admit(identity)
        row = {'id': identity, 'owner': owner, 'created': time.time(), 'payload': payload,
               'submitted_payload': submitted,
               'title': (payload.get('protein') or payload.get('accession') or 'Sequence research')[:160]}
        folder = self.folder(identity)
        write(folder/'request.json', row)
        info = self.records.register('report', identity, owner, 'research', row['title'], row['created'])
        self.records.grant(folder, 'report', owner)
        from backend.web_v2.research_report import EXPENSIVE
        env = dict(os.environ, VENUS_REPORT_ROOT=str(self.root))
        try:
            qid, _ = launch([sys.executable, '-u', '-m', 'backend.web_v2.research_jobs', identity], owner=owner,
                            kind='protein-research', resource='research',
                            timeout=2400, env=env, queue=self.queue, session_key='report:'+identity, metadata={'mode':'researcher','report_id':identity,'title': row['title'], 'job_number': info['job_number']})
        except Exception:
            write(folder/'state.json', {'status': 'failed', 'message': 'Research could not be queued'})
            raise
        row['queue_id'] = qid
        write(folder/'request.json', row)
        return self.get(owner, identity)

    def get(self, owner, identity):
        identity, row = self.owned(owner, identity)
        folder = self.folder(identity)
        state = json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else {}
        queue = self.queue.public(row['queue_id'], owner) if row.get('queue_id') else {}
        status = queue.get('status', state.get('status', 'queued'))
        if status == 'completed' and not state.get('report'):
            status = 'failed'
        if status == 'completed' and state.get('minimum_met') is False:
            status = 'incomplete'
        if state.get('chapters') and state.get('report'):
            from backend.web_v2.research_longform import assemble
            state['report'] = assemble(state, row['payload'].get('lang', 'en'))
        from backend.web_v2.research_longform import MAX_MODEL_CALLS
        resumable = status in {'failed', 'interrupted', 'incomplete'} and int(state.get('model_calls', 0)) < MAX_MODEL_CALLS and int(row.get('resume_count', 0)) < 2
        reason = ''
        if status in {'failed', 'interrupted', 'incomplete'}:
            reason = state.get('stop_reason') or ('timeout' if 'limit' in queue.get('message', '').lower() else 'interrupted' if status == 'interrupted' else 'incomplete' if status == 'incomplete' else 'unavailable')
        return {**state, 'resumable': resumable, 'stop_reason': reason, 'id': identity, 'title': row['title'], 'created': row['created'], 'status': status,
                'job_number': self.records.register('report', identity, owner, 'research', row['title'], row['created'])['job_number'],
                'queue': queue, 'model_id': row['payload']['model_id'], 'analyses': row['payload']['analyses'],
                'focus': row['payload'].get('focus', ''),
                'materials':[{k:r[k] for k in ('id','name','bytes','extraction','url')} for r in row['payload'].get('material_sources',[])]}

    def history(self, owner):
        rows = []
        for r in self.records.entries('report', owner)[:100]:
            try:
                j = self.get(owner, r['identity'])
                rows.append({k: j[k] for k in ('id', 'title', 'status', 'created', 'job_number')})
            except FileNotFoundError:
                continue
        return sorted(rows, key=lambda r: r['created'], reverse=True)

    def delete_online(self, owner, identity):
        from backend.web_v2.online_visibility import OnlineVisibility, is_online
        from mcp_tools.common.common_utils import get_web_v2_root_dir
        if not is_online():
            raise ValueError('Online deletion requires online mode')
        identity, row = self.owned(owner, identity)
        result = self.get(owner, identity)
        if result['status'] in {'queued', 'running'}:
            raise ValueError('Stop this research before deleting it')
        OnlineVisibility().hide('report', identity, [self.folder(identity), get_web_v2_root_dir() / 'results' / 'researcher' / identity])
        return {'success': True, 'id': identity}

    def resume(self, owner, identity, *, admit=None):
        from agent.rsi.engine.storage import file_lock
        identity, _ = self.owned(owner, identity)
        folder = self.folder(identity)
        with file_lock(folder / '.resume.lock'):
            _, row = self.owned(owner, identity)
            current = self.get(owner, identity)
            if current['status'] in {'queued', 'running'}: return current
            if not current['resumable']: raise ValueError('This research cannot resume within its run budget')
            self.validate(dict(row['payload']))  # same free route and current online policy
            if admit: admit(identity)
            env = dict(os.environ, VENUS_REPORT_ROOT=str(self.root))
            qid, _ = launch([sys.executable, '-u', '-m', 'backend.web_v2.research_jobs', identity], owner=owner,
                            kind='protein-research', resource='research', timeout=2400, env=env,
                            queue=self.queue, session_key='report:'+identity, metadata={'mode':'researcher','report_id':identity,'title': row['title'], 'job_number': current['job_number']})
            row.update(queue_id=qid, resume_count=int(row.get('resume_count', 0)) + 1)
            write(folder/'request.json', row)
            return self.get(owner, identity)

    def cancel(self, owner, identity):
        identity, row = self.owned(owner, identity)
        if row.get('queue_id'):
            self.queue.cancel(row['queue_id'], owner)
        return self.get(owner, identity)


def worker(identity):
    service = ResearchJobs()
    row = service.read(identity)
    folder = service.folder(identity)
    from backend.web_v2.research_activity import ActivityJournal
    checkpoint = json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else None
    journal = ActivityJournal((checkpoint or {}).get('activity'))
    def update(value):
        value['activity'] = journal.observe(value)
        write(folder/'state.json', value)
        from backend.web_v2.task_progress import publish_progress
        publish_progress(value)
    try:
        from backend.web_v2.research_longform import run_long_report
        checkpoint = json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else None
        asyncio.run(asyncio.wait_for(run_long_report(row['payload'], update, service_cache=folder/'remote', checkpoint=checkpoint), timeout=2300))
    except Exception as exc:
        existing = json.loads((folder/'state.json').read_text()) if (folder/'state.json').exists() else {}
        update({**existing, 'status': 'failed', 'stop_reason': 'timeout' if isinstance(exc, TimeoutError) else 'unavailable', 'message': 'Research stopped before completion'})
        raise SystemExit(1) from None


if __name__ == '__main__':
    worker(sys.argv[1])
