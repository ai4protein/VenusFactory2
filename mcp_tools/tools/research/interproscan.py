"""Owned, resumable InterProScan tickets over the official EBI REST service."""
import hashlib
import json
import os
import re
import time
import uuid
import asyncio

BASE = 'https://www.ebi.ac.uk/Tools/services/rest/iprscan5'


def configuration():
    from mcp_tools.models.profiles import load_runtime_settings
    value = load_runtime_settings().get('interproscan', {})
    email = value.get('contact_email') or os.getenv('EBI_EMAIL') or os.getenv('NCBI_EMAIL', '')
    if not value.get('enabled', True) or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise ValueError('InterProScan contact email is not configured')
    return email


def owner():
    from backend.web_v2.job_queue import JobQueue, _held_slot
    value = os.getenv('VENUS_QUEUE_OWNER')
    identity = _held_slot.get() or os.getenv('VENUS_QUEUE_JOB_ID')
    if not value and identity:
        value = JobQueue().get(identity)['owner']
    if not value:
        from mcp_tools.tools.runtime_tool_policy import is_online_mode
        if is_online_mode():
            raise PermissionError('A session-owned task is required')
        value = 'local:' + str(os.getuid())
    return hashlib.sha256(value.encode()).hexdigest()


class Tickets:
    def __init__(self):
        from backend.web_v2.job_queue import JobQueue
        self.queue = JobQueue()

    def get_or_create(self, identity='', sequence=''):
        account = owner(); now = time.time()
        digest = hashlib.sha256(sequence.encode()).hexdigest()
        with self.queue.connection(write=True) as db:
            db.execute('CREATE TABLE IF NOT EXISTS interproscan_tickets (id TEXT PRIMARY KEY, owner TEXT NOT NULL, digest TEXT NOT NULL, created REAL NOT NULL, state TEXT NOT NULL)')
            if identity:
                if not re.fullmatch('[a-f0-9]{32}', identity):
                    raise ValueError('Use the task ID returned by InterProScan')
                row = db.execute('SELECT * FROM interproscan_tickets WHERE id=? AND owner=?', (identity, account)).fetchone()
                if not row:
                    raise PermissionError('InterProScan task not found')
                if now-row['created'] > 86400:
                    raise ValueError('InterProScan results have expired')
                return json.loads(row['state']), False
            row = db.execute('SELECT * FROM interproscan_tickets WHERE owner=? AND digest=? AND created>? ORDER BY created DESC LIMIT 1', (account, digest, now-86400)).fetchone()
            if row:
                return json.loads(row['state']), False
            rows = db.execute('SELECT state FROM interproscan_tickets WHERE created>?', (now-86400,)).fetchall()
            active = [json.loads(row['state']) for row in rows if json.loads(row['state'])['status'] in {'submitting', 'pending', 'unknown'}]
            if len(active) >= 4:
                raise ValueError('InterProScan is busy; wait for an existing task to finish')
            ticket = {'id': uuid.uuid4().hex, 'status': 'submitting', 'created': now, 'next_poll': now+15}
            db.execute('INSERT INTO interproscan_tickets VALUES (?,?,?,?,?)', (ticket['id'], account, digest, now, json.dumps(ticket)))
            return ticket, True

    def save(self, ticket):
        with self.queue.connection(write=True) as db:
            db.execute('UPDATE interproscan_tickets SET state=? WHERE id=? AND owner=?', (json.dumps(ticket), ticket['id'], owner()))

    def claim_poll(self, identity):
        with self.queue.connection(write=True) as db:
            row = db.execute('SELECT state FROM interproscan_tickets WHERE id=? AND owner=?', (identity, owner())).fetchone()
            if not row: raise PermissionError('InterProScan task not found')
            ticket = json.loads(row['state'])
            if ticket.get('next_poll', 0) > time.time(): return None
            ticket['next_poll'] = time.time()+15
            db.execute('UPDATE interproscan_tickets SET state=? WHERE id=? AND owner=?', (json.dumps(ticket), identity, owner()))
            return ticket


def parse_tsv(raw):
    hits = []
    for line in raw.decode('utf-8').splitlines():
        if not line or line.startswith('#'): continue
        fields = line.split('\t')
        if len(fields) < 11: raise ValueError('Incomplete InterProScan result')
        hit = dict(analysis=fields[3], signature=fields[4], description=fields[5], start=int(fields[6]), end=int(fields[7]), score=fields[8])
        for i, key in [(11,'interpro_accession'),(12,'interpro_description'),(13,'go_terms'),(14,'pathways')]:
            if len(fields)>i and fields[i]!='-': hit[key]=fields[i]
        hits.append(hit)
    return {'hits':hits[:1000], 'count':len(hits), 'truncated':len(hits)>1000,
            'interpretation':'Sequence signatures and predicted domains; not experimental functional validation.'}


async def scan(context, row, emit):
    from .services import request, sequence, ServiceError
    email = configuration(); store = Tickets()
    seq = sequence(context.get('sequence', '')) if not context.get('remote_job_id') else ''
    ticket, fresh = store.get_or_create(context.get('remote_job_id', ''), seq)
    def publish():
        row.update(job_id=ticket['id'], status='pending', next_poll=ticket.get('next_poll'), evidence_level='sequence_signature')
        if ticket.get('remote_id'):
            row.update(remote_job_id=ticket['remote_id'], url=BASE+'/result/'+ticket['remote_id']+'/tsv')
        emit()
    if fresh:
        try:
            raw = await request('POST', BASE+'/run', data={'email':email,'title':'VenusFactory protein annotation','sequence':'>query\n'+seq,'stype':'p','goterms':'true','pathways':'true'}, submit=True)
            remote = raw.decode().strip()
            if not re.fullmatch(r'iprscan5-[A-Za-z0-9_-]{5,160}', remote):
                raise ServiceError('InterProScan did not return a valid job ID')
            ticket.update(remote_id=remote,status='pending',next_poll=time.time()+15);store.save(ticket)
        except BaseException:
            ticket.update(status='unknown');store.save(ticket)
            raise
    publish()
    if ticket['status']=='completed': return ticket['data']
    if ticket['status'] in {'failed','unknown'}:
        raise ServiceError('InterProScan submission failed or its outcome is unknown; it was not resubmitted')
    if fresh or context.get('submit_only'): return None
    if not ticket.get('remote_id'): return None
    claimed = store.claim_poll(ticket['id'])
    if claimed is None: return None
    ticket = claimed
    status = (await request('GET', BASE+'/status/'+ticket['remote_id'], rid=ticket['remote_id'])).decode().strip()
    if status=='FINISHED':
        data = parse_tsv(await request('GET', BASE+'/result/'+ticket['remote_id']+'/tsv'))
        ticket.update(status='completed',data=data);store.save(ticket);publish();return data
    if status in {'ERROR','FAILURE','NOT_FOUND'}:
        ticket.update(status='failed');store.save(ticket)
        raise ServiceError('InterProScan analysis failed or expired')
    if status not in {'RUNNING','PENDING','QUEUED'}:
        raise ServiceError('Unrecognized InterProScan status')
    store.save(ticket);publish();return None


async def run(context, row, emit):
    result=await scan(context,row,emit)
    if result is not None or context.get('submit_only') or context.get('poll_attempts')==1:
        return result
    for _ in range(24):
        await asyncio.sleep(max(0,row.get('next_poll',time.time()+15)-time.time()))
        result=await scan({'remote_job_id':row['job_id']},row,emit)
        if result is not None:return result
    return None
