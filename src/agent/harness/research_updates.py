"""User revisions are queued outside episode progress and applied at a boundary."""
from __future__ import annotations
import copy
import time
import uuid


def validate_revision(text='', plan=None, attachments=None):
    if not isinstance(text, str) or len(text) > 8000:
        raise ValueError('Research update must be at most 8000 characters')
    if plan is not None:
        if not isinstance(plan, dict) or set(plan) != {'summary', 'steps'} or not isinstance(plan['summary'], str) or len(plan['summary']) > 8000:
            raise ValueError('Invalid plan')
        if not isinstance(plan['steps'], list) or not 1 <= len(plan['steps']) <= 24:
            raise ValueError('Use 1 to 24 research steps')
        ids = set()
        for row in plan['steps']:
            if not isinstance(row, dict) or set(row) != {'id', 'title', 'acceptance'} or any(not isinstance(v, str) or not v.strip() or len(v) > 2000 for v in row.values()) or row['id'] in ids:
                raise ValueError('Each step needs a unique ID, title and acceptance criterion')
            ids.add(row['id'])
    if not text.strip() and plan is None and not attachments:
        raise ValueError('Add a message, plan change or file')


def queue_revision(store, text='', plan=None, attachments=None, interaction=None):
    validate_revision(text or ('User response' if interaction else ''), plan, attachments)
    entry = {'id': uuid.uuid4().hex, 'at': time.time(), 'text': text.strip(), 'plan': copy.deepcopy(plan), 'attachments': attachments or []}
    def change(state):
        if state['status'] not in {'active', 'paused', 'interrupted', 'waiting_for_input'}:
            raise ValueError('This research has ended')
        progress = state['progress']
        if interaction:
            from .interactions import InteractionResponse, response_text
            answer = InteractionResponse.model_validate(interaction)
            receipts = progress.setdefault('interaction_responses', {})
            previous = receipts.get(answer.interaction_id)
            if previous:
                if previous != answer.model_dump():raise ValueError('A different response was already submitted')
                return
            gate = progress.get('pending_interaction')
            if not gate:raise ValueError('This question is no longer pending')
            entry['text'] = response_text(gate, answer)
            receipts[answer.interaction_id] = answer.model_dump()
            entry['interaction'] = {'request': gate, 'response': answer.model_dump()}
            progress.pop('pending_interaction',None)
        elif progress.get('pending_interaction'):
            raise ValueError('Respond to the pending question first')
        pending = state.setdefault('pending_research_updates', [])
        if len(pending) >= 32:
            raise ValueError('Wait for pending updates to be applied')
        pending.append(entry)
    return store.mutate(change)


def apply_revisions(store):
    def change(state):
        pending = state.get('pending_research_updates', [])
        if not pending:
            return
        progress = state['progress']
        for entry in pending:
            old = copy.deepcopy(progress.get('plan'))
            progress.setdefault('revision_history', []).append({**entry, 'previous_plan': old})
            if entry['plan'] is not None:
                # Any change can invalidate dependent steps. Revalidate the entire
                # revised plan; keep previous results in revision history only.
                progress['plan'] = {**copy.deepcopy(entry['plan']), 'needs_input': [], 'steps': [dict(s, status='pending', evidence=[]) for s in entry['plan']['steps']]}
            if entry['text']:
                progress.setdefault('user_updates', []).append(entry['text'])
            progress.setdefault('inputs', []).extend(entry['attachments'])
        progress['questions'] = []
        progress.pop('reviewed_episode', None)
        state['assessment'] = None
        state['pending_research_updates'] = []
    return store.mutate(change)
