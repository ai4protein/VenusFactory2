"""Editable submission drafts with immutable, goal-local attempt snapshots."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from .contracts import digest
from .host import write_json
from utils.scoped_files import read_bytes, write_bytes

FIELDS = {'name', 'description', 'source', 'files', 'test_inputs', 'review_spec'}
REQUIRED = FIELDS - {'review_spec'}
DRAFT = 'review/proposal.json'


class ReviewSubmission:
    def __init__(self, workspace, control, inline, proposal_file=''):
        self.workspace = Path(workspace)
        self.inline = not proposal_file
        self.path = proposal_file or DRAFT
        if proposal_file:
            if any(value not in (None, '', [], {}) for value in inline.values()):
                raise ValueError('Use proposal_file alone, or supply the inline proposal fields')
            payload = json.loads(read_bytes(self.workspace, proposal_file, max_bytes=200000))
        else:
            payload = inline
        if not isinstance(payload, dict) or set(payload) - FIELDS:
            raise ValueError('A proposal file accepts only: ' + ', '.join(sorted(FIELDS)))
        missing = REQUIRED - set(payload)
        if missing:
            raise ValueError('Missing proposal fields: ' + ', '.join(sorted(missing)))
        for name in ('name', 'description', 'source'):
            if not isinstance(payload[name], str) or not payload[name].strip():
                raise ValueError('Declare proposal.' + name)
        if not isinstance(payload['files'], list) or any(not isinstance(v, str) for v in payload['files']):
            raise ValueError('proposal.files must list relative runtime filenames')
        if not isinstance(payload['test_inputs'], list) or any(not isinstance(v, dict) for v in payload['test_inputs']):
            raise ValueError('proposal.test_inputs must list input objects')
        self.payload = {**payload, 'review_spec': payload.get('review_spec')}
        self.raw = json.dumps(self.payload, ensure_ascii=False, indent=2, allow_nan=False).encode()
        if len(self.raw) > 200000:
            raise ValueError('Proposal exceeds 200 KB')
        self.attempt = Path(control) / 'submissions' / uuid.uuid4().hex
        self.attempt.mkdir(parents=True)
        write_json(self.attempt / 'request.json', self.payload)
        if self.inline:
            try:
                read_bytes(self.workspace, self.path, max_bytes=200000)
            except FileNotFoundError:
                write_bytes(self.workspace, self.path, self.raw)
        self.identity = digest(self.payload)

    def record(self, status, *, error=None, candidate_id=None):
        # Invalid inline retries cannot erase a previously complete draft.
        # Successful calls refresh the editable copy; immutable requests remain.
        if status == 'submitted' and self.inline:
            write_bytes(self.workspace, self.path, self.raw)
        state = {'proposal_file': self.path, 'attempt_id': self.attempt.name,
                 'request_sha256': self.identity, 'status': status}
        if error:
            state['error'] = str(error)[:1200]
        if candidate_id:
            state['candidate_id'] = candidate_id
        write_json(self.attempt / 'status.json', state)
        return state
