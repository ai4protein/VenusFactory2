"""Small persisted research event journal, derived from actual state transitions."""
import time


class ActivityJournal:
    def __init__(self, history=None):
        self.events = [dict(row) for row in (history or [])][-500:]
        self.seen = {(row["kind"], row["key"]): row["status"] for row in self.events}
        self.next_id = max((row["id"] for row in self.events), default=0) + 1

    def observe(self, state):
        items = [('phase', 'phase', state.get('phase', 'collecting'), '', '')]
        for group, field in [('step', 'steps'), ('service', 'remote_services'), ('chapter', 'chapters'), ('prediction', 'predictions')]:
            for row in state.get(field, []):
                label = row.get('title') or row.get('label') or row.get('task') or row['id']
                if group == 'prediction': label += ' · ' + row.get('model', '')
                items.append((group, row['id'], row.get('status', 'queued'), label, row.get('job_number', '')))
        for group, identity, status, label, job in items:
            key = (group, identity)
            if self.seen.get(key) == status: continue
            self.seen[key] = status
            self.events.append({'id': self.next_id, 'at': time.time(), 'kind': group, 'key': identity,
                                'status': status, 'label': label, 'job_number': job})
            self.next_id += 1
        return self.events[-500:]
