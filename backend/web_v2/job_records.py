"""Stable job numbers, owner-scoped lookup, and private artifact ownership.

Execution state stays in the scheduler/domain stores. This index stores only
identity and display metadata, so a job number never acts as an access token.
"""
from datetime import datetime, timezone
from pathlib import Path
from backend.web_v2.job_queue import JobQueue


class JobRecords:
    def __init__(self, queue=None):
        self.queue = queue or JobQueue()
        with self.queue.connection(True) as c:
            c.execute('''CREATE TABLE IF NOT EXISTS job_records (
                serial INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL,
                identity TEXT NOT NULL, owner TEXT NOT NULL, kind TEXT NOT NULL,
                title TEXT NOT NULL, created REAL NOT NULL, number TEXT UNIQUE,
                UNIQUE(source,identity))''')
            c.execute('CREATE INDEX IF NOT EXISTS job_records_owner ON job_records(source,owner,created)')
            c.execute('''CREATE TABLE IF NOT EXISTS job_artifacts (
                path TEXT PRIMARY KEY, source TEXT NOT NULL, owner TEXT NOT NULL)''')
            c.execute('CREATE TABLE IF NOT EXISTS job_private_roots (path TEXT PRIMARY KEY)')

    def register(self, source, identity, owner, kind, title, created):
        with self.queue.connection(True) as c:
            c.execute('INSERT OR IGNORE INTO job_records(source,identity,owner,kind,title,created) VALUES(?,?,?,?,?,?)',
                      (source, identity, owner, kind, str(title)[:160], created))
            row = c.execute('SELECT * FROM job_records WHERE source=? AND identity=?', (source, identity)).fetchone()
            if row['owner'] != owner:
                raise FileNotFoundError('Job not found')
            number = row['number']
            if not number:
                day = datetime.fromtimestamp(created, timezone.utc).strftime('%Y%m%d')
                number = f'VF-{day}-{row["serial"]:06d}'
                c.execute('UPDATE job_records SET number=? WHERE serial=?', (number, row['serial']))
            return {'job_number': number, 'title': row['title']}

    def resolve(self, source, value, owner):
        with self.queue.connection() as c:
            row = c.execute('SELECT identity FROM job_records WHERE source=? AND owner=? AND (identity=? OR number=?)',
                            (source, owner, value, value.upper())).fetchone()
        if not row:
            raise FileNotFoundError('Job not found')
        return row['identity']

    def entries(self, source, owner):
        with self.queue.connection() as c:
            return [dict(r) for r in c.execute('SELECT identity,kind,title,number,created FROM job_records WHERE source=? AND owner=? ORDER BY created DESC,serial DESC', (source, owner))]

    def metadata(self, source, identity, owner):
        with self.queue.connection() as c:
            row = c.execute('SELECT number,title FROM job_records WHERE source=? AND identity=? AND owner=?',
                            (source, identity, owner)).fetchone()
        return {'job_number': row['number'], 'title': row['title']} if row and row['number'] else None

    def protect(self, path):
        with self.queue.connection(True) as c:
            c.execute('INSERT OR IGNORE INTO job_private_roots VALUES(?)', (str(Path(path).resolve()),))

    def grant(self, path, source, owner):
        path = str(Path(path).resolve())
        with self.queue.connection(True) as c:
            old = c.execute('SELECT source,owner FROM job_artifacts WHERE path=?', (path,)).fetchone()
            if old and (old['source'], old['owner']) != (source, owner):
                raise ValueError('Artifact already belongs to another workspace')
            c.execute('INSERT OR IGNORE INTO job_artifacts VALUES(?,?,?)', (path, source, owner))

    def check_artifact(self, path, owners, *, require_grant=False):
        # Check lexical and resolved names: a link cannot bypass a private root.
        for candidate in {Path(path).absolute(), Path(path).resolve()}:
            names = [str(candidate), *(str(p) for p in candidate.parents)]
            placeholders = ','.join('?' for _ in names)
            with self.queue.connection() as c:
                grants = c.execute(f'SELECT * FROM job_artifacts WHERE path IN ({placeholders})', names).fetchall()
                protected = c.execute(f'SELECT 1 FROM job_private_roots WHERE path IN ({placeholders}) LIMIT 1', names).fetchone()
            if grants:
                grant = max(grants, key=lambda g: len(g['path']))
                if owners.get(grant['source']) != grant['owner']:
                    raise FileNotFoundError('File not found')
            elif protected or require_grant:
                raise FileNotFoundError('File not found')


def page_jobs(jobs, *, query='', status='', kind='', offset=0, limit=20):
    """Filter only already-authorized jobs; return compact list rows."""
    query = query.strip().casefold()
    filtered = [j for j in jobs if
                (not query or query in f'{j.get("job_number", "")} {j.get("title", "")}'.casefold()) and
                (not kind or j['kind'].removeprefix('custom:') == kind) and
                (not status or (j['status'] in {'queued', 'running'} if status == 'active' else j['status'] == status))]
    fields = {'id', 'job_number', 'title', 'kind', 'created', 'status', 'progress', 'queue', 'resource', 'position', 'cancel_requested'}
    return {'items': [{k: v for k, v in j.items() if k in fields} for j in filtered[offset:offset+limit]],
            'total': len(filtered), 'active': sum(j['status'] in {'queued', 'running'} for j in jobs),
            'offset': offset, 'limit': limit}
