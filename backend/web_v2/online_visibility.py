"""Online deletion hides conversations while retaining their server artifacts."""
from contextlib import contextmanager
from pathlib import Path
import os
import sqlite3
import time


def is_online():
    return os.getenv('WEBUI_V2_MODE', 'local').strip().lower() == 'online'


class OnlineVisibility:
    def __init__(self, path=None):
        if path is None:
            from backend.web_v2.online_access import AccessStore
            path = AccessStore().path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self.connection() as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS online_deleted_conversations (source TEXT NOT NULL, identity TEXT NOT NULL, deleted REAL NOT NULL, PRIMARY KEY(source,identity))')
            conn.execute('CREATE TABLE IF NOT EXISTS online_conversation_file_roots (source TEXT NOT NULL, identity TEXT NOT NULL, path TEXT NOT NULL, PRIMARY KEY(source,identity,path))')
            conn.execute('CREATE INDEX IF NOT EXISTS online_conversation_file_paths ON online_conversation_file_roots(path)')

    @contextmanager
    def connection(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def hide(self, source, identity, roots=()):
        # The caller authenticates the conversation before reaching this method.
        with self.connection() as conn:
            for root in roots:
                if root:
                    conn.execute('INSERT OR IGNORE INTO online_conversation_file_roots VALUES(?,?,?)', (source, identity, str(Path(root).resolve())))
            conn.execute('INSERT OR IGNORE INTO online_deleted_conversations VALUES(?,?,?)', (source, identity, time.time()))

    def is_hidden(self, source, identity):
        with self.connection() as conn:
            return conn.execute('SELECT 1 FROM online_deleted_conversations WHERE source=? AND identity=?', (source, identity)).fetchone() is not None

    def hidden_ids(self, source):
        with self.connection() as conn:
            return {row[0] for row in conn.execute('SELECT identity FROM online_deleted_conversations WHERE source=?', (source,))}

    def blocks_file(self, path):
        names = set()
        for candidate in (Path(path).absolute(), Path(path).resolve()):
            names.update(str(p) for p in [candidate, *candidate.parents])
        placeholders = ','.join('?' for _ in names)
        with self.connection() as conn:
            row = conn.execute(f'''SELECT 1 FROM online_conversation_file_roots r
                JOIN online_deleted_conversations d ON d.source=r.source AND d.identity=r.identity
                WHERE r.path IN ({placeholders}) LIMIT 1''', tuple(names)).fetchone()
        return row is not None


def guard_generic_file(path):
    """Conversation files must use an authenticated conversation file endpoint."""
    if not is_online():
        return
    from mcp_tools.common.common_utils import get_web_v2_root_dir, get_temp_outputs_base_dir
    from backend.web_v2.science_goals import service
    roots = [get_temp_outputs_base_dir() / 'agent', get_web_v2_root_dir() / 'sessions',
             get_web_v2_root_dir() / 'results' / 'researcher', service.state_root]
    candidates = [Path(path).absolute(), Path(path).resolve()]
    if any(candidate.is_relative_to(Path(root).resolve()) for candidate in candidates for root in roots):
        raise FileNotFoundError('Use the conversation file endpoint')
    if OnlineVisibility().blocks_file(path):
        raise FileNotFoundError('Conversation not found')
