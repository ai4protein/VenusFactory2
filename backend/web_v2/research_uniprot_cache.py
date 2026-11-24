"""Short-lived, bounded cache of public UniProt records selected in Researcher."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import time


def selected_entries(*, save=(), accession=None):
    directory = os.getenv("VENUS_REPORT_ROOT")
    if not directory:
        return None
    db = Path(directory) / ".uniprot.sqlite3"
    now = int(time.time())
    try:
        if accession and not db.is_file():
            return None
        db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db, timeout=0.5)) as conn, conn:
            conn.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, fetched INTEGER, body TEXT)")
            if accession:
                row = conn.execute("SELECT body FROM entries WHERE id=? AND fetched>=?",
                                   (accession.upper(), now - 86400)).fetchone()
                return json.loads(row[0]) if row else None
            for entry in save:
                identity = entry.get("primaryAccession")
                if not identity or not (entry.get("sequence") or {}).get("value"):
                    continue
                body = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                if len(body.encode("utf-8")) <= 512_000:
                    conn.execute("INSERT OR REPLACE INTO entries VALUES(?,?,?)", (identity.upper(), now, body))
            conn.execute("DELETE FROM entries WHERE fetched < ?", (now - 86400,))
            conn.execute("DELETE FROM entries WHERE rowid IN "
                         "(SELECT rowid FROM entries ORDER BY fetched DESC, rowid DESC LIMIT -1 OFFSET 128)")
    except (OSError, sqlite3.Error, ValueError):
        # The public endpoint remains available when caching is unavailable.
        return None
