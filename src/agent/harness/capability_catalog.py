"""Compact discovery index. Full source, schemas and evidence are loaded on demand."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time

CURRENT = "id NOT IN (SELECT f.id FROM capability_families f JOIN capability_heads h ON f.family=h.family WHERE f.id != h.id)"

@contextmanager
def database(library):
    library = Path(library); library.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(library/'catalog.sqlite3', timeout=20)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('CREATE TABLE IF NOT EXISTS capabilities (id TEXT PRIMARY KEY, creator TEXT NOT NULL, name TEXT NOT NULL, version TEXT NOT NULL, search_text TEXT NOT NULL, card TEXT NOT NULL, receipt_hash TEXT NOT NULL, created REAL NOT NULL, UNIQUE(creator,name,version))')
    con.execute('CREATE TABLE IF NOT EXISTS capability_families (id TEXT PRIMARY KEY, family TEXT NOT NULL)')
    con.execute('CREATE TABLE IF NOT EXISTS capability_heads (family TEXT PRIMARY KEY, id TEXT NOT NULL)')
    try:
        with con: yield con
    finally: con.close()

def register(library, manifest, receipt, goal_id, *, publication=None):
    from agent.rsi.artifacts.contract import family_id
    if publication is not None:
        from .curated_publication import validate_publication
        validate_publication(publication, manifest, receipt, goal_id)
    visibility = 'public' if publication is not None else 'private'
    spec = manifest['review_spec']
    family = family_id(goal_id, spec['kind'], spec['scope'])
    card = {k: manifest[k] for k in ['id', 'name', 'description', 'validation']}
    card.update(family_id=family, author=spec['author'], version=spec['version'], kind=spec['kind'], scope=spec['scope'], review_status='approved', visibility=visibility, origin_goal=goal_id)
    if publication is not None:
        card.update(author_email=publication['author_email'], publication=publication)
    search = ' '.join([card['name'], card['description'], spec['author'], *[x for values in spec['scope'].values() for x in values]]).casefold()
    with database(library) as con:
        con.execute('INSERT OR IGNORE INTO capabilities VALUES (?,?,?,?,?,?,?,?)', (manifest['id'], goal_id, manifest['name'], spec['version'], search, json.dumps(card, ensure_ascii=False), receipt['receipt_sha256'], time.time()))
        row = con.execute('SELECT id FROM capabilities WHERE creator=? AND name=? AND version=?', (goal_id, manifest['name'], spec['version'])).fetchone()
        if not row or row[0] != manifest['id']: raise ValueError('Publish a new version instead of replacing an existing one')
        if publication is not None:
            con.execute('UPDATE capabilities SET card=?,receipt_hash=? WHERE id=?', (json.dumps(card,ensure_ascii=False),receipt['receipt_sha256'],manifest['id']))
        con.execute('INSERT OR IGNORE INTO capability_families VALUES (?,?)', (manifest['id'], family))
        con.execute('INSERT INTO capability_heads VALUES (?,?) ON CONFLICT(family) DO UPDATE SET id=excluded.id', (family, manifest['id']))
    return card

def search(library, query='', limit=10, offset=0, *, public_only=False, excluded=()):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError('Use a search query of at most 200 characters')
    if type(limit) is not int or not 1 <= limit <= 50 or type(offset) is not int or offset < 0:
        raise ValueError('Use limit 1-50 and a nonnegative offset')
    terms = query.casefold().split()
    conditions = ' AND '.join("search_text LIKE ? ESCAPE '\\'" for _ in terms) or '1=1'
    conditions = '(' + conditions + ') AND ' + CURRENT
    values = ['%'+word.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%' for word in terms]
    if public_only:
        conditions += " AND json_extract(card,'$.visibility')='public' AND json_extract(card,'$.publication.policy')='operator_selected.v1'"
    if excluded:
        conditions += ' AND id NOT IN (SELECT value FROM json_each(?))'
        values.append(json.dumps(list(excluded)))
    with database(library) as con:
        total = con.execute('SELECT COUNT(*) FROM capabilities WHERE '+conditions, values).fetchone()[0]
        rows = con.execute('SELECT card FROM capabilities WHERE '+conditions+' ORDER BY created DESC,id LIMIT ? OFFSET ?', [*values,limit,offset]).fetchall()
    items = [json.loads(row[0]) for row in rows]
    return {'total':total,'items':items,'next_offset':offset+len(items) if offset+len(items)<total else None}


def discovery_card(card):
    result = {key:card[key] for key in ('id','family_id','kind','author','author_email','version','review_status') if key in card}
    result.update(name=card['name'][:120],description=card['description'][:240])
    scope = card.get('scope') or {}
    result['scope'] = {key:[value[:80] for value in scope.get(key,[])[:3]]
                       for key in ('proteins','properties','scenarios')}
    if any(len(scope.get(key,[]))>3 or any(len(value)>80 for value in scope.get(key,[]))
           for key in result['scope']):
        result['scope_truncated'] = True
    return result


def cards(library):
    with database(library) as con: return [json.loads(row[0]) for row in con.execute('SELECT card FROM capabilities WHERE '+CURRENT+' ORDER BY created DESC,id')]

def require_registered(library, identity, receipt_hash):
    with database(library) as con: row = con.execute('SELECT receipt_hash,card FROM capabilities WHERE id=?', (identity,)).fetchone()
    if not row or row[0] != receipt_hash: raise ValueError('Capability is not admitted to this registry')
    return json.loads(row[1])
