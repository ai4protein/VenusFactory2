"""Read-only RSI case repositories. Research admission never grants tool access."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import zipfile

from utils.scoped_files import open_file, parts
from backend.web_v2.research_delivery import project_files, report

MAX_PREVIEW = 200_000
MAX_ARCHIVE = 256 * 1024**2


class RsiPlaza:
    def __init__(self, science, library=None):
        self.science = science
        if library is None:
            from agent.harness.research_workspace import LIBRARY
            library = LIBRARY
        self.library = Path(library)

    @staticmethod
    def public_card(card):
        return {'id': 'capability:' + card['id'], 'title': card['name'],
                'summary': card.get('description', ''), 'author_email': card.get('author_email', ''),
                'visibility': 'public', 'status': card.get('review_status', 'reviewed'),
                'created_at': card.get('publication', {}).get('published_at', 0),
                'scope': card.get('scope', {}), 'version': card.get('version', '')}

    @staticmethod
    def private_card(sid, gid, meta):
        title = meta['spec']['objective']
        return {'id': f'research:{sid}:{gid}', 'title': title, 'summary': '',
                'author_email': '', 'visibility': 'private', 'status': 'saved',
                'created_at': meta.get('created_at', 0), 'session_id': sid}

    def private_cards(self, sessions, query):
        rows = []
        words = query.casefold().split()
        # Only browser-authorized session roots; never enumerate the global runs tree.
        for sid in sessions:
            for path in self.science.session_folder(sid).glob('*.json'):
                if not re.fullmatch('[a-f0-9]{32}', path.stem):
                    continue
                try:
                    folder, _, meta = self.science.metadata(sid, path.stem)
                    if meta.get('spec', {}).get('driver') != 'autonomous_research.v1' or not (folder / 'goal.json').is_file():
                        continue
                    card = self.private_card(sid, path.stem, meta)
                    if all(word in card['title'].casefold() for word in words):
                        rows.append(card)
                except (ValueError, OSError, KeyError):
                    continue
        return sorted(rows, key=lambda c: (str(c['created_at']), c['id']), reverse=True)

    def listing(self, sessions, query='', view='all', offset=0, limit=24):
        if view not in {'all', 'mine'} or not isinstance(query, str) or len(query) > 200 or type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('Invalid case filter')
        from backend.web_v2.rsi_legacy_cases import cards as legacy_cards
        mine = self.private_cards(sessions, query) + legacy_cards(query)
        items = mine[offset:offset + limit]
        total = len(mine)
        if view == 'all' and self.library.is_dir():
            from agent.harness.capability_catalog import search
            public_offset = max(0, offset - len(mine))
            public_limit = min(50, max(1, limit - len(items)))
            page = search(self.library, query, public_limit, public_offset, public_only=True)
            total += page['total']
            if len(items) < limit:
                items.extend(self.public_card(c) for c in page['items'])
                # The underlying catalog deliberately caps each metadata page at 50.
                if len(items) < limit and page['next_offset'] is not None:
                    next_page = search(self.library, query, min(50, limit-len(items)), page['next_offset'], public_only=True)
                    items.extend(self.public_card(c) for c in next_page['items'])
        next_offset = offset + len(items)
        return {'items': items, 'total': total, 'next_offset': next_offset if next_offset < total else None}

    def resolve(self, identity, sessions):
        if identity.startswith('research:'):
            bits = identity.split(':')
            if len(bits) != 3 or bits[1] not in sessions:
                raise FileNotFoundError('Case not found')
            _, sid, gid = bits
            folder, _, meta = self.science.metadata(sid, gid)
            if meta.get('spec', {}).get('driver') != 'autonomous_research.v1':
                raise FileNotFoundError('Case not found')
            with open_file(folder, 'goal.json') as stream:
                state = json.load(stream)
            card = self.private_card(sid, gid, meta)
            card['status'] = state['status']
            return card, folder / 'workspace', state, folder
        if identity.startswith('capability:') and re.fullmatch('[a-f0-9]{64}', identity[11:]):
            from agent.harness.research_workspace import admitted_manifest
            from agent.harness.capability_catalog import require_registered
            from agent.harness.curated_publication import validate_publication
            bundle, manifest, receipt = admitted_manifest(identity[11:], self.library)
            card = require_registered(self.library, manifest['id'], receipt['receipt_sha256'])
            if card.get('visibility') != 'public':
                raise FileNotFoundError('Case not found')
            validate_publication(card.get('publication'), manifest, receipt, card['origin_goal'])
            return self.public_card(card), bundle, manifest, None
        raise FileNotFoundError('Case not found')

    @staticmethod
    def _names(workspace, data, folder):
        if folder is not None:
            names = project_files(workspace, limit=10001)
        else:
            names = sorted(set([data['source'], *data.get('files', {})]))
        from backend.web_v2.research_delivery import EXCLUDED
        result = []
        for name in names:
            segments = parts(name)
            if any(s.startswith('.') or s in EXCLUDED or s.lower().startswith(('credential', 'service.key')) for s in segments):
                continue
            with open_file(workspace, name) as stream:
                import os
                size = os.fstat(stream.fileno()).st_size
            result.append({'path': name, 'size': size})
        if len(result) > 10000:
            raise ValueError('Case has too many files')
        return result

    @staticmethod
    def _documents(card, data, folder):
        readme = '# ' + card['title'] + '\n\n' + card['summary'] + '\n\nThis RSI case contains research code and evidence. Follow its recorded evaluation and limitations when reproducing it.\n'
        summary = report(data) if folder is not None else '# Research review\n\n' + json.dumps(data.get('review_spec', {}), ensure_ascii=False, indent=2)
        return {'README.md': readme, 'REPORT.md': summary}

    def detail(self, identity, sessions):
        if identity.startswith('scientist:'):
            from backend.web_v2.rsi_legacy_cases import documents
            card, docs = documents(identity[10:])
            return {**card, 'files': [{'path': name, 'size': len(text.encode())} for name, text in docs.items()],
                    'readme': 'README.md', 'report': 'REPORT.md', 'download_url': '/api/rsi-plaza/cases/' + identity + '/download'}
        card, workspace, data, folder = self.resolve(identity, sessions)
        files = self._names(workspace, data, folder)
        docs = self._documents(card, data, folder)
        files = [r for r in files if r['path'] != 'REPORT.md']
        for name, text in docs.items():
            if name not in {r['path'] for r in files}:
                files.insert(0, {'path': name, 'size': len(text.encode())})
        from urllib.parse import quote
        return {**card, 'files': files, 'readme': 'README.md', 'report': 'REPORT.md',
                'download_url': '/api/rsi-plaza/cases/' + quote(identity, safe='') + '/download'}

    def preview(self, identity, sessions, path):
        parts(path)  # Reject paths even when a virtual document is requested.
        if identity.startswith('scientist:'):
            from backend.web_v2.rsi_legacy_cases import documents
            _, docs = documents(identity[10:])
            if path not in docs: raise FileNotFoundError('File not found')
            raw = docs[path].encode()
            return {'path': path, 'text': raw[:MAX_PREVIEW].decode('utf-8', errors='replace'), 'size': len(raw), 'truncated': len(raw) > MAX_PREVIEW}
        card, workspace, data, folder = self.resolve(identity, sessions)
        from backend.web_v2.research_delivery import EXCLUDED
        segments = parts(path)
        if any(v.startswith('.') or v in EXCLUDED or v.lower().startswith(('credential', 'service.key')) for v in segments):
            raise FileNotFoundError('File not found')
        if folder is None and path not in {data['source'], *data.get('files', {}), 'README.md', 'REPORT.md'}:
            raise FileNotFoundError('File not found')
        docs = self._documents(card, data, folder)
        if path == 'REPORT.md' or (path == 'README.md' and folder is None and path not in {data['source'], *data.get('files', {})}):
            raw = docs[path].encode()
            size = len(raw)
        else:
            try:
                with open_file(workspace, path) as stream:
                    import os
                    size = os.fstat(stream.fileno()).st_size
                    raw = stream.read(MAX_PREVIEW + 1)
            except FileNotFoundError:
                if path != 'README.md': raise
                raw = docs[path].encode()
                size = len(raw)
        if b'\x00' in raw:
            return {'path': path, 'text': '', 'size': size, 'truncated': False, 'binary': True}
        return {'path': path, 'text': raw[:MAX_PREVIEW].decode('utf-8', errors='replace'),
                'size': size, 'truncated': size > MAX_PREVIEW}

    def download(self, identity, sessions):
        if identity.startswith('scientist:'):
            from backend.web_v2.rsi_legacy_cases import documents
            _, docs = documents(identity[10:])
            if sum(len(text.encode()) for text in docs.values()) > MAX_ARCHIVE: raise ValueError('Case exceeds the archive budget')
            archive = tempfile.SpooledTemporaryFile(max_size=8*1024**2)
            try:
                with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
                    for name, text in docs.items(): out.writestr(name, text)
                archive.seek(0)
                return archive
            except BaseException:
                archive.close()
                raise
        card, workspace, data, folder = self.resolve(identity, sessions)
        files = self._names(workspace, data, folder)
        if sum(r['size'] for r in files) > MAX_ARCHIVE:
            raise ValueError('Case exceeds the 256 MB archive budget')
        if folder is not None:
            if self.science.live(folder):
                raise ValueError('Pause the research before exporting its project')
        files = [r for r in files if r['path'] != 'REPORT.md']
        archive = tempfile.SpooledTemporaryFile(max_size=8*1024**2)
        try:
            docs = self._documents(card, data, folder)
            with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
                total = 0
                for name, text in docs.items():
                    if name not in {r['path'] for r in files}:
                        total += len(text.encode())
                        if total > MAX_ARCHIVE: raise ValueError('Case exceeds the archive budget')
                        out.writestr(name, text)
                for row in files:
                    with open_file(workspace, row['path']) as src, out.open(row['path'], 'w') as dst:
                        while chunk := src.read(1024**2):
                            total += len(chunk)
                            if total > MAX_ARCHIVE:
                                raise ValueError('Case exceeds the archive budget')
                            dst.write(chunk)
            archive.seek(0)
            return archive
        except BaseException:
            archive.close()
            raise
