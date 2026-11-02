"""Owner-only Scientist project export; never include worker control directories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
import stat

from agent.rsi.artifacts.contract import selected_deliveries

from utils.scoped_files import open_file

EXCLUDED = {'.git', '.env', '__pycache__', 'node_modules', '.venv', 'venv', 'logs', 'tmp', 'cache'}


def project_files(workspace, *, limit=10000):
    workspace = Path(workspace)
    if not workspace.is_dir() or workspace.is_symlink():
        return []
    result = []
    # Do not follow symlink directories, including links introduced by code.
    import os
    for base, dirs, files in os.walk(workspace, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in EXCLUDED
                         and not (Path(base) / d).is_symlink())
        for name in sorted(files):
            path = Path(base) / name
            if name.startswith('.') or name.lower().startswith(('credential', 'service.key')) or path.is_symlink():
                continue
            if path.is_file():
                result.append(path.relative_to(workspace).as_posix())
            if len(result) >= limit:
                return result
    return result


def project_summary(folder, state):
    names = project_files(Path(folder) / 'workspace', limit=10001)
    return {'file_count': len(names), 'available': True, 'status': state['status'],
            'selected_deliveries': selected_deliveries(state.get('progress', {}).get('deliveries', [])),
            'research_completed': bool(state.get('progress', {}).get('research_completed')),
            'files': names[:200], 'truncated': len(names) > 10000}


def report(state):
    progress = state.get('progress', {})
    lines = ['# Research report', '', '## Objective', '', state['definition']['goal']['objective'], '',
             '## Current outcome', '', f"Research status: **{state['status']}**.", '',
             (progress.get('last_review') or {}).get('summary', 'No scientific review has been completed yet.'), '',
             '## Research plan', '']
    for step in progress.get('plan', {}).get('steps', []):
        lines += [f"- [{'x' if step.get('status') == 'completed' else ' '}] {step['title']}",
                  f"  Acceptance: {step['acceptance']}"]
    lines += ['', '## Experiments and revisions', '']
    for cycle in state.get('cycles', []):
        lines += [f"### Iteration {cycle['index'] + 1} — {cycle.get('status', 'unknown')}", '',
                  cycle.get('summary', ''), '']
    lines += ['## Model and tool admission', '']
    reviews = progress.get('publication_reviews') or []
    if not reviews:
        lines += ['No independently approved model or tool is evidenced in this export.', '']
    for item in reviews:
        lines += [f"### {item.get('name', 'Candidate')} — {item.get('status', 'pending')}", '',
                  item.get('summary', ''), *['- ' + str(issue) for issue in item.get('issues', [])], '']
    lines += ['## Open questions and limitations', '']
    lines += ['- ' + str(q) for q in progress.get('questions', [])]
    if progress.get('delivery_pending'):
        lines += [str(progress['delivery_pending'])]
    lines += ['', 'This report describes saved evidence, including incomplete work. A downloadable project does not itself mean the research or public model admission passed.', '']
    return '\n'.join(lines)


def export_project(folder, state, *, include_working_files=False):
    """Export selected reviewed releases; full working files are an explicit option."""
    folder = Path(folder)
    workspace = folder / 'workspace'
    archive = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    checksums = {}
    progress = state.get('progress', {})
    deliveries = selected_deliveries(progress.get('deliveries', []))
    try:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as out:
            def write(name, content):
                data = content.encode('utf-8') if isinstance(content, str) else content
                out.writestr(name, data)
                checksums[name] = hashlib.sha256(data).hexdigest()

            write('README.md', '# Scientist research project\n\n' + state['definition']['goal']['objective'] + '\n\n'
                  '## Contents\n\n- `project/`: working files, included for drafts or when explicitly requested. Relative paths are preserved.\n'
                  '- `REPORT.md`: research outcome, iterations and outstanding limitations.\n'
                  '- `deliverables/`: selected reviewed models or methods, ready to use.\n'
                  '- `research/`: plan, recorded evaluation and admission evidence.\n'
                  '- `checksums.json`: SHA-256 for every exported file.\n\n'
                  '## Reproduce\n\nOpen a release in `deliverables/` and follow its README. A draft without admitted releases uses `project/`. '
                  'Use the supplied training, inference and test entry points. If these are absent, the project is incomplete; '
                  'the export does not invent execution commands or successful tests.\n\n'
                  f"Current research status: **{state['status']}**. See REPORT.md before using any result.\n")
            write('REPORT.md', report(state))
            data = {'status': state['status'], 'objective': state['definition']['goal']['objective'],
                    'budget': state['definition']['goal'], 'usage': state.get('usage', {}),
                    'assessment': state.get('assessment'), 'research_completed': progress.get('research_completed', False)}
            write('research/status.json', json.dumps(data, ensure_ascii=False, indent=2))
            for key in ['plan', 'development', 'publication_reviews', 'deliveries']:
                write(f'research/{key}.json', json.dumps(progress.get(key), ensure_ascii=False, indent=2))
            for delivery in deliveries:
                package = delivery.get('package') or delivery.get('model_package')
                expected = next((a['sha256'] for a in delivery.get('artifacts', []) if a['path'] == package), None)
                if not expected:
                    raise ValueError('Selected release has no verified package')
                with open_file(workspace, package) as source:
                    checksum = hashlib.file_digest(source, 'sha256').hexdigest()
                    if checksum != expected:
                        raise ValueError('Selected release package changed')
                    source.seek(0)
                    with zipfile.ZipFile(source) as release:
                        members = release.infolist()
                        if len(members) > 10000 or sum(m.file_size for m in members) > 1024**3:
                            raise ValueError('Selected release exceeds the export budget')
                        seen = set()
                        for member in members:
                            from utils.scoped_files import parts
                            segments = parts(member.filename.rstrip('/'))
                            if member.filename in seen or any(p in EXCLUDED or p.startswith('.') for p in segments) or stat.S_ISLNK(member.external_attr >> 16):
                                raise ValueError('Selected release contains an unsafe member')
                            seen.add(member.filename)
                            if member.is_dir():
                                continue
                            name = 'deliverables/' + delivery['candidate_id'][:16] + '/' + member.filename
                            checksum = hashlib.sha256()
                            with release.open(member) as src, out.open(name, 'w', force_zip64=True) as dest:
                                while chunk := src.read(1024 * 1024):
                                    dest.write(chunk); checksum.update(chunk)
                            checksums[name] = checksum.hexdigest()
            if include_working_files or not deliveries:
                names = project_files(workspace, limit=10001)
                if len(names) > 10000:
                    raise ValueError('Too many files for one project export')
                for name in names:
                    if name.startswith('deliveries/'):
                        continue  # Reviewed releases are already expanded above.
                    digest = hashlib.sha256()
                    with open_file(workspace, name) as source:
                        with out.open('project/' + name, 'w', force_zip64=True) as target:
                            while chunk := source.read(1024 * 1024):
                                digest.update(chunk)
                                target.write(chunk)
                    checksums['project/' + name] = digest.hexdigest()
            write('checksums.json', json.dumps(dict(checksums), indent=2))
        archive.seek(0)
        return archive
    except BaseException:
        archive.close()
        raise
