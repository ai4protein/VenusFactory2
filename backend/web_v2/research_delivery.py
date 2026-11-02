"""Owner-only Scientist project export; never include worker control directories."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from utils.scoped_files import open_file

EXCLUDED = {'.git', '.env', '__pycache__', 'node_modules', '.venv', 'venv'}


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


def export_project(folder, state):
    """Preserve authored relative paths, stream large weights, and hash every member."""
    folder = Path(folder)
    workspace = folder / 'workspace'
    archive = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)
    checksums = {}
    progress = state.get('progress', {})
    try:
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as out:
            def write(name, content):
                data = content.encode('utf-8') if isinstance(content, str) else content
                out.writestr(name, data)
                checksums[name] = hashlib.sha256(data).hexdigest()

            write('README.md', '# Scientist research project\n\n' + state['definition']['goal']['objective'] + '\n\n'
                  '## Contents\n\n- `project/`: authored code, data, fitted models, tests and reports. Relative paths are preserved.\n'
                  '- `REPORT.md`: research outcome, iterations and outstanding limitations.\n'
                  '- `research/`: plan, recorded evaluation and admission evidence.\n'
                  '- `checksums.json`: SHA-256 for every exported file.\n\n'
                  '## Reproduce\n\nStart in `project/` and follow its README and dependency declarations when present. '
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
            names = project_files(workspace, limit=10001)
            if len(names) > 10000:
                raise ValueError('Too many files for one project export')
            for name in names:
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
