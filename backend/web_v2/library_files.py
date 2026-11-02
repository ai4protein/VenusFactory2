"""Local Library artifacts; never expose conversation control or credential files."""
import json
import os
from pathlib import Path
import re

from mcp_tools.common.common_utils import get_web_v2_root_dir, get_temp_outputs_base_dir

HIDDEN_PARTS = {'__pycache__', 'node_modules', 'control', 'cache', 'logs'}
CONTROL_FILES = {'connection.json', 'request.json', 'state.json', 'session.json', 'session_state.json', 'worker.log'}


def visible_file(path, root):
    try:
        relative = path.relative_to(root)
        return (path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
                and not any(p.startswith('.') or p in HIDDEN_PARTS for p in relative.parts)
                and path.name not in CONTROL_FILES and path.suffix not in {'.pyc', '.log'})
    except (ValueError, OSError):
        return False


def area_files(area):
    # get_web_v2_area_dir('sessions') creates today's anonymous directory;
    # the Library must walk the session root across all dates and conversations.
    root = get_web_v2_root_dir() / area
    return [p for p in root.rglob('*') if visible_file(p, root)] if root.exists() else []


def agent_files():
    root = get_temp_outputs_base_dir() / 'agent'
    return [p for p in root.rglob('*') if visible_file(p, root)] if root.exists() else []


def agent_path(path):
    root = get_temp_outputs_base_dir() / 'agent'
    resolved = root / path.removeprefix('agent/')
    if not visible_file(resolved, root):
        raise ValueError('File not found')
    return resolved.resolve()


def science_root():
    from backend.web_v2.science_goals import service
    return service.state_root


def science_artifact_root(relative):
    """Only uploaded attachments and each goal's user-facing workspace."""
    parts = Path(relative).parts
    if len(parts) < 3 or not re.fullmatch('[a-f0-9]{64}', parts[0]):
        raise ValueError('Invalid research file')
    root = science_root()
    if parts[1] == 'uploads':
        return root / parts[0] / 'uploads'
    if len(parts) >= 4 and re.fullmatch('[a-f0-9]{32}', parts[1]) and parts[2] == 'workspace':
        return root / parts[0] / parts[1] / 'workspace'
    raise ValueError('Invalid research file')


def science_path(path):
    from backend.web_v2.research_files import safe_path
    relative = path.removeprefix('science/')
    resolved = safe_path(science_root(), relative)
    if not visible_file(resolved, science_artifact_root(relative)):
        raise ValueError('File not found')
    return resolved


def science_files():
    root = science_root()
    artifacts = []
    if not root.is_dir() or root.is_symlink():
        return artifacts
    for session in root.iterdir():
        if not session.is_dir() or session.is_symlink() or not re.fullmatch('[a-f0-9]{64}', session.name):
            continue
        scopes = [session / 'uploads']
        scopes.extend(p / 'workspace' for p in session.iterdir()
                      if p.is_dir() and not p.is_symlink() and re.fullmatch('[a-f0-9]{32}', p.name))
        for scope in scopes:
            if not scope.is_dir() or scope.is_symlink():
                continue
            for p in scope.rglob('*'):
                try:
                    relative = p.relative_to(root).as_posix()
                    artifacts.append((science_path('science/' + relative), 'science/' + relative))
                except (ValueError, OSError):
                    continue
    return artifacts


def report_exports():
    """Materialize only user-facing report artifacts, leaving historical runs intact."""
    from backend.web_v2.research_jobs import ResearchJobs
    root = ResearchJobs().root
    output = get_web_v2_root_dir() / 'results' / 'researcher'
    for state_file in root.glob('*/state.json'):
        if state_file.is_symlink() or not re.fullmatch('[a-f0-9]{32}', state_file.parent.name):
            continue
        try:
            if state_file.stat().st_size > 30_000_000:
                continue
            folder = output / state_file.parent.name
            marker = folder / '.source-version'
            version = str(state_file.stat().st_mtime_ns)
            if marker.exists() and marker.read_text() == version:
                continue
            state = json.loads(state_file.read_text())
            report = state.get('report')
            if not isinstance(report, str) or not report.strip():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            name = re.sub(r'[^\w .-]', '_', str((state.get('protein') or {}).get('name') or 'Research report'))[:100]
            data = {'sources': state.get('sources', []), 'sections': state.get('sections', []), 'remote_services': state.get('remote_services', [])}
            artifacts = {name + '.md': report, 'evidence.json': json.dumps(data, ensure_ascii=False, indent=2)}
            request_path = state_file.parent / 'request.json'
            if request_path.is_file() and not request_path.is_symlink():
                payload = json.loads(request_path.read_text()).get('payload', {})
                if payload.get('structure_pdb'): artifacts['input.pdb'] = payload['structure_pdb']
                elif payload.get('sequence') and not payload.get('accession'):
                    artifacts['input.fasta'] = '>input\n' + payload['sequence'] + '\n'
            for filename, content in artifacts.items():
                target = folder / filename
                temp = target.with_suffix(target.suffix + '.tmp')
                temp.write_text(content, encoding='utf-8'); temp.chmod(0o600); os.replace(temp, target)
                os.utime(target, (state_file.stat().st_mtime, state_file.stat().st_mtime))
            marker.write_text(version)
        except (OSError, ValueError, TypeError):
            continue
