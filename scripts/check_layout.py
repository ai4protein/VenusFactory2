#!/usr/bin/env python3
"""Check canonical source boundaries without importing models or starting services."""
import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_MODULES = ('tools', 'src.tools', 'tool_runtime', 'web_v2', 'src.tool_runtime', 'src.web_v2', 'web', 'src.web', 'gradio', 'archive', 'mcp_tools.runtimes', 'mcp_tools.tool_runtime', 'mcp_tools.tools.train', 'mcp_tools.tools.mutation', 'mcp_tools.tools.denovo', 'mcp_tools.tools.external', 'mcp_tools.tools.evolved', 'mcp_tools.tools.admet', 'mcp_tools.runtime_config', 'mcp_tools.model_identity')


def harness_worker_import_errors(path, tree, root):
    """Keep shared boundaries and worker adapters independent of entry points."""
    relative = path.relative_to(root)
    if relative.parent.as_posix() != "src/agent/harness":
        return []
    protected = path.stem in {"events", "session_boundary"} or (
        path.stem.endswith("_worker") and path.stem != "web_worker"
    )
    if not protected:
        return []
    errors = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                package = ["agent", "harness"][: 3 - node.level]
                base = ".".join([*package, *([base] if base else [])])
            names = [base, *(base + "." + alias.name for alias in node.names)]
        for name in names:
            name = name.removeprefix("src.")
            if name.startswith("agent.harness.") and name.split(".")[2].endswith("_worker"):
                errors.append(
                    f"{relative}:{node.lineno}: shared contracts and worker adapters must not import "
                    "worker entry points; use harness.events or harness.session_boundary"
                )
                break
    return errors


def check(root=ROOT):
    errors = []
    for name in ('config', 'script', 'Scripts', 'epitope_miner', 'insulin_variants', 'src/tools', 'src/tool_runtime', 'src/web_v2', 'src/web', 'src/webui.py', 'src/frpc_hub.py', 'Scripts_notebook.ipynb', 'install_config.json', 'mcp_tools/runtimes', 'mcp_tools/tool_runtime', 'mcp_tools/tools/train', 'mcp_tools/tools/mutation', 'mcp_tools/tools/denovo', 'mcp_tools/tools/external', 'mcp_tools/tools/evolved', 'third_party', 'requirements-sandbox-cube.txt', 'requirements-sandbox-e2b.txt'):
        path = root / name
        if path.exists() or path.is_symlink():
            errors.append('Relocate duplicate, legacy or generated root content: ' + name)
    for path in (root / 'scripts/examples').rglob('*.sh'):
        text = path.read_text(encoding='utf-8')
        if 'src/tools/' in text or 'script/tools/' in text or '-m tools.' in text:
            errors.append(str(path.relative_to(root)) + ': stale tool entry point')
        for module in re.findall(r'\bpython\s+-m\s+([\w.]+)', text):
            target = root.joinpath(*module.split('.'))
            if not target.with_suffix('.py').is_file() and not (target / '__main__.py').is_file():
                errors.append(f'{path.relative_to(root)}: missing Python entry point {module}')
    for area in ('backend', 'mcp_tools', 'src/agent', 'scripts', 'tests', '.install'):
        for path in (root / area).rglob('*.py'):
            # Frozen historical audit capsules are hash-verified evidence, not live workers.
            if '__pycache__' in path.parts or 'vendor' in path.parts or any(p.startswith('_legacy') for p in path.parts):
                continue
            try:
                tree = ast.parse(path.read_text(encoding='utf-8'))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(str(path.relative_to(root)) + ': ' + str(exc))
                continue
            errors.extend(harness_worker_import_errors(path, tree, root))
            for node in ast.walk(tree):
                names = ([node.module] if isinstance(node, ast.ImportFrom) and not node.level
                         else [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                if any(name and any(name == old or name.startswith(old + '.') for old in OLD_MODULES) for name in names):
                    errors.append(f'{path.relative_to(root)}:{node.lineno}: use a canonical backend/mcp_tools import')
    example = root / 'configs/model_runtimes.example.json'
    data = json.loads(example.read_text(encoding='utf-8'))
    for name, entry in data.items():
        if entry.get('enabled') is not False:
            errors.append('Runtime templates must start disabled: ' + name)
        for key in ('python', 'root', 'executable', 'weights', 'installation', 'config', 'filters', 'advanced', 'model_dir', 'confidence_model_dir', 'cache', 'structure_weights'):
            if entry.get(key) and Path(entry[key]).is_absolute():
                errors.append('Template paths must be repository-relative: ' + name + '.' + key)
    return errors


if __name__ == '__main__':
    issues = check()
    if issues:
        raise SystemExit('\n'.join(issues))
    print('Canonical packages, configuration layout and relative runtime templates: OK')
