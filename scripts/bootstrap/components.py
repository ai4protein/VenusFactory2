"""Built-in capabilities and four optional installation families."""
from pathlib import Path

from .common import atomic_json, capture, inside


FAMILIES = ('external', 'factory', 'rsi', 'harnesses')
BUILTINS = {
    'database': {'module': 'mcp_tools.tools.database.tools_mcp', 'services': ['UniProt', 'PDB', 'NCBI', 'PubMed']},
    'web_search': {'module': 'mcp_tools.tools.search.tools_mcp', 'services': ['Tavily', 'SerpAPI'],
                   'settings': ['TAVILY_API_KEY', 'SERPAPI_API_KEY (or SERP_API_KEY)']},
    'public_protein': {'module': 'mcp_tools.tools.research.tools_mcp',
                       'services': ['BLAST', 'Foldseek', 'InterProScan', 'gget'],
                       'settings': ['EBI_EMAIL or NCBI_EMAIL for InterProScan']},
    'platform_help': {'module': 'mcp_tools.tools.platform_manual', 'services': ['manual', 'tool discovery']},
    'files_and_analysis': {'module': 'mcp_tools.tools.tools_agent_hub', 'services': ['session files', 'sandbox', 'basic analysis']},
}


def families(value):
    if value in (None, 'all'):
        return list(FAMILIES)
    if value == 'none':
        return []
    selected = list(dict.fromkeys(x.strip() for x in value.split(',') if x.strip()))
    if not selected or set(selected) - set(FAMILIES):
        raise ValueError('Components must be all, none, or external,factory,rsi,harnesses')
    return selected


def builtin_health(root, python):
    from mcp_tools.common.installation import gget_python
    missing = [item['module'] for item in BUILTINS.values()
               if not (root / (item['module'].replace('.', '/') + '.py')).is_file()]
    try:
        # Import package dependencies only: health checks do not submit public
        # queries, initialize a running MCP service or consume API credits.
        capture([python, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); import httpx,requests,Bio,fastmcp; import mcp_tools.tools.research.basic_queries; import mcp_tools.tools.search.web_provider', root / 'src'], cwd=root, timeout=90)
        capture([gget_python(root), '-c', 'from importlib.metadata import version; assert version("gget")=="0.30.8"; import gget'], timeout=90)
    except Exception:
        missing.append('builtin_dependencies')
    return {'status': 'failed' if missing else 'ready', 'missing': missing,
            'capabilities': list(BUILTINS), 'remote_credentials': 'configured separately; no network requests during checks'}


def install_builtins(root, env, python):
    from mcp_tools.common.installation import gget_python
    executable = gget_python(root)
    if executable.is_file():
        capture([executable, '-c', 'from importlib.metadata import version; assert version("gget")=="0.30.8"; import gget'], timeout=90)
    else:
        # Each built-in with incompatible dependencies may still have an isolated
        # environment; it is included in core installation, not a model plugin.
        executable = env.create(inside(root, 'runtime/local/builtin-tools/gget/env'), '3.12')
        env.pip(executable, 'gget==0.30.8')
    result = builtin_health(root, python)
    atomic_json(inside(root, 'runtime/local/components/builtins.json'), result)
    return result


def rsi_health(root, python, library=None):
    from mcp_tools.common.installation import scientist_library_root
    library = (root / Path(library).expanduser()).absolute() if library is not None else scientist_library_root(root)
    try:
        capture([python, '-c', 'import sys; sys.path.insert(0,sys.argv[1]); from mcp_tools.models.evolved.service import configured_library; from agent.rsi.scientist_model_library import ScientistModelLibrary', root / 'src'], cwd=root, timeout=90)
        healthy = library.is_dir()
    except Exception:
        healthy = False
    return {'status': 'ready' if healthy else 'missing', 'library_root': str(library),
            'review': 'existing publication and evidence checks remain required', 'models_installed_by_bootstrap': 0}


def install_rsi(root, python, library=None):
    from mcp_tools.common.installation import scientist_library_root
    if library is not None:
        target = (root / Path(library).expanduser()).resolve()
        if not target.is_dir():
            raise ValueError('Selected RSI library must already exist; use a reviewed ScientistModelLibrary, not a source-code folder')
        configured = __import__('os').environ.get('VENUS_SCIENTIST_MODEL_LIBRARY')
        if configured and Path(configured).expanduser().resolve() != target:
            raise ValueError('VENUS_SCIENTIST_MODEL_LIBRARY overrides this selection; update that explicit setting first')
    else:
        target = scientist_library_root(root)
        if not target.exists():
            inside(root, target).mkdir(parents=True, exist_ok=True)
    value = str(target.relative_to(root)) if target.is_relative_to(root) else str(target)
    atomic_json(inside(root, 'runtime/local/components/rsi.json'), {'library_root': value, 'auto_publish': False})
    return rsi_health(root, python)


def factory_backbones(root, manifest, preset):
    from .checks import factory_files
    buckets = {item['path'].split('/')[1] for item in factory_files(root, preset)
               if len(item['path'].split('/')) > 2 and not item['path'].startswith('ProteinMPNN/')}
    mapping = manifest['factory_backbones']
    unknown = buckets - set(mapping)
    if unknown:
        raise ValueError('Missing Factory backbone mapping: ' + ', '.join(sorted(unknown)))
    ids = [mapping[key] for key in sorted(buckets)]
    if any(item['path'].startswith('demo/') for item in factory_files(root, preset)):
        ids.append('facebook/esm2_t6_8M_UR50D')
    return list(dict.fromkeys(ids))
