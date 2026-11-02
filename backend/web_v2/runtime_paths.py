"""Mode-specific runtime data layout, configured before importing API services.

Code, public datasets and pretrained weights remain shared. Mutable application
state belongs to one mode. Explicit per-service paths remain supported for
existing installations; the launcher can supply a separate root for each mode.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
AREAS = {
    'TEMP_OUTPUTS_DIR': 'files',
    'VENUSFACTORY_SESSION_DB_PATH': 'state/sessions.sqlite3',
    'VENUS_JOB_QUEUE_ROOT': 'queue',
    'VENUS_ACCESS_DB': 'state/access.sqlite3',
    'VENUS_REPORT_ROOT': 'reports',
    'VENUS_SCIENCE_GOAL_ROOT': 'scientist',
    'VENUS_AGENT_EVIDENCE_ROOT': 'agent',
    'VENUS_MODEL_PLATFORM_ROOT': 'models',
    'WEBUI_V2_ANALYTICS_DB_PATH': 'state/analytics.sqlite3',
    'VENUS_USAGE_DB': 'state/usage.sqlite3',
}


def runtime_root(mode=None, env=None):
    env = os.environ if env is None else env
    mode = (mode or env.get('WEBUI_V2_MODE', 'local')).strip().lower()
    if mode not in {'local', 'online'}:
        raise ValueError('Runtime mode must be local or online')
    from mcp_tools.models.profiles import repository_path
    base = repository_path(env.get('VENUS_RUNTIME_ROOT') or 'runtime', root=PROJECT)
    # The mode is always appended: setting one base cannot merge the two modes.
    return base / mode


def configure_runtime_paths(mode=None, env=None):
    env = os.environ if env is None else env
    root = runtime_root(mode, env)
    for key, area in AREAS.items():
        value = env.get(key, '').strip()
        from mcp_tools.models.profiles import repository_path
        env[key] = str(repository_path(value, root=PROJECT) if value else root / area)
    env.setdefault('MCP_HTTP_PORT', '8081' if root.name == 'online' else '8080')
    return root
