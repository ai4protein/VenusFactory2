#!/usr/bin/env python3
"""Canonical Linux service launcher. One codebase, separate local/online state."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT/'src', ROOT):
    sys.path.insert(0, str(path))


def arguments(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['local','online'], default='local')
    p.add_argument('--component', choices=['all','api','mcp','workers'], default='all')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int)
    p.add_argument('--mcp-port', type=int)
    p.add_argument('--api-workers', type=int, default=1)
    p.add_argument('--runtime-root', default=str(ROOT/'runtime'))
    p.add_argument('--env-file', default=str(ROOT/'.env'))
    p.add_argument('--frontend-dist', default=str(ROOT/'frontend/dist'))
    p.add_argument('--check', action='store_true', help='Print non-secret effective paths and exit')
    args=p.parse_args(argv)
    if not 1 <= args.api_workers <= 16: p.error('--api-workers must be 1–16')
    return args


def configure(args):
    from dotenv import load_dotenv
    from mcp_tools.models.profiles import repository_path
    for name in ('env_file', 'runtime_root', 'frontend_dist'):
        setattr(args, name, str(repository_path(getattr(args, name), root=ROOT)))
    load_dotenv(args.env_file, override=False)
    os.chdir(ROOT)
    os.environ.update(WEBUI_V2_MODE=args.mode, VENUS_RUNTIME_ROOT=args.runtime_root,
                      WEBUI_V2_FRONTEND_DIST=args.frontend_dist, WEBUI_V2_DEV_MODE='0',
                      MCP_HTTP_PORT=str(args.mcp_port or (8080 if args.mode=='local' else 8081)),
                      MCP_HTTP_HOST='127.0.0.1')
    from backend.web_v2.runtime_paths import configure_runtime_paths
    root=configure_runtime_paths()
    # Spawned API/tool processes use exactly the same import environment.
    os.environ['PYTHONPATH']=os.pathsep.join(dict.fromkeys(p for p in [*sys.path,*os.getenv('PYTHONPATH','').split(os.pathsep)] if p))
    return root


def workers():
    """Start fixed pools for queued configuration namespaces; no process per wait."""
    import time
    from backend.web_v2.job_queue import JobQueue
    from backend.web_v2.worker_pool import ensure_pool
    q=JobQueue()
    while True:
        with q.connection() as c:
            namespaces=[r[0] for r in c.execute("SELECT DISTINCT namespace FROM jobs WHERE status='queued' AND dispatch IS NOT NULL")]
        for namespace in namespaces:
            if namespace: ensure_pool(q, namespace, dict(os.environ))
        time.sleep(1)


def main(argv=None):
    args=arguments(argv);root=configure(args)
    if args.check:
        import json
        from backend.web_v2.runtime_paths import AREAS
        print(json.dumps({'mode':args.mode,'runtime':str(root),'port':args.port or (7861 if args.mode=='local' else 7863),
                          'mcp_port':int(os.environ['MCP_HTTP_PORT']),'api_workers':args.api_workers,
                          'paths':{k:os.environ[k] for k in AREAS}},indent=2))
        return
    if args.component=='workers': return workers()
    if args.component=='mcp':
        import runpy
        return runpy.run_module('mcp_tools.mcp_server',run_name='__main__')
    if not (Path(args.frontend_dist)/'index.html').is_file():
        raise SystemExit('Frontend is not built. Run npm ci and npm run build in frontend/.')
    from mcp_tools.scheduler.service import ensure_service
    ensure_service()
    import uvicorn
    uvicorn.run('backend.api_server:app',host=args.host,port=args.port or (7861 if args.mode=='local' else 7863),
                workers=args.api_workers,proxy_headers=False,timeout_graceful_shutdown=15)


if __name__=='__main__': main()
