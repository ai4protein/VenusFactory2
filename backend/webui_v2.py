import argparse
import os

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser(
        description="VenusFactory WebUI v2 entry (FastAPI + React)."
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Enable online mode (disable settings/custom-model features). Default is local mode.",
    )
    parser.add_argument("--host", type=str, default=os.getenv("WEBUI_V2_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("WEBUI_V2_PORT", "7861")))
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable frontend dev mode (proxy to Vite dev server).",
    )
    parser.add_argument(
        "--frontend-dev-url",
        type=str,
        default=os.getenv("WEBUI_V2_FRONTEND_DEV_URL", "http://127.0.0.1:5173"),
        help="Frontend dev server URL when --dev is enabled.",
    )
    parser.add_argument(
        "--frontend-dist",
        type=str,
        default=os.getenv("WEBUI_V2_FRONTEND_DIST", "frontend/dist"),
        help="Path to frontend production build directory.",
    )
    parser.add_argument("--runtime-root", default=os.getenv("VENUS_RUNTIME_ROOT"), help="Runtime data base; local/online subfolders are created automatically.")
    return parser.parse_args()


def main():
    args = parse_args()
    run_mode = "online" if args.online else "local"
    if args.runtime_root:
        os.environ["VENUS_RUNTIME_ROOT"] = args.runtime_root

    os.environ["WEBUI_V2_DEV_MODE"] = "1" if args.dev else "0"
    os.environ["WEBUI_V2_FRONTEND_DEV_URL"] = args.frontend_dev_url
    os.environ["WEBUI_V2_FRONTEND_DIST"] = args.frontend_dist
    os.environ["WEBUI_V2_MODE"] = run_mode

    uvicorn.run(
        "backend.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
        # Resolve trusted forwarding in the application, never in an earlier middleware.
        proxy_headers=False,
        timeout_graceful_shutdown=int(os.getenv("WEBUI_V2_SHUTDOWN_TIMEOUT", "15")),
        log_level=os.getenv("WEBUI_V2_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
