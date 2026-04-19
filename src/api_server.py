"""
API server: FastAPI app that mounts all tools_api routers (mutation, predict, search, database)
and provides health and file download. Started by webui or run directly.
"""
from __future__ import annotations

import importlib
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import get_config
from exceptions import RateLimitError, SessionNotFoundError, VenusFactoryError
from logger import get_logger, setup_logging
from web.utils.common_utils import (
    ensure_within_roots,
    get_web_v2_area_dir,
    get_web_v2_root_dir,
    redact_path_text,
)

setup_logging()
logger = get_logger(__name__)

# Unified import helper: resolves both `src.X` and `X` import paths.
def _import_attr(module_path: str, attr: str) -> Any:
    for prefix in ("src.", ""):
        try:
            mod = importlib.import_module(f"{prefix}{module_path}")
            return getattr(mod, attr)
        except (ModuleNotFoundError, AttributeError):
            continue
    raise ImportError(f"Cannot import '{attr}' from '{module_path}'")

mutation_router = _import_attr("tools.mutation.tools_api", "router")
predict_router = _import_attr("tools.predict.tools_api", "router")
search_router = _import_attr("tools.search.tools_api", "router")
database_router = _import_attr("tools.database.tools_api", "router")
chat_v2_router = _import_attr("web_v2.chat_api", "router")
report_v2_router = _import_attr("web_v2.report_api", "router")
quick_tools_v2_router = _import_attr("web_v2.quick_tools_api", "router")
custom_model_v2_router = _import_attr("web_v2.custom_model_api", "router")
advanced_tools_v2_router = _import_attr("web_v2.advanced_tools_api", "router")
download_v2_router = _import_attr("web_v2.download_api", "router")
settings_v2_router = _import_attr("web_v2.settings_api", "router")
workspace_v2_router = _import_attr("web_v2.workspace_api", "router")
analytics_store = _import_attr("web_v2.analytics_store", "analytics_store")

_cfg = get_config()

WEB_V2_ROOT = get_web_v2_root_dir()
WEB_V2_UPLOAD_ROOT = get_web_v2_area_dir("uploads")
WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
WEB_V2_MANIFESTS_ROOT = get_web_v2_area_dir("manifests")
WEBUI_V2_MODE = _cfg.server.mode

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VenusFactory2 API server starting up...")
    logger.info("Web v2 storage roots initialized")
    analytics_store.ensure_initialized()
    logger.info("Web v2 cleanup: disabled (no automatic deletion)")
    yield
    logger.info("VenusFactory2 API server shutting down...")


app = FastAPI(
    title="VenusFactory2 API",
    description="API for protein sequence and structure analysis tools (tools_api layer)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

_cors_origins = _cfg.server.cors_origins or [
    "http://127.0.0.1:7861", "http://localhost:7861",
    "http://127.0.0.1:5173", "http://localhost:5173",
]
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins or ["http://127.0.0.1:7861"],
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mutation_router)
app.include_router(predict_router)
app.include_router(search_router)
app.include_router(database_router)
app.include_router(chat_v2_router)
app.include_router(report_v2_router)
app.include_router(quick_tools_v2_router)
app.include_router(advanced_tools_v2_router)
app.include_router(download_v2_router)
app.include_router(workspace_v2_router)
app.include_router(settings_v2_router)
if WEBUI_V2_MODE == "local":
    app.include_router(custom_model_v2_router)

_frontend_dist = Path(_cfg.server.frontend_dist)
if _frontend_dist.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist / "assets")),
        name="webui_v2_assets",
    )

_img_dir = Path("img")
if _img_dir.exists():
    app.mount("/img", StaticFiles(directory=str(_img_dir)), name="img_static")

_manual_docs_dir = Path(__file__).resolve().parent.parent / "docs" / "manual"
if _manual_docs_dir.exists():
    app.mount("/manual-docs", StaticFiles(directory=str(_manual_docs_dir)), name="manual_docs_static")


# ---------- Shared models and helpers ----------
class StandardResponse(BaseModel):
    success: bool
    data: Any | None = None
    message: str | None = None
    error: str | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid4()))


def create_response(
    success: bool,
    data: Any = None,
    message: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "data": data,
        "message": message,
        "error": error,
        "timestamp": datetime.now().isoformat(),
        "request_id": str(uuid4()),
    }


# ---------- Health and download ----------
@app.get("/health", response_model=StandardResponse)
async def health_check():
    try:
        dirs_ok = WEB_V2_UPLOAD_ROOT.exists() and WEB_V2_RESULTS_ROOT.exists()
        return create_response(
            success=True,
            data={
                "status": "healthy",
                "mode": WEBUI_V2_MODE,
                "directories": {
                    "accessible": dirs_ok,
                    "upload_ready": WEB_V2_UPLOAD_ROOT.exists(),
                    "results_ready": WEB_V2_RESULTS_ROOT.exists(),
                    "manifests_ready": WEB_V2_MANIFESTS_ROOT.exists(),
                },
            },
            message="Service is running",
        )
    except Exception as e:
        return create_response(success=False, error=f"Health check failed: {str(e)}")


@app.get("/api/runtime-config")
async def runtime_config():
    return {"mode": WEBUI_V2_MODE}


_INLINE_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    ".bmp": "image/bmp", ".ico": "image/x-icon", ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    try:
        candidate_roots = [WEB_V2_RESULTS_ROOT, WEB_V2_MANIFESTS_ROOT]
        full_path: Path | None = None
        for root in candidate_roots:
            candidate = (root / file_path).resolve()
            if ensure_within_roots(candidate, [root]) and candidate.exists() and candidate.is_file():
                full_path = candidate
                break
        if full_path is None:
            raise HTTPException(status_code=403, detail="Access denied")
        ext = full_path.suffix.lower()
        media_type = _INLINE_MEDIA_TYPES.get(ext, "application/octet-stream")
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type=media_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {redact_path_text(str(e))}")
        raise HTTPException(status_code=500, detail="Failed to download file") from e


_STRUCTURE_EXTENSIONS = {".pdb", ".cif", ".mmcif", ".ent", ".mol", ".sdf", ".mol2"}

@app.get("/api/structure/content")
async def get_structure_content(path: str, download: int = 0):
    """Serve structure file content for inline Molstar visualization."""
    import os as _os

    project_root = Path(__file__).resolve().parent.parent
    temp_base = Path(_os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs"))
    if not temp_base.is_absolute():
        temp_base = project_root / temp_base
    temp_base = temp_base.resolve()

    candidate = (project_root / path).resolve()
    allowed_roots = [temp_base, WEB_V2_RESULTS_ROOT, WEB_V2_UPLOAD_ROOT]
    if not ensure_within_roots(candidate, allowed_roots):
        raise HTTPException(status_code=403, detail="Access denied")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if candidate.suffix.lower() not in _STRUCTURE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {candidate.suffix}")

    if download:
        return FileResponse(
            path=str(candidate),
            filename=candidate.name,
            media_type="application/octet-stream",
        )

    content = candidate.read_text(encoding="utf-8", errors="replace")
    return {
        "content": content,
        "filename": candidate.name,
        "format": candidate.suffix.lstrip(".").lower(),
    }


@app.get("/", response_class=HTMLResponse)
async def webui_v2_entry():
    frontend_dev_mode = _cfg.server.dev_mode
    frontend_dev_url = _cfg.server.frontend_dev_url
    dist_index = _frontend_dist / "index.html"

    if frontend_dev_mode:
        return HTMLResponse(
            content=(
                "<!doctype html><html><head><meta charset='utf-8'/>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
                "<title>VenusFactory2 v2</title>"
                f"<script>window.location.replace('{frontend_dev_url}');</script>"
                "</head><body>Redirecting to frontend dev server...</body></html>"
            )
        )
    if dist_index.exists():
        return HTMLResponse(content=dist_index.read_text(encoding="utf-8"))
    return HTMLResponse(
        status_code=503,
        content=(
            "WebUI v2 frontend not built. Run frontend build first, or start "
            "`python src/webui_v2.py --dev` with a Vite dev server."
        ),
    )


@app.get("/{full_path:path}", response_class=HTMLResponse)
async def webui_v2_spa_fallback(full_path: str):
    frontend_dev_mode = _cfg.server.dev_mode
    frontend_dev_url = _cfg.server.frontend_dev_url
    dist_index = _frontend_dist / "index.html"

    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not found")

    if frontend_dev_mode:
        return HTMLResponse(
            content=(
                "<!doctype html><html><head><meta charset='utf-8'/>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'/>"
                "<title>VenusFactory2 v2</title>"
                f"<script>window.location.replace('{frontend_dev_url}/{full_path}');</script>"
                "</head><body>Redirecting to frontend dev server...</body></html>"
            )
        )
    if dist_index.exists():
        return HTMLResponse(content=dist_index.read_text(encoding="utf-8"))
    raise HTTPException(status_code=404, detail="WebUI v2 route not available")


# ---------- Exception handlers ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=create_response(success=False, error=exc.detail),
    )


@app.exception_handler(RateLimitError)
async def rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    headers = {}
    if exc.retry_after is not None:
        headers["Retry-After"] = str(int(exc.retry_after))
    return JSONResponse(
        status_code=429,
        content=create_response(success=False, error=str(exc)),
        headers=headers,
    )


@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content=create_response(success=False, error=str(exc)),
    )


@app.exception_handler(VenusFactoryError)
async def venus_error_handler(request: Request, exc: VenusFactoryError) -> JSONResponse:
    logger.warning("VenusFactoryError: %s (context=%s)", exc, exc.context)
    return JSONResponse(
        status_code=400,
        content=create_response(success=False, error=str(exc)),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content=create_response(
            success=False,
            error=f"Internal server error: {type(exc).__name__}",
        ),
    )




if __name__ == "__main__":
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
    )
