"""
API server: FastAPI app that mounts all tools_api routers (mutation, predict, search, database)
and provides health and file download. Started by webui or run directly.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

from web.utils.common_utils import (
    redact_path_text,
    ensure_within_roots,
    get_web_v2_root_dir,
    get_web_v2_area_dir,
)

# Mount all tools_api routers
try:
    # Works when importing as `src.api_server`
    from src.tools.mutation.tools_api import router as mutation_router
    from src.tools.predict.tools_api import router as predict_router
    from src.tools.search.tools_api import router as search_router
    from src.tools.database.tools_api import router as database_router
    from src.web_v2.chat_api import router as chat_v2_router
    from src.web_v2.report_api import router as report_v2_router
    from src.web_v2.quick_tools_api import router as quick_tools_v2_router
    from src.web_v2.custom_model_api import router as custom_model_v2_router
    from src.web_v2.advanced_tools_api import router as advanced_tools_v2_router
    from src.web_v2.download_api import router as download_v2_router
    from src.web_v2.settings_api import router as settings_v2_router
except ModuleNotFoundError:
    # Works when running from `python src/webui_v2.py` (sys.path rooted at src/)
    from tools.mutation.tools_api import router as mutation_router
    from tools.predict.tools_api import router as predict_router
    from tools.search.tools_api import router as search_router
    from tools.database.tools_api import router as database_router
    from web_v2.chat_api import router as chat_v2_router
    from web_v2.report_api import router as report_v2_router
    from web_v2.quick_tools_api import router as quick_tools_v2_router
    from web_v2.custom_model_api import router as custom_model_v2_router
    from web_v2.advanced_tools_api import router as advanced_tools_v2_router
    from web_v2.download_api import router as download_v2_router
    from web_v2.settings_api import router as settings_v2_router

WEB_V2_ROOT = get_web_v2_root_dir()
WEB_V2_UPLOAD_ROOT = get_web_v2_area_dir("uploads")
WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
WEB_V2_MANIFESTS_ROOT = get_web_v2_area_dir("manifests")
WEBUI_V2_MODE = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
if WEBUI_V2_MODE not in {"local", "online"}:
    WEBUI_V2_MODE = "local"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VenusFactory2 API server starting up...")
    logger.info("Web v2 storage roots initialized")
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

_cors_origins_raw = os.getenv(
    "WEBUI_V2_CORS_ORIGINS",
    "http://127.0.0.1:7861,http://localhost:7861,http://127.0.0.1:5173,http://localhost:5173",
)
_cors_origins = [origin.strip() for origin in _cors_origins_raw.split(",") if origin.strip()]
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
if WEBUI_V2_MODE == "local":
    app.include_router(custom_model_v2_router)
    app.include_router(settings_v2_router)

_frontend_dist = Path(os.getenv("WEBUI_V2_FRONTEND_DIST", "frontend/dist"))
if _frontend_dist.exists():
    app.mount(
        "/v2/assets",
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
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    request_id: str = Field(default_factory=lambda: str(uuid4()))


def create_response(
    success: bool,
    data: Any = None,
    message: str = None,
    error: str = None,
) -> Dict[str, Any]:
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


@app.get("/api/v2/runtime-config")
async def runtime_config():
    return {"mode": WEBUI_V2_MODE}


@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    try:
        candidate_roots = [WEB_V2_RESULTS_ROOT, WEB_V2_MANIFESTS_ROOT]
        full_path: Optional[Path] = None
        for root in candidate_roots:
            candidate = (root / file_path).resolve()
            if ensure_within_roots(candidate, [root]) and candidate.exists() and candidate.is_file():
                full_path = candidate
                break
        if full_path is None:
            raise HTTPException(status_code=403, detail="Access denied")
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {redact_path_text(str(e))}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@app.get("/v2", response_class=HTMLResponse)
async def webui_v2_entry():
    frontend_dev_mode = os.getenv("WEBUI_V2_DEV_MODE", "0") == "1"
    frontend_dev_url = os.getenv("WEBUI_V2_FRONTEND_DEV_URL", "http://127.0.0.1:5173")
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


@app.get("/v2/{full_path:path}", response_class=HTMLResponse)
async def webui_v2_spa_fallback(full_path: str):
    frontend_dev_mode = os.getenv("WEBUI_V2_DEV_MODE", "0") == "1"
    frontend_dev_url = os.getenv("WEBUI_V2_FRONTEND_DEV_URL", "http://127.0.0.1:5173")
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


# ---------- Exception handlers and lifecycle ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=create_response(success=False, error=exc.detail),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
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
