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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from web.utils.common_utils import get_save_path

# Mount all tools_api routers
from src.tools.mutation.tools_api import router as mutation_router
from src.tools.predict.tools_api import router as predict_router
from src.tools.search.tools_api import router as search_router
from src.tools.database.tools_api import router as database_router

UPLOAD_DIR = get_save_path("Fast_API", "Uploads")
OUTPUT_DIR = get_save_path("Fast_API", "Temp_Outputs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VenusFactory API",
    description="API for protein sequence and structure analysis tools (tools_api layer)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mutation_router)
app.include_router(predict_router)
app.include_router(search_router)
app.include_router(database_router)


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
        dirs_ok = UPLOAD_DIR.exists() and OUTPUT_DIR.exists()
        return create_response(
            success=True,
            data={
                "status": "healthy",
                "directories": {
                    "uploads": str(UPLOAD_DIR),
                    "outputs": str(OUTPUT_DIR),
                    "accessible": dirs_ok,
                },
            },
            message="Service is running",
        )
    except Exception as e:
        return create_response(success=False, error=f"Health check failed: {str(e)}")


@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    try:
        full_path = OUTPUT_DIR / file_path
        if not str(full_path.resolve()).startswith(str(OUTPUT_DIR.resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(
            path=str(full_path),
            filename=full_path.name,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.on_event("startup")
async def startup_event():
    logger.info("VenusFactory API server starting up...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("VenusFactory API server shutting down...")


if __name__ == "__main__":
    uvicorn.run(
        "src.api_server:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info",
    )
