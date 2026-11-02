"""Cache content-addressed frontend assets without touching API/SSE responses."""
import re
from pathlib import PurePosixPath
from starlette.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

class HashedAssets(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        # Only Vite's fingerprinted filenames are immutable, never user files or HTML.
        if response.status_code in (200, 304) and re.search(r"-[A-Za-z0-9_-]{8,}\.[a-z0-9]+$", PurePosixPath(path).name):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

def frontend_assets(directory):
    # Scope compression to assets. Never buffer chat event streams or downloads.
    return GZipMiddleware(HashedAssets(directory=str(directory)), minimum_size=1024, compresslevel=5)
