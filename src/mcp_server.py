"""Legacy entry point. Prefer python scripts/serve.py."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
if __name__ == "__main__":
    import runpy
    runpy.run_module("mcp_tools.mcp_server", run_name="__main__")
else:
    from importlib import import_module
    sys.modules[__name__] = import_module("mcp_tools.mcp_server")
